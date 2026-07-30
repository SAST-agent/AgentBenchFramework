import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from agentbench_frame.eval.measurement import canonical_state_id
from agentbench_frame.games.miracle import replay_reading_v1 as replay
from agentbench_frame.games.miracle.research_protocol import (
    build_action_support,
    enumerate_legal_commands,
)


def observation(round_number):
    return {
        "round": round_number,
        "camp": 0,
        "map": {"units": [], "barracks": [-1] * 4, "miracles": [30, 30]},
        "players": [[[], 0, 0, [], []], [[], 0, 0, [], []]],
    }


def state(value):
    return {
        "canonical_state_id": canonical_state_id(value),
        "observation": value,
    }


def canonical_support(value):
    support = build_action_support(enumerate_legal_commands(value))
    return support, {
        "schema_version": support.schema_version,
        "support_id": support.support_id,
        "actions": [
            {"action_id": item.action_id, "command": item.action}
            for item in support.actions
        ],
    }


def artifact_fixture(tmp_path, *, frames=1, role="train"):
    root = tmp_path / "approved"
    replay_dir = root / "replays"
    replay_dir.mkdir(parents=True)
    case = {
        "case_id": "train-rank01-c0-m0-d0-r1",
        "opponent": "rank01",
        "evaluated_agent_camp": 0,
        "map_type": 0,
        "day_time": 0,
        "repeat": 1,
    }
    seeds = {"logic_seed": 101, "evaluated_agent_seed": 202, "opponent_seed": 303}
    policy = {"version": "candidate-v1", "source_sha256": "1" * 64}
    champion = {
        "logical_id": "rank01",
        "version": "champion-v1",
        "descriptor_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
    }
    identities = {
        "acting_policy_version": policy["version"],
        "policy_source_sha256": policy["source_sha256"],
        "champion_logical_id": champion["logical_id"],
        "champion_version": champion["version"],
        "champion_descriptor_sha256": champion["descriptor_sha256"],
        "champion_artifact_sha256": champion["artifact_sha256"],
    }
    seed_ref = hashlib.sha256(replay.canonical_replay_json_bytes(seeds)).hexdigest()
    observations = [observation(index) for index in range(1, frames + 2)]
    decision_frames = []
    for index in range(frames):
        support, supplied = canonical_support(observations[index])
        terminal = index == frames - 1
        decision_frames.append(
            {
                "decision_step": index + 1,
                "state_before": state(observations[index]),
                "action_support": supplied,
                "chosen_action": supplied["actions"][0],
                "acting_identity_refs": identities,
                "state_after": state(observations[index + 1]),
                "reward": 1.0 if terminal else 0.0,
                "outcome": "win" if terminal else "ongoing",
                "terminated": terminal,
                "truncated": False,
                "case_identity_ref": case["case_id"],
                "seed_bundle_ref": seed_ref,
                "rationale_status": "not_recorded",
            }
        )
    terminal = {
        "outcome": "win",
        "reward": 1.0,
        "termination_reason": "synthetic_rule",
        "terminated": True,
        "truncated": False,
    }
    replay_document = {
        "schema_version": replay.SYNTHETIC_REPLAY_SCHEMA_VERSION,
        "match_plan_sha256": "4" * 64,
        "case_identity": case,
        "role": role,
        "seeds": seeds,
        "acting_policy": policy,
        "champion": champion,
        "rationale_status": "not_recorded",
        "decision_count": frames,
        "terminal_recorded": True,
        "decision_frames": decision_frames,
        "terminal": terminal,
    }
    replay_path = replay_dir / "train.json"
    replay_path.write_bytes(replay.canonical_replay_json_bytes(replay_document))
    manifest = {
        "schema_version": replay.REPLAY_READING_MANIFEST_SCHEMA_VERSION,
        "replay_artifact_path": "replays/train.json",
        "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "match_plan_sha256": replay_document["match_plan_sha256"],
        "case_identity": case,
        "role": role,
        "seeds": seeds,
        "acting_policy": policy,
        "champion": champion,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(replay.canonical_replay_json_bytes(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return root, manifest_path, replay_path, manifest, replay_document, digest


def approve(monkeypatch, digest):
    monkeypatch.setattr(replay, "APPROVED_TRAINING_REPLAY_MANIFESTS", frozenset({digest}))


def approved_context(tmp_path, monkeypatch, *, frames=1):
    data = artifact_fixture(tmp_path, frames=frames)
    approve(monkeypatch, data[-1])
    return data, replay.preflight_replay_reading(data[1], approved_root=data[0])


def rewrite_document(path, value):
    path.write_bytes(replay.canonical_replay_json_bytes(value))


def test_context_is_issuer_only_and_object_new_forgery_cannot_open(tmp_path):
    with pytest.raises(TypeError):
        replay.ReplayReadingContext()
    forged = object.__new__(replay.ReplayReadingContext)
    with pytest.raises(ValueError, match="trusted|issued"):
        replay.open_replay_reading(forged)


def test_renderer_rejects_publicly_forged_validation_and_empty_packets(
    tmp_path, monkeypatch
):
    assert tuple(inspect.signature(replay.render_replay_timeline).parameters) == (
        "context",
    )
    _, context = approved_context(tmp_path, monkeypatch)
    packet = replay.open_replay_reading(context)
    for forged in (
        replace(packet, role="validation"),
        replace(packet, decision_frames=()),
    ):
        with pytest.raises(TypeError, match="context"):
            replay.render_replay_timeline(forged)


def test_renderer_rejects_object_new_context_forgery():
    forged = object.__new__(replay.ReplayReadingContext)
    with pytest.raises(ValueError, match="trusted|issued"):
        replay.render_replay_timeline(forged)


def test_context_has_no_mutable_packet_or_role_and_nested_view_is_frozen(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch)
    for name in ("packet", "role"):
        with pytest.raises((AttributeError, TypeError)):
            setattr(context, name, "forged")
    assert isinstance(context.manifest, MappingProxyType)
    with pytest.raises(TypeError):
        context.manifest["case_identity"]["case_id"] = "forged"
    packet = replay.open_replay_reading(context)
    assert packet.role == "train"


def test_open_rechecks_independent_approval_and_manifest_bytes(tmp_path, monkeypatch):
    (root, manifest_path, _, manifest, _, _), context = approved_context(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "APPROVED_TRAINING_REPLAY_MANIFESTS", frozenset())
    with pytest.raises(ValueError, match="approved"):
        replay.open_replay_reading(context)
    rewrite_document(manifest_path, {**manifest, "role": "validation"})
    with pytest.raises(ValueError, match="manifest|digest|bytes"):
        replay.open_replay_reading(context)
    assert root.exists()


def test_valid_chosen_action_replacement_still_breaks_replay_sha(tmp_path, monkeypatch):
    (root, _, replay_path, _, document, _), context = approved_context(tmp_path, monkeypatch)
    document["decision_frames"][0]["chosen_action"] = document["decision_frames"][0]["action_support"]["actions"][1]
    rewrite_document(replay_path, document)
    with pytest.raises(ValueError, match="SHA"):
        replay.open_replay_reading(context)
    assert root.exists()


def test_manifest_or_replay_independent_replacement_is_rejected(tmp_path, monkeypatch):
    (root, manifest_path, replay_path, manifest, document, _), context = approved_context(tmp_path, monkeypatch)
    replay_path.write_bytes(replay.canonical_replay_json_bytes({**document, "extra": 1}))
    with pytest.raises(ValueError):
        replay.open_replay_reading(context)
    rewrite_document(replay_path, document)
    rewrite_document(manifest_path, {**manifest, "replay_sha256": "f" * 64})
    with pytest.raises(ValueError):
        replay.open_replay_reading(context)
    assert root.exists()


@pytest.mark.parametrize("role", ["validation", "test"])
def test_non_training_role_is_rejected_even_with_approved_filename(tmp_path, monkeypatch, role):
    root, manifest_path, _, _, _, digest = artifact_fixture(tmp_path, role=role)
    approve(monkeypatch, digest)
    with pytest.raises(ValueError, match="role=train"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


@pytest.mark.parametrize("target", ["state", "action", "terminal", "identity"])
def test_deep_tampering_fails_after_rebinding_outer_digests(tmp_path, monkeypatch, target):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(tmp_path)
    frame = document["decision_frames"][0]
    if target == "state":
        frame["state_before"]["canonical_state_id"] = "0" * 64
    elif target == "action":
        frame["chosen_action"]["action_id"] = "0" * 64
    elif target == "terminal":
        document["terminal"]["outcome"] = "loss"
    else:
        frame["acting_identity_refs"]["policy_source_sha256"] = "f" * 64
    rewrite_document(replay_path, document)
    manifest["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    rewrite_document(manifest_path, manifest)
    approve(monkeypatch, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


@pytest.mark.parametrize("replacement", [0.0, False], ids=["float-zero", "bool-false"])
def test_action_support_rejects_weak_json_types_after_rebinding_digests(
    tmp_path, monkeypatch, replacement
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    frame = document["decision_frames"][0]
    unselected = frame["action_support"]["actions"][1]
    assert unselected != frame["chosen_action"]
    assert unselected["command"]["player"] == 0
    unselected["command"]["player"] = replacement
    rewrite_document(replay_path, document)
    manifest["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    rewrite_document(manifest_path, manifest)
    approve(monkeypatch, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="ActionSupport"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_frame_chain_and_terminal_consistency_are_strict(tmp_path, monkeypatch):
    for target in ("chain", "terminal"):
        root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(tmp_path / target, frames=2)
        if target == "chain":
            disconnected = copy.deepcopy(
                document["decision_frames"][1]["state_before"]["observation"]
            )
            disconnected["opaque_disconnected_marker"] = True
            document["decision_frames"][1]["state_before"] = state(disconnected)
        else:
            document["decision_frames"][-1]["reward"] = 0.0
        rewrite_document(replay_path, document)
        manifest["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
        rewrite_document(manifest_path, manifest)
        approve(monkeypatch, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        with pytest.raises(ValueError, match="continuous|terminal"):
            replay.preflight_replay_reading(manifest_path, approved_root=root)


@pytest.mark.parametrize("damage", ["bom", "truncated", "noncanonical", "nan"])
def test_noncanonical_or_nonstandard_json_fails_closed(tmp_path, monkeypatch, damage):
    root, manifest_path, _, _, _, _ = artifact_fixture(tmp_path)
    payload = manifest_path.read_bytes()
    if damage == "bom":
        payload = b"\xef\xbb\xbf" + payload
    elif damage == "truncated":
        payload = payload[:-2]
    elif damage == "noncanonical":
        payload = json.dumps(json.loads(payload), indent=2).encode() + b"\n"
    else:
        payload = payload.replace(b'"repeat":1', b'"repeat":NaN')
    manifest_path.write_bytes(payload)
    approve(monkeypatch, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="canonical|UTF-8|JSON"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


@pytest.mark.parametrize("unsafe", ["../escape.json", "/absolute.json", "C:/drive.json", "name:stream"])
def test_replay_path_escape_absolute_drive_and_ads_are_rejected(tmp_path, monkeypatch, unsafe):
    root, manifest_path, _, manifest, _, _ = artifact_fixture(tmp_path)
    manifest["replay_artifact_path"] = unsafe
    rewrite_document(manifest_path, manifest)
    approve(monkeypatch, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="path|escape|relative"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_symlink_component_is_rejected_without_reading_target(tmp_path, monkeypatch):
    root, manifest_path, replay_path, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    monkeypatch.setattr(replay, "_is_symlink", lambda path: path == replay_path)
    with pytest.raises(ValueError, match="symlink"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_timeline_is_stable_ordered_complete_and_has_no_invented_rationale(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch, frames=2)
    first = replay.render_replay_timeline(context)
    second = replay.render_replay_timeline(context)
    assert first == second
    assert first.index("step=1") < first.index("step=2") < first.index("terminal")
    for token in (
        "state=", "legal_actions=", "chosen=", "policy=", "champion=",
        "reward=", "outcome=", "case=", "logic_seed=",
        "evaluated_agent_seed=", "opponent_seed=",
        "rationale_status=not_recorded",
    ):
        assert token in first
    assert "because" not in first.lower()


@pytest.mark.parametrize("mutation", ["manifest", "approval", "replay"])
def test_renderer_reopens_and_revalidates_files_and_approval(
    tmp_path, monkeypatch, mutation
):
    (root, manifest_path, replay_path, manifest, document, _), context = (
        approved_context(tmp_path, monkeypatch)
    )
    if mutation == "manifest":
        rewrite_document(manifest_path, {**manifest, "role": "validation"})
    elif mutation == "approval":
        monkeypatch.setattr(replay, "APPROVED_TRAINING_REPLAY_MANIFESTS", frozenset())
    else:
        document["decision_frames"][0]["reward"] = 0.5
        rewrite_document(replay_path, document)
    with pytest.raises(ValueError, match="manifest|approved|SHA|changed"):
        replay.render_replay_timeline(context)
    assert root.exists()


def test_no_factories_workspace_judge_provider_policy_or_session_execution(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch)
    calls = []
    with pytest.raises(TypeError):
        replay.open_replay_reading(context, provider_factory=lambda: calls.append("provider"))
    replay.render_replay_timeline(context)
    assert calls == []
    assert not (tmp_path / "workspace").exists()
    source = inspect.getsource(replay)
    assert "subprocess" not in source
    assert "import Judge" not in source
    assert replay.APPROVED_TRAINING_REPLAY_MANIFESTS != {"fake-production-digest"}

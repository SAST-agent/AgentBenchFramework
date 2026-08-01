import copy
import gc
import hashlib
import inspect
import json
import os
import shutil
import stat
import weakref
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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


def rebind_and_approve(
    monkeypatch, manifest_path, replay_path, manifest, replay_document
):
    rewrite_document(replay_path, replay_document)
    manifest["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    rewrite_document(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    approve(monkeypatch, digest)
    return digest


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


def test_open_returns_deeply_immutable_validated_packet(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch)
    packet = replay.open_replay_reading(context)
    frame = packet.decision_frames[0]

    assert isinstance(packet.case_identity, MappingProxyType)
    assert isinstance(frame.state_before, MappingProxyType)
    assert isinstance(frame.state_before["observation"], MappingProxyType)
    assert isinstance(frame.action_support["actions"], tuple)
    assert isinstance(frame.chosen_action["command"], MappingProxyType)
    assert isinstance(packet.terminal, MappingProxyType)

    with pytest.raises(TypeError):
        packet.case_identity["case_id"] = "forged"
    with pytest.raises(TypeError):
        frame.state_before["observation"]["camp"] = 1
    with pytest.raises(TypeError):
        frame.action_support["actions"][0]["command"]["player"] = 1
    with pytest.raises(TypeError):
        frame.chosen_action["command"]["player"] = 1
    with pytest.raises(TypeError):
        packet.terminal["reward"] = 0.0
    with pytest.raises(AttributeError):
        frame.action_support["actions"].append("forged")

    assert replay.render_replay_timeline(context).endswith("\n")


def test_open_rechecks_independent_approval_and_manifest_bytes(tmp_path, monkeypatch):
    (root, manifest_path, _, manifest, _, _), context = approved_context(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "APPROVED_TRAINING_REPLAY_MANIFESTS", frozenset())
    with pytest.raises(ValueError, match="approved"):
        replay.open_replay_reading(context)
    rewrite_document(manifest_path, {**manifest, "role": "validation"})
    with pytest.raises(ValueError, match="manifest|digest|bytes"):
        replay.open_replay_reading(context)
    assert root.exists()


def test_unapproved_manifest_is_rejected_before_json_parsing(tmp_path, monkeypatch):
    root, manifest_path, _, _, _, _ = artifact_fixture(tmp_path)
    manifest_path.write_bytes(b'{' + b'"nested":[' * 2000 + b']' * 2000 + b'}')
    monkeypatch.setattr(replay, "APPROVED_TRAINING_REPLAY_MANIFESTS", frozenset())
    calls = []

    def reject_parser(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unapproved bytes reached JSON parsing")

    monkeypatch.setattr(replay, "_strict_object_bytes", reject_parser)
    with pytest.raises(ValueError, match="approved"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)
    assert calls == []


def test_approved_deep_manifest_recursion_fails_closed(tmp_path, monkeypatch):
    root, manifest_path, _, _, _, _ = artifact_fixture(tmp_path)
    payload = b'{"nested":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}'
    manifest_path.write_bytes(payload)
    approve(monkeypatch, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="canonical|JSON"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


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


@pytest.mark.parametrize(
    ("state_name", "camp"),
    [
        ("state_before", 1),
        ("state_after", 1),
        ("state_before", False),
        ("state_before", 0.0),
        ("state_before", "0"),
        ("state_after", False),
        ("state_after", 0.0),
        ("state_after", "0"),
    ],
    ids=[
        "before-other-camp",
        "after-other-camp",
        "before-bool",
        "before-float",
        "before-string",
        "after-bool",
        "after-float",
        "after-string",
    ],
)
def test_frame_perspective_camp_is_strictly_bound_after_digest_rebinding(
    tmp_path, monkeypatch, state_name, camp
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    frame = document["decision_frames"][0]
    changed = copy.deepcopy(frame[state_name]["observation"])
    changed["camp"] = camp
    frame[state_name] = state(changed)
    if state_name == "state_before" and type(camp) is int:
        _, supplied = canonical_support(changed)
        frame["action_support"] = supplied
        frame["chosen_action"] = supplied["actions"][0]
    rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    with pytest.raises(ValueError, match="camp"):
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


@pytest.mark.parametrize(
    ("frame_reward", "terminal_reward"),
    [
        (2**53, 2**53 + 1),
        (10**1000, 10**1000),
    ],
    ids=["adjacent-above-2**53", "overflowing-integer"],
)
def test_invalid_integer_rewards_fail_closed(
    tmp_path, monkeypatch, frame_reward, terminal_reward
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    document["decision_frames"][-1]["reward"] = frame_reward
    document["terminal"]["reward"] = terminal_reward
    rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    with pytest.raises(ValueError, match="reward"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_exact_large_integer_reward_is_normalized_once_for_frame_and_terminal(
    tmp_path, monkeypatch
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    document["decision_frames"][-1]["reward"] = 2**53
    document["terminal"]["reward"] = 2**53
    rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    context = replay.preflight_replay_reading(manifest_path, approved_root=root)
    packet = replay.open_replay_reading(context)
    assert type(packet.decision_frames[-1].reward) is float
    assert type(packet.terminal["reward"]) is float
    assert packet.decision_frames[-1].reward == packet.terminal["reward"]


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
    if os.name == "nt":
        original = replay._win_information
        def flagged(handle, label):
            information = original(handle, label)
            if label == "replay path":
                information.dwFileAttributes |= replay._FILE_ATTRIBUTE_REPARSE_POINT
            return information
        monkeypatch.setattr(replay, "_win_information", flagged)
    else:
        outside = tmp_path / "outside.json"
        outside.write_bytes(replay_path.read_bytes())
        replay_path.unlink()
        os.symlink(outside, replay_path)
    with pytest.raises(ValueError, match="reparse|unsafe"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)
@pytest.mark.parametrize("blocked", ["manifest", "replay"])
def test_preflight_reads_from_safe_open_handle_without_pathname_reopen(
    tmp_path, monkeypatch, blocked
):
    root, manifest_path, replay_path, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    blocked_path = manifest_path if blocked == "manifest" else replay_path
    original = Path.read_bytes
    def reject_pathname_reopen(path):
        if path == blocked_path:
            raise AssertionError(f"unsafe pathname reopen for {blocked}")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", reject_pathname_reopen)
    context = replay.preflight_replay_reading(manifest_path, approved_root=root)
    assert replay.open_replay_reading(context).role == "train"
    assert replay.render_replay_timeline(context).endswith("\n")
def _can_create_symlink(tmp_path):
    target = tmp_path / "symlink-probe-target"
    link = tmp_path / "symlink-probe-link"
    target.write_bytes(b"probe")
    try:
        os.symlink(target, link)
    except OSError:
        return False
    else:
        link.unlink()
        return True
def _simulate_windows_reparse(monkeypatch, label):
    original = replay._win_information
    def flagged(handle, current_label):
        information = original(handle, current_label)
        if current_label == label:
            information.dwFileAttributes |= replay._FILE_ATTRIBUTE_REPARSE_POINT
        return information
    monkeypatch.setattr(replay, "_win_information", flagged)
@pytest.mark.parametrize(
    ("target", "label", "component_index"),
    [
        ("manifest", "manifest path", 0),
        ("replay", "replay path", 1),
        ("replay-directory", "replay path", 0),
    ],
)
def test_safe_open_rejects_controlled_final_and_intermediate_reparse_races(
    tmp_path, monkeypatch, target, label, component_index
):
    root, manifest_path, replay_path, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    if not _can_create_symlink(tmp_path):
        assert os.name == "nt"
        _simulate_windows_reparse(monkeypatch, label)
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        def replace_component(current_label, relative, index):
            if current_label != label or index != component_index:
                return
            monkeypatch.setattr(replay, "_SAFE_OPEN_BARRIER", None)
            if target == "manifest":
                outside_file = outside / "manifest.json"
                outside_file.write_bytes(manifest_path.read_bytes())
                manifest_path.unlink()
                os.symlink(outside_file, manifest_path)
            elif target == "replay":
                outside_file = outside / "train.json"
                outside_file.write_bytes(replay_path.read_bytes())
                replay_path.unlink()
                os.symlink(outside_file, replay_path)
            else:
                outside_file = outside / "train.json"
                outside_file.write_bytes(replay_path.read_bytes())
                replay_dir = replay_path.parent
                replay_dir.rename(root / "replays-original")
                os.symlink(outside, replay_dir, target_is_directory=True)
        monkeypatch.setattr(replay, "_SAFE_OPEN_BARRIER", replace_component)
    with pytest.raises(ValueError, match="unsafe|reparse|escape"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)
@pytest.mark.parametrize("operation", ["open", "render"])
def test_post_preflight_reparse_replacement_is_rejected(
    tmp_path, monkeypatch, operation
):
    (root, _, replay_path, _, _, _), context = approved_context(
        tmp_path, monkeypatch
    )
    if not _can_create_symlink(tmp_path):
        assert os.name == "nt"
        _simulate_windows_reparse(monkeypatch, "replay path")
    else:
        outside = tmp_path / "outside-replay.json"
        outside.write_bytes(replay_path.read_bytes())
        replay_path.unlink()
        os.symlink(outside, replay_path)
    reader = (
        replay.open_replay_reading
        if operation == "open"
        else replay.render_replay_timeline
    )
    with pytest.raises(ValueError, match="unsafe|reparse|escape"):
        reader(context)
    assert root.exists()
def test_approved_root_identity_is_stable_across_context_reopen(
    tmp_path, monkeypatch
):
    (root, _, _, _, _, _), context = approved_context(tmp_path, monkeypatch)
    moved = tmp_path / "approved-original"
    root.rename(moved)
    shutil.copytree(moved, root)
    with pytest.raises(ValueError, match="root identity"):
        replay.open_replay_reading(context)
def test_special_file_type_is_rejected_before_payload_read(tmp_path, monkeypatch):
    root, manifest_path, replay_path, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    if os.name == "nt":
        original = replay._GetFileType
        calls = 0
        def report_pipe_for_replay(handle):
            nonlocal calls
            calls += 1
            return 3 if calls == 2 else original(handle)
        monkeypatch.setattr(replay, "_GetFileType", report_pipe_for_replay)
    else:
        replay_path.unlink()
        os.mkfifo(replay_path)
    with pytest.raises(ValueError, match="regular"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)
def test_safe_reader_has_no_checked_path_return_or_pathlib_reopen():
    source = inspect.getsource(replay)
    assert "_safe_relative_file" not in source
    assert ".read_bytes()" not in source
    assert "manifest_bytes = approved.read(" in source
    assert "replay_bytes = approved.read(" in source
    assert "MAX_REPLAY_MANIFEST_BYTES" in source
    assert "MAX_SYNTHETIC_REPLAY_BYTES" in source


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


def test_timeline_percent_encodes_all_dynamic_text_and_command_json(
    tmp_path, monkeypatch
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    case_id = "case\nforged=1 | tail\\\r"
    policy_version = "policy\r\n|\\=v"
    champion_id = "champion\n|\\=id"
    champion_version = "champion\r|\\=v"
    for container in (manifest, document):
        container["case_identity"]["case_id"] = case_id
        container["acting_policy"]["version"] = policy_version
        container["champion"]["logical_id"] = champion_id
        container["champion"]["version"] = champion_version
    frame = document["decision_frames"][0]
    frame["case_identity_ref"] = case_id
    frame["acting_identity_refs"]["acting_policy_version"] = policy_version
    frame["acting_identity_refs"]["champion_logical_id"] = champion_id
    frame["acting_identity_refs"]["champion_version"] = champion_version
    document["terminal"]["termination_reason"] = "terminal\nforged=1 | tail\\\r"
    rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    context = replay.preflight_replay_reading(manifest_path, approved_root=root)
    timeline = replay.render_replay_timeline(context)
    assert len(timeline.splitlines()) == 3
    assert timeline.count("\n") == 3
    assert "\r" not in timeline
    assert " | tail" not in timeline
    for escaped in ("%0A", "%0D", "%7C", "%5C", "%3D"):
        assert escaped in timeline
    chosen = next(line for line in timeline.splitlines() if line.startswith("step=1"))
    assert "chosen=" in chosen and ":%7B" in chosen
    assert '{"operation_type"' not in timeline
    assert "rationale_status=not_recorded" in timeline
    assert "because" not in timeline.lower()


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
def _install_fake_posix(monkeypatch, *, open_file, fstat, close, dup=None):
    monkeypatch.setattr(replay.os, "name", "posix")
    for name, value in (
        ("O_NOFOLLOW", 0x100),
        ("O_DIRECTORY", 0x200),
        ("O_CLOEXEC", 0x400),
        ("O_NONBLOCK", 0x800),
    ):
        monkeypatch.setattr(replay.os, name, value, raising=False)
    monkeypatch.setattr(replay.os, "open", open_file)
    monkeypatch.setattr(replay.os, "fstat", fstat)
    monkeypatch.setattr(replay.os, "close", close)
    monkeypatch.setattr(replay.os, "supports_dir_fd", {open_file})
    if dup is not None:
        monkeypatch.setattr(replay.os, "dup", dup)
class _FakePosixReader:
    def __init__(self, payload=b"data", *, fail=None, before=None, after=None, mode=stat.S_IFREG | 0o600, short=None, device=1):
        self.payload, self.fail, self.mode, self.short, self.device = payload, fail, mode, short, device
        self.before = len(payload) if before is None else before
        self.after = self.before if after is None else after
        self.open_calls, self.closed, self.read_calls = [], [], []
        self.offset, self.fstat_calls = 0, {}
    def _info(self, mode, size, ino):
        return SimpleNamespace(st_mode=mode, st_dev=self.device, st_ino=ino, st_size=size, st_mtime_ns=7, st_ctime_ns=9)
    def dup(self, fd):
        assert fd == 100
        return 101
    def open(self, path, flags, *, dir_fd=None):
        index = len(self.open_calls)
        if self.fail == f"open_{index}":
            raise OSError("synthetic no-follow open failure")
        fd = 102 + index
        self.open_calls.append((path, flags, dir_fd, fd))
        return fd
    def fstat(self, fd):
        count = self.fstat_calls.get(fd, 0)
        self.fstat_calls[fd] = count + 1
        if self.fail == f"fstat_{fd}" or (fd == 104 and self.fail == ("before" if count == 0 else "after")):
            raise OSError("synthetic fstat failure")
        if fd in (102, 103):
            return self._info(stat.S_IFDIR | 0o700, 0, fd)
        return self._info(self.mode, self.before if count == 0 else self.after, fd)
    def read(self, fd, requested):
        assert fd == 104
        self.read_calls.append(requested)
        if isinstance(self.fail, tuple) and self.fail[0] == "chunk":
            return self.fail[1]
        if self.fail == "read":
            raise OSError("synthetic read failure")
        size = min(requested, self.short or requested)
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk
    def install(self, monkeypatch):
        _install_fake_posix(monkeypatch, open_file=self.open, fstat=self.fstat, close=self.closed.append, dup=self.dup)
        monkeypatch.setattr(replay.os, "read", self.read)
    def run(self, monkeypatch, limit=32):
        self.install(monkeypatch)
        root = replay._ApprovedRoot("approved", ("posix", 1, 2), 100)
        return replay._read_relative_posix(root, ("one", "two", "file"), "one/two/file", "replay path", limit)
def test_fake_posix_success_flags_dirfd_chain_bounded_short_reads_and_close_order(monkeypatch):
    backend = _FakePosixReader(b"abcdef", short=2)
    assert backend.run(monkeypatch, 6) == b"abcdef"
    directory_flags = os.O_RDONLY | 0x100 | 0x200 | 0x400
    final_flags = os.O_RDONLY | 0x100 | 0x400 | 0x800
    assert backend.open_calls == [("one", directory_flags, 101, 102), ("two", directory_flags, 102, 103), ("file", final_flags, 103, 104)]
    assert backend.read_calls == [7, 5, 3, 1]
    assert backend.closed == [104, 103, 102, 101]
@pytest.mark.parametrize(
    ("failure", "closed"),
    [("open_1", [102, 101]), ("open_2", [103, 102, 101]), ("fstat_102", [102, 101]),
     ("fstat_103", [103, 102, 101]), ("before", [104, 103, 102, 101]),
     ("read", [104, 103, 102, 101]), ("after", [104, 103, 102, 101])],
)
def test_fake_posix_every_exception_closes_each_opened_fd_once(monkeypatch, failure, closed):
    backend = _FakePosixReader(fail=failure)
    with pytest.raises(ValueError, match="unsafe|unavailable"):
        backend.run(monkeypatch)
    assert backend.closed == closed and len(closed) == len(set(closed))
@pytest.mark.parametrize(
    ("payload", "before", "after", "limit", "passes"),
    [(b"1234", 4, 4, 5, True), (b"12345", 5, 5, 5, True), (b"123456", 1, 1, 5, False),
     (b"1234", 3, 4, 5, False), (b"123", 4, 3, 5, False)],
)
def test_fake_posix_bounded_read_and_growth_truncation(monkeypatch, payload, before, after, limit, passes):
    backend = _FakePosixReader(payload, before=before, after=after, short=1)
    if passes:
        assert backend.run(monkeypatch, limit) == payload
    else:
        with pytest.raises(ValueError, match="limit|changed"):
            backend.run(monkeypatch, limit)
    assert backend.closed == [104, 103, 102, 101]
@pytest.mark.parametrize(("kwargs", "reason", "closed"), [({"mode": stat.S_IFIFO}, "regular file", [104, 103, 102, 101]), ({"device": 2}, "device", [102, 101])])
def test_fake_posix_special_file_and_cross_device_fail_closed(monkeypatch, kwargs, reason, closed):
    backend = _FakePosixReader(**kwargs)
    with pytest.raises(ValueError, match=reason):
        backend.run(monkeypatch)
    assert backend.closed == closed
@pytest.mark.parametrize(("size", "passes"), [(31, True), (32, True), (33, False)])
def test_platform_safe_handle_enforces_limit_minus_exact_and_plus_one(tmp_path, size, passes):
    root, target = tmp_path / "approved", tmp_path / "approved" / "payload"
    root.mkdir(); target.write_bytes(b"x" * size)
    with replay._open_approved_root(root) as approved:
        if passes:
            assert approved.read("payload", "replay path", 32) == b"x" * size
        else:
            with pytest.raises(ValueError, match="limit"):
                approved.read("payload", "replay path", 32)
@pytest.mark.parametrize("stage", ["root", "intermediate"])
def test_fake_posix_fstat_failures_close_all_opened_fds_once(monkeypatch, stage):
    closed, opened = [], []
    opened_fd = 41 if stage == "root" else 52
    def open_file(*args, **_kwargs):
        opened.append(args); return opened_fd
    def failing_fstat(_fd):
        raise OSError("synthetic fstat failure")
    _install_fake_posix(monkeypatch, open_file=open_file, fstat=failing_fstat,
                        close=closed.append, dup=lambda _fd: 51)
    with pytest.raises(ValueError, match="approved replay root|unsafe|unavailable"):
        if stage == "root":
            replay._open_approved_root(Path("approved"))
        else:
            root = replay._ApprovedRoot("approved", ("posix", 1, 2), 50)
            replay._read_relative_posix(root, ("dir", "file"), "dir/file",
                                        "replay path", 1024)
    assert closed == ([41] if stage == "root" else [52, 51])
    if stage == "root":
        assert opened[0][1] == os.O_RDONLY | 0x100 | 0x200 | 0x400
@pytest.mark.parametrize(
    "identity",
    [("posix", True, 2), ("posix", 1, 2.0),
     ("posix", type("IntSubclass", (int,), {})(1), 2), ("posix", 1, 3),
     ("windows", True, 2, 3), ("posix", 1), ("unknown", 1, 2),
     ("posix", 1, 2, 3)],
)
def test_root_identity_rejects_schema_and_exact_type_drift(monkeypatch, identity):
    closed = []
    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=2)
    _install_fake_posix(monkeypatch, open_file=lambda *_a, **_k: 61,
                        fstat=lambda _fd: info, close=closed.append)
    with pytest.raises(ValueError, match="identity"):
        replay._open_approved_root(Path("approved"), identity)
    assert closed == [61]
def test_context_registry_rejects_complete_field_copy_and_old_sentinel(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch)
    forged = object.__new__(replay.ReplayReadingContext)
    for name in replay.ReplayReadingContext.__slots__:
        if name != "__weakref__":
            object.__setattr__(forged, name, getattr(context, name))
    with pytest.raises(ValueError, match="trusted issued"):
        replay.open_replay_reading(forged)
    assert not hasattr(replay, "_CONTEXT_ISSUER")
def test_context_registry_rejects_mutation_and_collected_identity_reuse(tmp_path, monkeypatch):
    _, context = approved_context(tmp_path, monkeypatch)
    snapshot = {name: getattr(context, name) for name in replay.ReplayReadingContext.__slots__
                if name != "__weakref__"}
    object.__setattr__(context, "_manifest_relative_path", "other.json")
    with pytest.raises(ValueError, match="trusted issued"):
        replay.open_replay_reading(context)
    reference = weakref.ref(context)
    del context
    gc.collect()
    assert reference() is None
    forged = object.__new__(replay.ReplayReadingContext)
    for name, value in snapshot.items():
        object.__setattr__(forged, name, value)
    with pytest.raises(ValueError, match="trusted issued"):
        replay.open_replay_reading(forged)
def test_frozen_replay_size_limits_are_finite_and_distinct():
    assert (replay.MAX_REPLAY_MANIFEST_BYTES, replay.MAX_SYNTHETIC_REPLAY_BYTES) == (256 * 1024, 16 * 1024 * 1024)
    assert replay.MAX_REPLAY_MANIFEST_BYTES < replay.MAX_SYNTHETIC_REPLAY_BYTES
@pytest.mark.parametrize(
    "relative",
    ["a//b", "a/./b", "./a", "a/.", "a/../b", "a/", "/a", "", "a\x00b", "C:/x", "name:stream", "a\\b"],
)
def test_relative_path_rejects_raw_unsafe_components(relative):
    with pytest.raises(ValueError, match="safe|relative|escape"):
        replay._relative_parts(relative, "replay path")
@pytest.mark.parametrize(
    ("lexical", "preserve_with_path"),
    [("replays/../manifest.json", True), ("./manifest.json", False)],
    ids=["parent", "current"],
)
def test_preflight_rejects_manifest_lexical_components_before_normalization(tmp_path, monkeypatch, lexical, preserve_with_path):
    root, manifest_path, _, _, _, digest = artifact_fixture(tmp_path)
    raw = str(root) + os.sep + lexical.replace("/", os.sep)
    assert Path(raw).is_file() and Path(raw).read_bytes() == manifest_path.read_bytes()
    approve(monkeypatch, digest)
    supplied = Path(raw) if preserve_with_path else raw
    with pytest.raises(ValueError, match="lexical"):
        replay.preflight_replay_reading(supplied, approved_root=root)
def test_preflight_accepts_normal_nested_manifest_path(tmp_path, monkeypatch):
    root, manifest_path, _, _, _, digest = artifact_fixture(tmp_path)
    nested = root / "nested" / "manifest.json"
    nested.parent.mkdir()
    shutil.copyfile(manifest_path, nested)
    approve(monkeypatch, digest)
    assert replay._relative_parts("nested/manifest.json", "manifest path") == ("nested", "manifest.json")
    assert replay.preflight_replay_reading(nested, approved_root=root).manifest
@pytest.mark.parametrize(
    "bad_chunk",
    [None, 1, object(), "x", bytearray(b"x"), memoryview(b"x"), type("BytesSubclass", (bytes,), {})(b"x")],
    ids=["none", "int", "object", "str", "bytearray", "memoryview", "bytes-subclass"],
)
def test_fake_posix_reader_rejects_non_exact_bytes_and_closes_fds(monkeypatch, bad_chunk):
    backend = _FakePosixReader(fail=("chunk", bad_chunk))
    with pytest.raises(ValueError, match="reader returned a non-bytes chunk"):
        backend.run(monkeypatch)
    assert backend.closed == [104, 103, 102, 101]
@pytest.mark.parametrize("attack", ["path-subclass", "flipping-path", "str-subclass", "pathlike", "root-subclass"])
def test_preflight_rejects_nonexact_path_inputs_before_path_protocol(tmp_path, monkeypatch, attack):
    root, manifest, _, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    concrete = type(Path())
    class TrackingPath(concrete):
        calls = 0
        def __fspath__(self):
            type(self).calls += 1
            return super().__fspath__()
    class FlippingPath(TrackingPath):
        values = (str(manifest), str(root / "replays" / ".." / "manifest.json"))
        def __fspath__(self):
            cls = type(self); value = cls.values[min(cls.calls, 1)]; cls.calls += 1
            return value
    class StringSubclass(str): pass
    class CustomPathLike(os.PathLike):
        calls = 0
        def __fspath__(self):
            type(self).calls += 1
            return str(manifest)
    manifest_input, approved_root, tracker = {
        "path-subclass": (TrackingPath(manifest), root, TrackingPath),
        "flipping-path": (FlippingPath(manifest), root, FlippingPath),
        "str-subclass": (StringSubclass(str(manifest)), root, None),
        "pathlike": (CustomPathLike(), root, CustomPathLike),
        "root-subclass": (manifest, TrackingPath(root), TrackingPath),
    }[attack]
    with pytest.raises(ValueError, match="exact|Path|string"):
        replay.preflight_replay_reading(manifest_input, approved_root=approved_root)
    if tracker is not None:
        assert tracker.calls == 0
@pytest.mark.parametrize("as_string", [False, True, "forward"], ids=["exact-path", "native-str", "forward-str"])
def test_preflight_accepts_exact_path_and_string_and_remains_reopenable(tmp_path, monkeypatch, as_string):
    root, manifest, _, _, _, digest = artifact_fixture(tmp_path)
    assert root.is_absolute() and manifest.is_absolute(); approve(monkeypatch, digest)
    supplied = str(manifest).replace("\\", "/") if as_string == "forward" else str(manifest) if as_string else manifest
    context = replay.preflight_replay_reading(supplied, approved_root=root)
    assert replay.open_replay_reading(context).role == "train"
    assert replay.render_replay_timeline(context).endswith("\n")
def test_preflight_binds_exact_path_representations_once(tmp_path, monkeypatch):
    root, manifest, _, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    original, calls = replay.os.fspath, {id(root): 0, id(manifest): 0}
    def recording(value):
        if id(value) in calls:
            calls[id(value)] += 1
        return original(value)
    monkeypatch.setattr(replay.os, "fspath", recording)
    context = replay.preflight_replay_reading(manifest, approved_root=root)
    assert calls == {id(root): 1, id(manifest): 1}
    assert replay.open_replay_reading(context).role == "train"
@pytest.mark.skipif(os.name != "nt", reason="Windows lexical separator contract")
@pytest.mark.parametrize("form", ["backslash-root", "slash-root", "namespace-mixed", "namespace-native"])
def test_preflight_rejects_windows_mixed_manifest_separators_before_open(tmp_path, monkeypatch, form):
    root, manifest, _, _, _, digest = artifact_fixture(tmp_path)
    nested = root / "mixed" / "manifest.json"; nested.parent.mkdir(); shutil.copyfile(manifest, nested)
    namespace = form.startswith("namespace"); raw_root = "\\\\?\\" + str(root) if namespace else str(root) if form == "backslash-root" else str(root).replace("\\", "/")
    supplied = "\\\\?\\" + (str(nested).replace("\\", "/") if form == "namespace-mixed" else str(nested)) if namespace else raw_root + ("\\mixed/manifest.json" if form == "slash-root" else "/mixed\\manifest.json")
    approved_root = Path("\\\\?\\" + str(root)) if namespace else root
    assert Path(supplied).is_file(); approve(monkeypatch, digest)
    monkeypatch.setattr(replay, "_read_and_validate", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("safe open reached")))
    with pytest.raises(ValueError, match="separator|namespace|lexical"):
        replay.preflight_replay_reading(supplied, approved_root=approved_root)
    for raw in (r"\\.\C:\approved\manifest.json", r"\\?\UNC\server\share\manifest.json", "//?/C:/approved/manifest.json", "//./C:/approved/manifest.json", "//?/uNc/server/share/manifest.json", r"\\server\share/path/manifest.json", r"\\server/share\path\manifest.json"):
        with pytest.raises(ValueError, match="namespace|separator"):
            replay._validate_lexical_filesystem_path(raw, "manifest path")
@pytest.mark.parametrize(("kind", "location"), [("str", "root"), ("str", "parent"), ("path", "root"), ("path", "parent")])
def test_preflight_rejects_relative_manifest_independently_of_cwd(tmp_path, monkeypatch, kind, location):
    root, _, _, _, _, digest = artifact_fixture(tmp_path); approve(monkeypatch, digest); monkeypatch.chdir(root if location == "root" else root.parent)
    supplied = "manifest.json" if kind == "str" else Path("manifest.json")
    with pytest.raises(ValueError, match="absolute"):
        replay.preflight_replay_reading(supplied, approved_root=root)
def test_preflight_rejects_relative_approved_root(tmp_path, monkeypatch):
    root, manifest, _, _, _, digest = artifact_fixture(tmp_path); approve(monkeypatch, digest); monkeypatch.chdir(root.parent)
    with pytest.raises(ValueError, match="absolute"):
        replay.preflight_replay_reading(manifest, approved_root=Path(root.name))

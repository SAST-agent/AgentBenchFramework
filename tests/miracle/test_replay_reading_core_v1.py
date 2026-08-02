import copy
import errno
import gc
import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import weakref
from contextlib import nullcontext
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


def replace_champion_identity(manifest, document, champion):
    for container in (manifest, document):
        container["champion"] = copy.deepcopy(champion)
    for frame in document["decision_frames"]:
        frame["acting_identity_refs"].update(
            {
                "champion_logical_id": champion["logical_id"],
                "champion_version": champion["version"],
                "champion_descriptor_sha256": champion["descriptor_sha256"],
                "champion_artifact_sha256": champion["artifact_sha256"],
            }
        )


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


def test_manifest_rejects_case_opponent_not_bound_to_champion(
    tmp_path, monkeypatch
):
    root, manifest_path, replay_path, manifest, document, _ = artifact_fixture(
        tmp_path
    )
    for container in (manifest, document):
        container["case_identity"]["opponent"] = "rank02"
    rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )

    with pytest.raises(
        ValueError,
        match="^case opponent does not match champion logical ID$",
    ):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_digest_rebound_opponent_attack_is_rejected_after_test_approval(
    tmp_path, monkeypatch
):
    root, manifest_path, replay_path, manifest, document, original_digest = (
        artifact_fixture(tmp_path)
    )
    assert manifest["case_identity"]["opponent"] == "rank01"
    for container in (manifest, document):
        container["case_identity"]["opponent"] = "attacker-opponent"
    rebound_digest = rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    assert rebound_digest != original_digest
    assert rebound_digest in replay.APPROVED_TRAINING_REPLAY_MANIFESTS

    with pytest.raises(
        ValueError,
        match="^case opponent does not match champion logical ID$",
    ):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_digest_rebound_champion_replacement_is_rejected_after_test_approval(
    tmp_path, monkeypatch
):
    root, manifest_path, replay_path, manifest, document, original_digest = (
        artifact_fixture(tmp_path)
    )
    replacement = {
        "logical_id": "rank02",
        "version": "replacement-v2",
        "descriptor_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
    }
    replace_champion_identity(manifest, document, replacement)
    rebound_digest = rebind_and_approve(
        monkeypatch, manifest_path, replay_path, manifest, document
    )
    assert rebound_digest != original_digest
    assert manifest["replay_sha256"] == hashlib.sha256(
        replay_path.read_bytes()
    ).hexdigest()
    assert rebound_digest in replay.APPROVED_TRAINING_REPLAY_MANIFESTS

    with pytest.raises(
        ValueError,
        match="^case opponent does not match champion logical ID$",
    ):
        replay.preflight_replay_reading(manifest_path, approved_root=root)


def test_valid_opponent_champion_binding_is_rechecked_by_lifecycle_open(
    tmp_path, monkeypatch
):
    calls = []
    original = replay._validate_case_champion_binding

    def tracked(case, champion):
        calls.append((case["opponent"], champion["logical_id"]))
        return original(case, champion)

    monkeypatch.setattr(replay, "_validate_case_champion_binding", tracked)
    _, context = approved_context(tmp_path, monkeypatch)
    assert calls == [("rank01", "rank01"), ("rank01", "rank01")]

    calls.clear()
    packet = replay.open_replay_reading(context)
    assert packet.case_identity["opponent"] == packet.champion["logical_id"]
    assert calls == [("rank01", "rank01"), ("rank01", "rank01")]


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


@pytest.mark.parametrize(
    "damage", ["bom", "truncated", "noncanonical", "nan", "lone_surrogate"]
)
def test_noncanonical_or_nonstandard_json_fails_closed(tmp_path, monkeypatch, damage):
    root, manifest_path, _, _, _, _ = artifact_fixture(tmp_path)
    payload = manifest_path.read_bytes()
    if damage == "bom":
        payload = b"\xef\xbb\xbf" + payload
    elif damage == "truncated":
        payload = payload[:-2]
    elif damage == "noncanonical":
        payload = json.dumps(json.loads(payload), indent=2).encode() + b"\n"
    elif damage == "nan":
        payload = payload.replace(b'"repeat":1', b'"repeat":NaN')
    else:
        payload = payload.replace(b'"opponent":"rank01"', b'"opponent":"\\ud800"')
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


def test_posix_approved_root_rejects_intermediate_ancestor_symlink(monkeypatch):
    opened = []
    closed = []
    directory_flags = os.O_RDONLY | 0x100 | 0x200 | 0x400

    def open_component(path, flags, *, dir_fd=None):
        opened.append((path, flags, dir_fd))
        if path == "approved":
            raise OSError("synthetic ancestor symlink rejected by O_NOFOLLOW")
        return 71 + len(opened) - 1

    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=3, st_ino=9)
    _install_fake_posix(
        monkeypatch,
        open_file=open_component,
        fstat=lambda _fd: info,
        close=closed.append,
    )
    monkeypatch.setattr(
        replay.os.path, "abspath", lambda _path: "/trusted/approved/root"
    )

    with pytest.raises(ValueError, match="approved replay root.*unsafe|unavailable"):
        with replay._open_approved_root(Path("synthetic-root")):
            pass

    assert opened == [
        ("/", directory_flags, None),
        ("trusted", directory_flags, 71),
        ("approved", directory_flags, 72),
    ]
    assert closed == [72, 71]


def test_windows_approved_root_rejects_intermediate_ancestor_reparse(monkeypatch):
    root_path = r"C:\trusted\approved\root"
    opened = []
    closed = []

    def open_checked(path, *, directory, label):
        assert path == "C:\\"
        opened.append((None, path, directory, label))
        return path, _fake_windows_information(directory=True, file_id=1)

    def open_relative(parent_handle, component, *, directory, label):
        opened.append((parent_handle, component, directory, label))
        if component == "approved":
            raise ValueError(
                "approved replay root contains a reparse-point component"
            )
        handle = replay.ntpath.join(parent_handle, component)
        return handle, _fake_windows_information(
            directory=True, file_id=len(opened)
        )

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    monkeypatch.setattr(replay.ntpath, "abspath", lambda _path: root_path)
    monkeypatch.setattr(replay, "_win_open_checked", open_checked)
    monkeypatch.setattr(replay, "_win_open_relative_checked", open_relative)
    monkeypatch.setattr(replay, "_win_final_path", lambda handle, _label: handle)
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)

    with pytest.raises(ValueError, match="reparse-point component"):
        with replay._open_approved_root(Path("synthetic-root")):
            pass

    assert opened == [
        (None, "C:\\", True, "approved replay root"),
        ("C:\\", "trusted", True, "approved replay root"),
        (r"C:\trusted", "approved", True, "approved replay root"),
    ]
    assert closed == [r"C:\trusted", "C:\\"]


def test_posix_approved_root_rejects_cross_device_mount_ancestor(monkeypatch):
    opened = []
    closed = []

    def open_component(path, flags, *, dir_fd=None):
        fd = 71 + len(opened)
        opened.append((path, flags, dir_fd, fd))
        return fd

    def fstat(fd):
        device = 1 if fd in {71, 72} else 2
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=device, st_ino=fd)

    _install_fake_posix(
        monkeypatch,
        open_file=open_component,
        fstat=fstat,
        close=closed.append,
    )
    monkeypatch.setattr(
        replay,
        "_posix_open_component",
        lambda parent_fd, component, flags: open_component(
            component, flags, dir_fd=parent_fd
        ),
        raising=False,
    )
    monkeypatch.setattr(
        replay.os.path, "abspath", lambda _path: "/trusted/mount/root"
    )

    with pytest.raises(ValueError, match="mount|device|cross"):
        with replay._open_approved_root(Path("synthetic-root")):
            pass

    assert [call[0] for call in opened] == ["/", "trusted", "mount"]
    assert closed == [73, 72, 71]


def test_posix_approved_root_rejects_same_device_bind_mount(monkeypatch):
    opened = []
    closed = []

    def legacy_open(path, flags, *, dir_fd=None):
        fd = 81 + len(opened)
        opened.append((path, flags, dir_fd, fd))
        return fd

    def protected_open(parent_fd, component, flags):
        if component == "mount":
            raise OSError(errno.EXDEV, "openat2 rejected same-device bind mount")
        return legacy_open(component, flags, dir_fd=parent_fd)

    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=9)
    _install_fake_posix(
        monkeypatch,
        open_file=legacy_open,
        fstat=lambda _fd: info,
        close=closed.append,
    )
    monkeypatch.setattr(
        replay, "_posix_open_component", protected_open, raising=False
    )
    monkeypatch.setattr(
        replay.os.path, "abspath", lambda _path: "/trusted/mount/root"
    )

    with pytest.raises(ValueError, match="unsafe|mount|unavailable"):
        with replay._open_approved_root(Path("synthetic-root")):
            pass

    assert [call[0] for call in opened] == ["/", "trusted"]
    assert closed == [82, 81]


def test_posix_approved_root_fails_closed_without_mount_safe_primitive(monkeypatch):
    opened = []
    closed = []

    def legacy_open(path, _flags, *, dir_fd=None):
        fd = 91 + len(opened)
        opened.append((path, dir_fd, fd))
        return fd

    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=9)
    _install_fake_posix(
        monkeypatch,
        open_file=legacy_open,
        fstat=lambda _fd: info,
        close=closed.append,
    )
    monkeypatch.setattr(
        replay,
        "_posix_open_component",
        lambda *_args: (_ for _ in ()).throw(
            NotImplementedError("openat2 unavailable")
        ),
        raising=False,
    )
    monkeypatch.setattr(replay.os.path, "abspath", lambda _path: "/trusted/root")

    with pytest.raises(ValueError, match="cannot prove|unsafe|unavailable"):
        with replay._open_approved_root(Path("synthetic-root")):
            pass

    assert opened == [("/", None, 91)]
    assert closed == [91]


def test_linux_openat2_component_uses_no_xdev_no_symlinks_and_beneath(monkeypatch):
    captured = {}

    class FakeSyscall:
        restype = None

        def __call__(self, number, parent_fd, path, how_pointer, size):
            how = replay.ctypes.cast(
                how_pointer, replay.ctypes.POINTER(replay._OpenHow)
            ).contents
            captured.update(
                number=number.value,
                parent_fd=parent_fd.value,
                path=path.value,
                flags=how.flags,
                mode=how.mode,
                resolve=how.resolve,
                size=size.value,
            )
            return 123

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", False)
    monkeypatch.setattr(replay.platform, "system", lambda: "Linux")
    monkeypatch.setattr(replay.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        replay.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(syscall=FakeSyscall()),
    )

    assert replay._posix_open_component(17, "child", 0x701) == 123
    assert captured == {
        "number": 437,
        "parent_fd": 17,
        "path": b"child",
        "flags": 0x701,
        "mode": 0,
        "resolve": (
            replay._RESOLVE_NO_XDEV
            | replay._RESOLVE_NO_SYMLINKS
            | replay._RESOLVE_BENEATH
        ),
        "size": replay.ctypes.sizeof(replay._OpenHow),
    }


def test_linux_openat2_unknown_architecture_fails_closed(monkeypatch):
    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", False)
    monkeypatch.setattr(replay.platform, "system", lambda: "Linux")
    monkeypatch.setattr(replay.platform, "machine", lambda: "unverified-arch")
    with pytest.raises(NotImplementedError, match="unknown"):
        replay._posix_open_component(17, "child", 0x701)


def _fake_windows_information(*, directory, file_id, size=0, volume=5):
    timestamp = SimpleNamespace(dwHighDateTime=0, dwLowDateTime=0)
    return SimpleNamespace(
        dwFileAttributes=(replay._FILE_ATTRIBUTE_DIRECTORY if directory else 0),
        ftCreationTime=timestamp,
        ftLastWriteTime=timestamp,
        nFileSizeHigh=0,
        nFileSizeLow=size,
        nNumberOfLinks=1,
        dwVolumeSerialNumber=volume,
        nFileIndexHigh=0,
        nFileIndexLow=file_id,
    )


@pytest.mark.skipif(os.name != "posix", reason="requires a real POSIX interpreter")
def test_posix_import_does_not_access_native_win32_apis():
    script = textwrap.dedent(
        """
        import ctypes

        calls = []
        def forbidden(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("Win32 API accessed during POSIX import")

        ctypes.WinDLL = forbidden
        from agentbench_frame.games.miracle import replay_reading_v1
        assert calls == []
        assert replay_reading_v1.os.name == "posix"
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires a real POSIX interpreter")
def test_posix_exposes_all_pure_windows_constants_and_validation_helper():
    names = (
        "_INVALID_HANDLE_VALUE",
        "_GENERIC_READ",
        "_FILE_LIST_DIRECTORY",
        "_FILE_TRAVERSE",
        "_FILE_READ_ATTRIBUTES",
        "_SYNCHRONIZE",
        "_FILE_SHARE_READ",
        "_OPEN_EXISTING",
        "_FILE_ATTRIBUTE_DIRECTORY",
        "_FILE_ATTRIBUTE_REPARSE_POINT",
        "_FILE_FLAG_OPEN_REPARSE_POINT",
        "_FILE_FLAG_BACKUP_SEMANTICS",
        "_FILE_TYPE_DISK",
        "_OBJ_CASE_INSENSITIVE",
        "_FILE_DIRECTORY_FILE",
        "_FILE_SYNCHRONOUS_IO_NONALERT",
        "_FILE_NON_DIRECTORY_FILE",
        "_FILE_OPEN",
    )
    assert all(type(getattr(replay, name)) is int for name in names)
    replay._win_validate_opened_handle(
        701,
        _fake_windows_information(directory=True, file_id=1),
        directory=True,
        label="synthetic directory",
    )


@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        (r"C:\approved\child", r"C:\approved", True),
        (r"D:\approved\child", r"C:\approved", False),
        (r"C:\root-evil\child", r"C:\root", False),
        (r"C:\approved\..\escape", r"C:\approved", False),
        (r"c:\APPROVED\Child", r"C:\approved", True),
        (r"\\server\share\approved\child", r"\\SERVER\SHARE\approved", True),
        (r"\\server\other\approved\child", r"\\server\share\approved", False),
    ],
    ids=(
        "drive-child",
        "different-drive",
        "prefix-collision",
        "parent-escape",
        "case-insensitive",
        "unc-child",
        "different-unc-share",
    ),
)
def test_windows_beneath_uses_windows_semantics_on_every_host(
    path, root, expected
):
    assert replay._win_is_beneath(path, root) is expected


def test_windows_manifest_location_uses_ntpath_on_every_host(monkeypatch):
    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    root, relative = replay._manifest_location(
        Path(r"C:\approved"),
        r"c:\APPROVED\nested\manifest.json",
    )
    assert replay.ntpath.normcase(str(root)) == r"c:\approved"
    assert relative == "nested/manifest.json"

    for escaped in (
        r"C:\approved-evil\manifest.json",
        r"D:\approved\manifest.json",
    ):
        with pytest.raises(ValueError, match="escapes"):
            replay._manifest_location(Path(r"C:\approved"), escaped)


@pytest.mark.parametrize("validation_fails", [False, True], ids=["success", "error"])
def test_synthetic_windows_checked_open_has_explicit_handle_ownership(
    monkeypatch, validation_fails
):
    closed = []
    information = _fake_windows_information(directory=True, file_id=1)
    monkeypatch.setattr(replay, "_CreateFileW", lambda *_args: 701, raising=False)
    monkeypatch.setattr(
        replay,
        "_win_information",
        lambda handle, _label: information,
    )

    def validate(*_args, **_kwargs):
        if validation_fails:
            raise ValueError("synthetic handle validation failure")

    monkeypatch.setattr(replay, "_win_validate_opened_handle", validate)
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)

    if validation_fails:
        with pytest.raises(ValueError, match="synthetic handle validation failure"):
            replay._win_open_checked(
                r"C:\approved", directory=True, label="approved replay root"
            )
        assert closed == [701]
    else:
        handle, returned = replay._win_open_checked(
            r"C:\approved", directory=True, label="approved replay root"
        )
        assert (handle, returned) == (701, information)
        assert closed == []
        replay._CloseHandle(handle)
        assert closed == [701]


@pytest.mark.parametrize(
    ("root_path", "anchor", "components"),
    [
        (
            r"C:\trusted\approved\root",
            "C:\\",
            ("trusted", "approved", "root"),
        ),
        (
            r"\\server\share\approved\root",
            "\\\\server\\share\\",
            ("approved", "root"),
        ),
    ],
    ids=["drive", "unc-share"],
)
def test_windows_approved_root_opens_components_relative_to_anchor_handles(
    monkeypatch, root_path, anchor, components
):
    closed = []
    relative_calls = []
    paths = {401: anchor}
    infos = {401: _fake_windows_information(directory=True, file_id=1)}
    current_path = anchor
    for offset, component in enumerate(components, start=1):
        current_path = replay.ntpath.join(current_path, component)
        handle = 401 + offset
        paths[handle] = current_path
        infos[handle] = _fake_windows_information(
            directory=True, file_id=offset + 1
        )

    def open_anchor(path, *, directory, label):
        assert path == anchor
        assert directory is True
        return 401, infos[401]

    def open_relative(parent_handle, component, *, directory, label):
        relative_calls.append((parent_handle, component, directory, label))
        handle = parent_handle + 1
        assert paths[handle] == replay.ntpath.join(paths[parent_handle], component)
        return handle, infos[handle]

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    monkeypatch.setattr(replay.ntpath, "abspath", lambda _path: root_path)
    monkeypatch.setattr(replay, "_win_open_checked", open_anchor)
    monkeypatch.setattr(
        replay, "_win_open_relative_checked", open_relative, raising=False
    )
    monkeypatch.setattr(
        replay, "_win_final_path", lambda handle, _label: paths[handle]
    )
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)

    with replay._open_approved_root(Path("synthetic-root")) as approved:
        assert approved.handle == 401 + len(components)
        assert approved.final_path == root_path

    assert relative_calls == [
        (401 + index, component, True, "approved replay root")
        for index, component in enumerate(components)
    ]
    final_handle = 401 + len(components)
    assert closed == list(range(final_handle - 1, 400, -1)) + [final_handle]


def _install_windows_root_revalidation_backend(
    monkeypatch,
    root_path,
    *,
    failure=None,
    failure_depth=0,
):
    drive, tail = replay.ntpath.splitdrive(root_path)
    anchor = drive + "\\"
    components = tuple(component for component in tail.split("\\") if component)
    retained_final_path = (
        root_path + "-retained" if failure == "final-path" else root_path
    )
    approved = replay._ApprovedRoot(
        root_path,
        ("windows", 5, 0, 111),
        111,
        retained_final_path,
    )
    root_handles = tuple(range(401, 402 + len(components)))
    component_handles = root_handles[1:]
    paths = {111: retained_final_path, 499: retained_final_path}
    current_path = anchor
    paths[root_handles[0]] = anchor
    for handle, component in zip(component_handles, components):
        current_path = replay.ntpath.join(current_path, component)
        paths[handle] = current_path
    target_path = retained_final_path + r"\payload"
    paths[800] = target_path
    root_infos = {
        handle: _fake_windows_information(
            directory=True,
            file_id=(
                999
                if failure == "identity" and handle == root_handles[-1]
                else (111 if handle == root_handles[-1] else handle)
            ),
        )
        for handle in root_handles
    }
    retained_info = _fake_windows_information(directory=True, file_id=111)
    target_info = _fake_windows_information(directory=False, file_id=800, size=1)
    anchor_calls = []
    root_relative_calls = []
    target_calls = []
    reads = []
    closed = []

    def open_checked(path, *, directory, label):
        assert directory is True
        assert label == "approved replay root"
        if path == root_path:
            return 499, retained_info
        assert path == anchor
        anchor_calls.append(path)
        return root_handles[0], root_infos[root_handles[0]]

    def open_relative(parent_handle, component, *, directory, label):
        if label == "approved replay root":
            index = len(root_relative_calls)
            handle = component_handles[index]
            root_relative_calls.append(
                (parent_handle, component, directory, label, handle)
            )
            if failure in {"reparse", "component"} and index == failure_depth:
                closed.append(handle)
                message = (
                    "approved replay root contains a reparse-point component"
                    if failure == "reparse"
                    else "approved replay root component is unsafe or unavailable"
                )
                raise ValueError(message)
            return handle, root_infos[handle]
        target_calls.append((parent_handle, component, directory, label))
        assert parent_handle == 111
        assert component == "payload"
        assert directory is False
        return 800, target_info

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    monkeypatch.setattr(replay, "_win_open_checked", open_checked)
    monkeypatch.setattr(
        replay, "_win_open_relative_checked", open_relative, raising=False
    )
    monkeypatch.setattr(
        replay,
        "_win_information",
        lambda handle, _label: {
            111: retained_info,
            499: retained_info,
            800: target_info,
        }[handle],
    )
    monkeypatch.setattr(replay, "_win_final_path", lambda handle, _label: paths[handle])
    monkeypatch.setattr(
        replay, "_win_read", lambda *_args: reads.append("read") or b"X"
    )
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)
    return approved, SimpleNamespace(
        anchor=anchor,
        components=components,
        root_handles=root_handles,
        anchor_calls=anchor_calls,
        root_relative_calls=root_relative_calls,
        target_calls=target_calls,
        reads=reads,
        closed=closed,
    )


@pytest.mark.parametrize(
    ("root_path", "failure_depth"),
    [
        (r"C:\trusted\approved\root", 0),
        (r"C:\trusted\approved\root", 2),
        (r"\\server\share\approved\root", 0),
        (r"\\server\share\approved\root", 1),
    ],
    ids=["drive-shallow", "drive-deep", "unc-shallow", "unc-deep"],
)
def test_windows_each_read_rejects_new_approved_root_ancestor_reparse(
    monkeypatch, root_path, failure_depth
):
    approved, backend = _install_windows_root_revalidation_backend(
        monkeypatch,
        root_path,
        failure="reparse",
        failure_depth=failure_depth,
    )

    with pytest.raises(ValueError, match="reparse-point component"):
        replay._read_relative_windows(
            approved, ("payload",), "payload", "replay path", 8
        )

    assert backend.anchor_calls == [backend.anchor]
    assert [call[1] for call in backend.root_relative_calls] == list(
        backend.components[: failure_depth + 1]
    )
    opened_handles = list(backend.root_handles[: failure_depth + 2])
    assert sorted(backend.closed) == sorted(opened_handles)
    assert len(backend.closed) == len(set(backend.closed))
    assert 111 not in backend.closed
    assert backend.target_calls == []
    assert backend.reads == []


@pytest.mark.parametrize(
    "root_path",
    [r"C:\trusted\approved\root", r"\\server\share\approved\root"],
    ids=["drive", "unc-share"],
)
def test_windows_each_read_revalidates_root_from_anchor_and_closes_temporaries(
    monkeypatch, root_path
):
    approved, backend = _install_windows_root_revalidation_backend(
        monkeypatch, root_path
    )

    assert replay._read_relative_windows(
        approved, ("payload",), "payload", "replay path", 8
    ) == b"X"

    assert backend.anchor_calls == [backend.anchor]
    assert backend.root_relative_calls == [
        (
            backend.root_handles[index],
            component,
            True,
            "approved replay root",
            backend.root_handles[index + 1],
        )
        for index, component in enumerate(backend.components)
    ]
    assert backend.target_calls == [(111, "payload", False, "replay path")]
    assert backend.reads == ["read"]
    expected_closed = list(backend.root_handles) + [800]
    assert sorted(backend.closed) == sorted(expected_closed)
    assert len(backend.closed) == len(set(backend.closed))
    assert 111 not in backend.closed
    assert approved.handle == 111


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("identity", "root identity"),
        ("final-path", "root path"),
        ("component", "unsafe or unavailable"),
    ],
)
def test_windows_root_revalidation_failure_closes_temporaries_before_target_open(
    monkeypatch, failure, message
):
    approved, backend = _install_windows_root_revalidation_backend(
        monkeypatch,
        r"C:\trusted\approved\root",
        failure=failure,
        failure_depth=1,
    )

    with pytest.raises(ValueError, match=message):
        replay._read_relative_windows(
            approved, ("payload",), "payload", "replay path", 8
        )

    expected_count = 3 if failure == "component" else len(backend.root_handles)
    assert len(backend.closed) == expected_count
    assert len(backend.closed) == len(set(backend.closed))
    assert 111 not in backend.closed
    assert approved.handle == 111
    assert backend.target_calls == []
    assert backend.reads == []


def test_windows_read_rejects_replaced_lexical_root_identity(monkeypatch):
    root_path = r"C:\trusted\approved\root"
    approved, backend = _install_windows_root_revalidation_backend(
        monkeypatch, root_path, failure="identity"
    )

    with pytest.raises(ValueError, match="root identity|root.*changed"):
        replay._read_relative_windows(
            approved, ("payload",), "payload", "replay path", 8
        )

    assert backend.reads == []
    assert backend.target_calls == []
    assert sorted(backend.closed) == sorted(backend.root_handles)
    assert len(backend.closed) == len(set(backend.closed))
    assert approved.handle == 111


@pytest.mark.parametrize(
    "root_path",
    [r"C:\trusted\approved\root", r"\\server\share\approved\root"],
    ids=["drive", "unc-share"],
)
def test_windows_read_opens_every_child_relative_to_approved_handles(
    monkeypatch, root_path
):
    approved = replay._ApprovedRoot(
        root_path,
        ("windows", 5, 0, 111),
        111,
        root_path,
    )
    closed = []
    root_relative_calls = []
    relative_calls = []
    root_info = _fake_windows_information(directory=True, file_id=111)
    directory_info = _fake_windows_information(directory=True, file_id=222)
    payload_info = _fake_windows_information(directory=False, file_id=333, size=1)
    drive, tail = replay.ntpath.splitdrive(root_path)
    anchor = drive + "\\"
    root_components = tuple(component for component in tail.split("\\") if component)
    root_handles = tuple(range(401, 402 + len(root_components)))
    child_paths = {
        111: root_path,
        222: root_path + r"\nested",
        333: root_path + r"\nested\payload",
    }
    root_infos = {}
    current_path = anchor
    child_paths[root_handles[0]] = anchor
    root_infos[root_handles[0]] = _fake_windows_information(
        directory=True, file_id=root_handles[0]
    )
    for handle, component in zip(root_handles[1:], root_components):
        current_path = replay.ntpath.join(current_path, component)
        child_paths[handle] = current_path
        root_infos[handle] = _fake_windows_information(
            directory=True,
            file_id=111 if handle == root_handles[-1] else handle,
        )

    def open_checked(path, *, directory, label):
        assert path == anchor
        assert directory is True
        return root_handles[0], root_infos[root_handles[0]]

    def open_relative(parent_handle, component, *, directory, label):
        if label == "approved replay root":
            index = len(root_relative_calls)
            handle = root_handles[index + 1]
            root_relative_calls.append((parent_handle, component, directory, label))
            return handle, root_infos[handle]
        relative_calls.append((parent_handle, component, directory, label))
        if component == "nested":
            assert parent_handle == 111
            return 222, directory_info
        assert component == "payload" and parent_handle == 222
        return 333, payload_info

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    monkeypatch.setattr(replay, "_win_open_checked", open_checked)
    monkeypatch.setattr(
        replay, "_win_open_relative_checked", open_relative, raising=False
    )
    monkeypatch.setattr(
        replay,
        "_win_information",
        lambda handle, _label: {
            111: root_info,
            222: directory_info,
            333: payload_info,
            **root_infos,
        }[handle],
    )
    monkeypatch.setattr(
        replay, "_win_final_path", lambda handle, _label: child_paths[handle]
    )
    monkeypatch.setattr(replay, "_win_read", lambda *_args: b"X")
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)

    assert replay._read_relative_windows(
        approved,
        ("nested", "payload"),
        "nested/payload",
        "replay path",
        8,
    ) == b"X"
    assert relative_calls == [
        (111, "nested", True, "replay path"),
        (222, "payload", False, "replay path"),
    ]
    assert root_relative_calls == [
        (
            root_handles[index],
            component,
            True,
            "approved replay root",
        )
        for index, component in enumerate(root_components)
    ]
    assert closed == (
        list(reversed(root_handles[:-1])) + [root_handles[-1], 333, 222]
    )
    assert 111 not in closed


def test_windows_relative_open_failure_closes_every_owned_handle_once(monkeypatch):
    root_path = r"C:\trusted\approved\root"
    approved = replay._ApprovedRoot(
        root_path,
        ("windows", 5, 0, 111),
        111,
        root_path,
    )
    root_info = _fake_windows_information(directory=True, file_id=111)
    directory_info = _fake_windows_information(directory=True, file_id=222)
    closed = []

    def open_relative(parent_handle, component, *, directory, label):
        if component == "nested":
            return 222, directory_info
        raise ValueError("synthetic relative reparse rejection")

    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", True)
    monkeypatch.setattr(
        replay,
        "_open_windows_root",
        lambda *_args, **_kwargs: replay._ApprovedRoot(
            root_path, ("windows", 5, 0, 111), 444, root_path
        ),
    )
    monkeypatch.setattr(
        replay, "_win_open_relative_checked", open_relative, raising=False
    )
    monkeypatch.setattr(
        replay,
        "_win_information",
        lambda handle, _label: root_info if handle in {111, 444} else directory_info,
    )
    monkeypatch.setattr(
        replay,
        "_win_final_path",
        lambda handle, _label: (
            root_path if handle in {111, 444} else root_path + r"\nested"
        ),
    )
    monkeypatch.setattr(replay, "_CloseHandle", closed.append, raising=False)

    with pytest.raises(ValueError, match="relative reparse"):
        replay._read_relative_windows(
            approved,
            ("nested", "payload"),
            "nested/payload",
            "replay path",
            8,
        )

    assert closed == [444, 222]


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
        container["case_identity"]["opponent"] = champion_id
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
    monkeypatch.setattr(replay, "_NATIVE_WINDOWS", False)
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
    monkeypatch.setattr(
        replay,
        "_posix_open_component",
        lambda parent_fd, component, flags: open_file(
            component, flags, dir_fd=parent_fd
        ),
    )
    monkeypatch.setattr(
        replay,
        "_reopen_current_posix_root",
        lambda root: nullcontext(root),
    )
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


def test_fake_posix_relative_read_rejects_same_device_bind_mount(monkeypatch):
    backend = _FakePosixReader()
    backend.install(monkeypatch)
    original = replay._posix_open_component

    def reject_bind_mount(parent_fd, component, flags):
        if component == "two":
            raise OSError(errno.EXDEV, "openat2 rejected same-device bind mount")
        return original(parent_fd, component, flags)

    monkeypatch.setattr(replay, "_posix_open_component", reject_bind_mount)
    root = replay._ApprovedRoot("approved", ("posix", 1, 2), 100)
    with pytest.raises(ValueError, match="unsafe|unavailable"):
        replay._read_relative_posix(
            root,
            ("one", "two", "file"),
            "one/two/file",
            "replay path",
            32,
        )
    assert backend.closed == [102, 101]


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
    closed, opened = [], []
    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=2)
    def open_component(*_args, **_kwargs):
        fd = 61 + len(opened)
        opened.append(fd)
        return fd
    _install_fake_posix(monkeypatch, open_file=open_component,
                        fstat=lambda _fd: info, close=closed.append)
    monkeypatch.setattr(replay.os.path, "abspath", lambda _path: "/approved")
    with pytest.raises(ValueError, match="identity"):
        replay._open_approved_root(Path("approved"), identity)
    assert opened == [61, 62]
    assert closed == [62, 61]


def test_fake_posix_relative_read_revalidates_current_lexical_root(monkeypatch):
    backend = _FakePosixReader()
    backend.install(monkeypatch)
    monkeypatch.setattr(
        replay,
        "_reopen_current_posix_root",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("approved replay root identity changed")
        ),
    )
    root = replay._ApprovedRoot("approved", ("posix", 1, 2), 100)

    with pytest.raises(ValueError, match="root identity"):
        replay._read_relative_posix(
            root,
            ("one", "two", "file"),
            "one/two/file",
            "replay path",
            32,
        )

    assert backend.open_calls == []
    assert backend.closed == []


def test_fake_posix_refreshed_root_and_relative_descriptors_close_once(monkeypatch):
    backend = _FakePosixReader(b"closed")
    backend.install(monkeypatch)
    retained = replay._ApprovedRoot("approved", ("posix", 1, 2), 200)
    refreshed = replay._ApprovedRoot("approved", ("posix", 1, 2), 100)
    monkeypatch.setattr(
        replay,
        "_reopen_current_posix_root",
        lambda _root: refreshed,
    )

    assert replay._read_relative_posix(
        retained,
        ("one", "two", "file"),
        "one/two/file",
        "replay path",
        32,
    ) == b"closed"

    assert refreshed.handle is None
    assert retained.handle == 200
    assert backend.closed == [104, 103, 102, 101, 100]


@pytest.mark.skipif(
    os.name != "posix" or replay.platform.system() != "Linux",
    reason="requires Linux openat2 and POSIX rename semantics",
)
@pytest.mark.parametrize("read_number", [1, 2], ids=["manifest", "replay"])
def test_posix_each_read_rejects_lexical_root_replacement_and_closes_descriptors(
    tmp_path, monkeypatch, read_number
):
    root, manifest_path, _, _, _, digest = artifact_fixture(tmp_path)
    approve(monkeypatch, digest)
    original_read = replay._read_relative_posix
    original_open_root = replay._open_approved_root
    moved = tmp_path / "approved-detached"
    reads = 0
    opened_roots = []

    def tracked_open_root(*args, **kwargs):
        approved = original_open_root(*args, **kwargs)
        opened_roots.append(approved)
        return approved

    def replace_before_read(approved, parts, relative, label, limit):
        nonlocal reads
        reads += 1
        if reads == read_number:
            root.rename(moved)
            shutil.copytree(moved, root)
            assert (moved / "manifest.json").read_bytes() == manifest_path.read_bytes()
            assert os.fstat(approved.handle).st_ino == approved.identity[2]
        return original_read(approved, parts, relative, label, limit)

    monkeypatch.setattr(replay, "_open_approved_root", tracked_open_root)
    monkeypatch.setattr(replay, "_read_relative_posix", replace_before_read)

    with pytest.raises(ValueError, match="root identity"):
        replay.preflight_replay_reading(manifest_path, approved_root=root)

    assert reads == read_number
    assert opened_roots
    assert all(approved.handle is None for approved in opened_roots)
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

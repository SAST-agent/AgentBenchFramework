import copy
import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from agentbench_frame.games.miracle import replay_reading_v1 as replay
from agentbench_frame.games.miracle import real_replay_approvals_v1 as approvals
from agentbench_frame.games.miracle import real_replay_adapter_v1 as real


H08_ROOT = Path(
    r"C:\Users\gongh\Documents\agentbench\skill_drafts"
    r"\24-miracle-read-replay\assets\h08-role-train-run-20260801"
)
H08_FINAL_MANIFEST = (
    Path(__file__).parents[2]
    / "docs"
    / "games"
    / "assets"
    / "24_miracle_h08_real_judge_manifest.v1.json"
)
H08_MANIFEST_SHA256 = (
    "7fe0f543abaee3e2e63b40fe2933dd541df6fc078fc635116262e55333e5ff8c"
)
H08_HUMAN_SKILL_SHA256 = (
    "cd16e9eec4c9549a8384debad8f5e6ab8bd7865dcdc83c1cf00dfdc061657e37"
)
H08_HUMAN_SKILL_FILE_SHA256 = (
    "30f542ec2e374b17cc43f5dff8d0175040c9907b3f1340406c2ac01eea5dcd8e"
)
H08_HUMAN_SKILL_CANDIDATE = (
    H08_FINAL_MANIFEST.parent
    / "24_miracle_h08_human_replay_skill.final-candidate.v1.json"
)
H08_PRODUCTION_PREFLIGHT = (
    H08_FINAL_MANIFEST.parent
    / "24_miracle_h08_production_replay_preflight.v1.json"
)
H08_HUMAN_SKILL_APPROVAL = (
    H08_FINAL_MANIFEST.parent
    / "24_miracle_h08_human_replay_skill_approval.v1.json"
)
HUMAN_SKILL_V4_BASE = (
    H08_ROOT.parent / "human-replay-skill.draft.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frame(observation):
    body = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
    return f"{len(body):06d}{body}"


def _observation(camp: int, round_number: int):
    return {
        "map": {"units": [], "barracks": [-1] * 4, "miracles": [30, 30]},
        "players": [[[], 0, 0, [], []], [[], 0, 0, [], []]],
        "round": round_number,
        "camp": camp,
    }


def _judge_observation(camp: int, round_number: int, state: int):
    return {
        "kind": "judge_frame",
        "goal": -1,
        "frame": {
            "state": state,
            "listen": [camp],
            "player": [camp],
            "content": [_frame(_observation(camp, round_number))],
        },
    }


def _operation(camp: int, name: str, operation_type: str, round_number: int, parameters):
    return {
        "kind": "ai_operation",
        "player": camp,
        "name": name,
        "operation": {
            "operation_parameters": parameters,
            "operation_type": operation_type,
            "player": camp,
            "round": round_number,
        },
    }


def _replay_bytes(*, map_type=0, day_time=0, corrupt=None):
    records = [
        (0, 11, 0, 1, 2, 3, 4),
        (0, 11, 1, 11, 12, 13, 14),
        (0, 1, 0, 0, 0, 0, 0),
        (2, 10, 1, 0, 0, 0, 0),
    ]
    integers = [0, 0, 0, map_type, day_time, 0, 0]
    for record in records:
        integers.extend(record)
    integers.append(-1)
    if corrupt == "header":
        integers[0] = 1
    elif corrupt == "event":
        integers[8] = 19
    elif corrupt == "game-end":
        integers[8 + 7 * 3] = 9
    elif corrupt == "trailer":
        integers[-1] = 0
    payload = struct.pack(f">{len(integers)}i", *integers)
    return payload[:-1] if corrupt == "alignment" else payload


def _trace_events():
    init0 = {
        "artifacts": ["HolyLight"],
        "creatures": ["Archer", "Swordsman", "BlackBat"],
    }
    events = [
        {
            "kind": "match_start",
            "players": ["miracle_ifelse", "rank02"],
            "replay": r"C:\capture\fixture.replay",
            "judge_dir": r"C:\judge",
        },
        {
            "kind": "judge_frame",
            "goal": -1,
            "frame": {
                "state": 1,
                "listen": [0],
                "player": [0],
                "content": [_frame({"camp": 0})],
            },
        },
        _operation(0, "miracle_ifelse", "init", 0, init0),
        {
            "kind": "judge_frame",
            "goal": -1,
            "frame": {
                "state": 2,
                "listen": [1],
                "player": [1],
                "content": [_frame({"camp": 1})],
            },
        },
        _operation(1, "rank02", "init", 0, None),
        _judge_observation(0, 0, 3),
        _operation(0, "miracle_ifelse", "endround", 0, {}),
        _judge_observation(1, 1, 4),
        _operation(1, "rank02", "endround", 1, None),
        _judge_observation(0, 2, 5),
        _operation(0, "miracle_ifelse", "endround", 2, {}),
        {"kind": "judge_frame", "goal": -1, "frame": {"state": 0, "time": 3, "length": 1024}},
        {
            "kind": "judge_frame",
            "goal": -1,
            "frame": {"state": -1, "end_info": '{"0": 10, "1": 20}'},
        },
        {"kind": "match_end", "end_info": '{"0": 10, "1": 20}'},
    ]
    return events


def _trace_bytes(events) -> bytes:
    return (
        "\n".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            for event in events
        )
        + "\n"
    ).encode("utf-8")


def _result_bytes():
    value = {
        "schema_version": 1,
        "tag": "fixture",
        "started_at": 1.0,
        "finished_at": 2.0,
        "duration_s": 1.0,
        "judge_dir_resolved": r"C:\judge",
        "p0": {"name": "miracle_ifelse", "dir": r"C:\p0"},
        "p1": {"name": "rank02", "dir": r"C:\p1"},
        "judge": {
            "pid": 10,
            "role": "judge",
            "started_at": 1.0,
            "natural_exit": True,
            "natural_returncode": 0,
            "termination_requested": True,
            "termination_reason": "match-end",
            "final_returncode": 0,
            "forced_kill": False,
            "cleanup_succeeded": True,
            "identity_confirmed": True,
        },
        "ai0": {
            "pid": 11,
            "role": "ai0",
            "started_at": 1.0,
            "natural_exit": True,
            "natural_returncode": 15,
            "termination_requested": True,
            "termination_reason": "match-end",
            "final_returncode": 15,
            "forced_kill": False,
            "cleanup_succeeded": True,
            "identity_confirmed": True,
        },
        "ai1": {
            "pid": 12,
            "role": "ai1",
            "started_at": 1.0,
            "natural_exit": True,
            "natural_returncode": 15,
            "termination_requested": True,
            "termination_reason": "match-end",
            "final_returncode": 15,
            "forced_kill": False,
            "cleanup_succeeded": True,
            "identity_confirmed": True,
        },
        "end_info_received": True,
        "end_info": {"0": 10, "1": 20},
        "scores": {"0": 10, "1": 20},
        "raw_winner": 1,
        "score_tie": False,
        "judge_tiebreak_applied": False,
        "timeout": {"ai0": False, "ai1": False},
        "ai_error": {"ai0": False, "ai1": False},
        "trace_path": r"C:\capture\fixture.jsonl",
        "replay_path": r"C:\capture\fixture.replay",
        "cleanup_all_succeeded": True,
        "exception": None,
        "run_match_returncode": 0,
    }
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _run_intent(players, *, role="train", evaluated_camp=0) -> bytes:
    value = {
        "schema_version": "24-miracle-h08-run-intent-v1",
        "created_at": "2026-08-01T00:00:00+00:00",
        "purpose": "fixture",
        "evidence_role": role,
        "case": "fixture-case",
        "evaluated_camp": evaluated_camp,
        "authority": {"basis": "test", "human_author": "test", "date": "2026-08-01"},
        "seed_policy": {
            "external_deterministic_seed_supported": False,
            "requirement": "Record realized map_type and day_time from the replay header; do not invent a seed",
        },
        "runtime": {
            "python": "3.13",
            "framework_git_head": "9" * 40,
            "framework_branch": "fixture",
            "framework_worktree_clean": True,
            "runner_sha256": "1" * 64,
        },
        "judge": {
            "path": r"C:\judge",
            "repository_git_head": "8" * 40,
            "subtree_tracked_clean": True,
            "file_count": 1,
            "tree_sha256": "2" * 64,
        },
        "players": players,
        "tree_digest_algorithm": "sha256(utf8(sorted relative-posix-path + TAB + file-sha256 + LF))",
        "production_approval_claimed": False,
    }
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _players():
    return [
        {
            "camp": 0,
            "name": "miracle_ifelse",
            "entry": "main.py",
            "path": r"C:\p0",
            "file_count": 1,
            "tree_sha256": "3" * 64,
        },
        {
            "camp": 1,
            "name": "rank02",
            "entry": "main.exe",
            "path": r"C:\p1",
            "file_count": 1,
            "tree_sha256": "4" * 64,
        },
    ]


def _manifest(run_intent, replay_bytes, trace_bytes, result_bytes, *, role="train", evaluated_camp=0):
    players = _players()
    return {
        "schema_version": real.REAL_JUDGE_MANIFEST_SCHEMA_VERSION,
        "role": role,
        "case_id": "fixture-case",
        "decision_scope": "evaluated_camp_only",
        "evaluated_camp": evaluated_camp,
        "players": [
            {key: player[key] for key in ("camp", "name", "entry", "tree_sha256")}
            for player in players
        ],
        "capture_identity": {
            "framework_git_head": "9" * 40,
            "framework_branch": "fixture",
            "runner_sha256": "1" * 64,
            "judge_repository_git_head": "8" * 40,
            "judge_tree_sha256": "2" * 64,
        },
        "adapter_identity": real.current_adapter_identity(),
        "run_intent": {"path": "run-intent.json", "bytes": len(run_intent), "sha256": _sha(run_intent)},
        "artifacts": {
            "replay": {"path": "fixture.replay", "bytes": len(replay_bytes), "sha256": _sha(replay_bytes)},
            "trace": {"path": "fixture.jsonl", "bytes": len(trace_bytes), "sha256": _sha(trace_bytes)},
            "result": {"path": "fixture.result.json", "bytes": len(result_bytes), "sha256": _sha(result_bytes)},
        },
        "environment": {
            "external_deterministic_seed_supported": False,
            "seed_status": "unsupported_not_recorded",
            "realized_map_type": 0,
            "realized_day_time": 0,
        },
        "expected_result": {
            "status": "complete",
            "scores": {"0": 10, "1": 20},
            "winner": 1,
            "timeout": False,
            "ai_error": False,
            "runner_exception": False,
            "cleanup_succeeded": True,
        },
        "expected_adapter_output": {
            "decision_frames": 3,
            "support_complete": 3,
            "chosen_in_support": 3,
            "opponent_protocol_compatible": False,
        },
    }


def candidate_fixture(tmp_path, *, role="train", evaluated_camp=0, events=None, replay_damage=None):
    manifest_root = tmp_path / "manifests"
    evidence_root = tmp_path / "evidence"
    manifest_root.mkdir()
    evidence_root.mkdir()
    players = _players()
    intent = _run_intent(players, role=role, evaluated_camp=evaluated_camp)
    replay_bytes = _replay_bytes(corrupt=replay_damage)
    trace_bytes = _trace_bytes(_trace_events() if events is None else events)
    result_bytes = _result_bytes()
    (evidence_root / "run-intent.json").write_bytes(intent)
    (evidence_root / "fixture.replay").write_bytes(replay_bytes)
    (evidence_root / "fixture.jsonl").write_bytes(trace_bytes)
    (evidence_root / "fixture.result.json").write_bytes(result_bytes)
    manifest = _manifest(
        intent,
        replay_bytes,
        trace_bytes,
        result_bytes,
        role=role,
        evaluated_camp=evaluated_camp,
    )
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_bytes(real.canonical_real_replay_json_bytes(manifest))
    return manifest_root, evidence_root, manifest_path, manifest


def _rewrite_manifest(path, manifest):
    path.write_bytes(real.canonical_real_replay_json_bytes(manifest))


def _rebind_trace(evidence_root, manifest_path, manifest, events):
    payload = _trace_bytes(events)
    (evidence_root / "fixture.jsonl").write_bytes(payload)
    manifest["artifacts"]["trace"].update(bytes=len(payload), sha256=_sha(payload))
    _rewrite_manifest(manifest_path, manifest)


def test_unapproved_real_manifest_is_rejected_before_json_parsing(tmp_path, monkeypatch):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    calls = []

    def forbidden_parser(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unapproved bytes reached JSON parsing")

    monkeypatch.setattr(real, "_parse_json_object", forbidden_parser)
    with pytest.raises(ValueError, match="independently approved"):
        real.preflight_real_judge_replay(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )
    assert calls == []
    assert approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS == frozenset(
        {H08_MANIFEST_SHA256}
    )


def test_independent_test_approval_preserves_identity_and_enables_issuer_flow(
    tmp_path, monkeypatch
):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    identity_before = real.current_adapter_identity()
    manifest_bytes_before = manifest_path.read_bytes()
    digest = _sha(manifest_bytes_before)

    monkeypatch.setattr(
        approvals,
        "APPROVED_REAL_JUDGE_REPLAY_MANIFESTS",
        frozenset({digest}),
    )
    assert real.current_adapter_identity() == identity_before
    assert manifest_path.read_bytes() == manifest_bytes_before
    assert _sha(manifest_path.read_bytes()) == digest

    context = real.preflight_real_judge_replay(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    )
    packet = real.open_real_judge_replay(context)
    assert packet.authoritative is True
    assert packet.evidence_scope == "approved_real_judge_training_replay"
    assert real.require_authoritative_real_replay_packet(packet) is packet

    candidate_packet = real.audit_real_judge_replay_candidate(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    ).packet
    assert candidate_packet.authoritative is False
    assert candidate_packet.evidence_scope == (
        "candidate_unapproved_real_judge_evidence"
    )
    with pytest.raises(ValueError, match="authoritative"):
        real.require_authoritative_real_replay_packet(candidate_packet)
    forged_packet = replace(
        candidate_packet,
        authoritative=True,
        evidence_scope="approved_real_judge_training_replay",
    )
    with pytest.raises(ValueError, match="authoritative"):
        real.require_authoritative_real_replay_packet(forged_packet)

    monkeypatch.setattr(
        approvals,
        "APPROVED_REAL_JUDGE_REPLAY_MANIFESTS",
        frozenset(),
    )
    with pytest.raises(ValueError, match="no longer approved"):
        real.open_real_judge_replay(context)
    with pytest.raises(ValueError, match="authoritative"):
        real.require_authoritative_real_replay_packet(packet)


def test_approved_context_reopen_rejects_input_replacement(tmp_path, monkeypatch):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    digest = _sha(manifest_path.read_bytes())
    monkeypatch.setattr(
        approvals,
        "APPROVED_REAL_JUDGE_REPLAY_MANIFESTS",
        frozenset({digest}),
    )
    context = real.preflight_real_judge_replay(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    )
    trace_path = evidence_root / "fixture.jsonl"
    trace_path.write_bytes(trace_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="trace.*(bytes|SHA)"):
        real.open_real_judge_replay(context)


@pytest.mark.parametrize("artifact", ["replay", "trace", "result"])
def test_any_artifact_digest_mismatch_fails_closed(tmp_path, artifact):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    path = {
        "replay": evidence_root / "fixture.replay",
        "trace": evidence_root / "fixture.jsonl",
        "result": evidence_root / "fixture.result.json",
    }[artifact]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match=f"{artifact}.*(bytes|SHA)"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "C:/drive", "name:stream", "a\\b"])
def test_artifact_paths_cannot_escape(tmp_path, unsafe):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    manifest["artifacts"]["trace"]["path"] = unsafe
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="path|relative|escape"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_symlink_or_reparse_artifact_is_rejected(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    original = evidence_root / "fixture.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="unsafe|unavailable|reparse|symbolic"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_input_replacement_during_safe_open_fails_closed(tmp_path, monkeypatch):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    target = evidence_root / "fixture.jsonl"
    replacement = evidence_root / "replacement.jsonl"
    replacement.write_bytes(target.read_bytes() + b"x")
    fired = False

    def replace(_label, _relative, component_index):
        nonlocal fired
        if not fired and _label == "trace path" and component_index == 0:
            fired = True
            os.replace(replacement, target)

    monkeypatch.setattr(replay, "_SAFE_OPEN_BARRIER", replace)
    with pytest.raises(ValueError, match="trace.*(bytes|SHA)|unsafe|unavailable"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )
    assert fired


@pytest.mark.parametrize("role", ["validation", "test"])
def test_only_train_role_is_eligible_for_candidate_or_production(tmp_path, role):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path, role=role)
    with pytest.raises(ValueError, match="role=train"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


@pytest.mark.parametrize("mutation", ["camp", "evaluated-name", "opponent-name", "judge", "framework"])
def test_camp_and_all_code_identities_are_bound(tmp_path, mutation):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    if mutation == "camp":
        manifest["evaluated_camp"] = 1
    elif mutation == "evaluated-name":
        manifest["players"][0]["name"] = "other"
    elif mutation == "opponent-name":
        manifest["players"][1]["name"] = "other"
    elif mutation == "judge":
        manifest["capture_identity"]["judge_tree_sha256"] = "f" * 64
    else:
        manifest["capture_identity"]["framework_git_head"] = "f" * 40
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="camp|identity|player|Judge|Framework|framework"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


@pytest.mark.parametrize("damage", ["header", "event", "game-end", "trailer", "alignment"])
def test_binary_replay_corruption_fails_closed(tmp_path, damage):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(
        tmp_path, replay_damage=damage
    )
    with pytest.raises(ValueError, match="replay|header|event|GameEnd|trailer|align"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


@pytest.mark.parametrize(
    "damage",
    ["utf8", "json", "order", "missing-observation", "duplicate-observation"],
)
def test_trace_encoding_json_order_and_observation_are_strict(tmp_path, damage):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    if damage == "utf8":
        payload = b"\xff\n"
        (evidence_root / "fixture.jsonl").write_bytes(payload)
        manifest["artifacts"]["trace"].update(bytes=len(payload), sha256=_sha(payload))
        _rewrite_manifest(manifest_path, manifest)
    elif damage == "json":
        payload = b'{"kind":}\n'
        (evidence_root / "fixture.jsonl").write_bytes(payload)
        manifest["artifacts"]["trace"].update(bytes=len(payload), sha256=_sha(payload))
        _rewrite_manifest(manifest_path, manifest)
    else:
        if damage == "order":
            events[0], events[1] = events[1], events[0]
        elif damage == "missing-observation":
            del events[5]
        else:
            events.insert(6, copy.deepcopy(events[5]))
        _rebind_trace(evidence_root, manifest_path, manifest, events)
    with pytest.raises(ValueError, match="UTF-8|JSON|order|match_start|observation"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_terminal_judge_frame_is_unique_and_final(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    events.insert(-1, copy.deepcopy(events[-2]))
    _rebind_trace(evidence_root, manifest_path, manifest, events)
    with pytest.raises(ValueError, match="terminal|order|duplicate"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_captured_windows_and_posix_path_names_are_platform_independent():
    assert real._captured_path_name(r"C:\capture\match.replay", "path") == (
        "match.replay"
    )
    assert real._captured_path_name("/capture/match.replay", "path") == (
        "match.replay"
    )


def test_chosen_action_outside_regenerated_support_fails_closed(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    events[6] = _operation(
        0,
        "miracle_ifelse",
        "move",
        0,
        {"mover": 999, "position": [0, 0, 0]},
    )
    _rebind_trace(evidence_root, manifest_path, manifest, events)
    with pytest.raises(ValueError, match="outside.*ActionSupport"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_incomplete_action_support_fails_closed(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    wind = _observation(0, 0)
    wind["players"][0] = [[[*range(0, 8)]], 99, 0, [], []]
    wind["players"][0][0][0] = [0, 3, 0, 1, 0, 0, 0, [0, 0, 0]]
    events[5]["frame"]["content"] = [_frame(wind)]
    _rebind_trace(evidence_root, manifest_path, manifest, events)
    with pytest.raises(ValueError, match="complete|WindBlessing|ActionSupport"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_terminal_transition_has_no_invented_state_after(tmp_path):
    manifest_root, evidence_root, manifest_path, _ = candidate_fixture(tmp_path)
    candidate = real.audit_real_judge_replay_candidate(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    )
    packet = candidate.packet
    assert candidate.authoritative is False
    assert len(packet.decision_frames) == 3
    assert packet.decision_frames[-1].state_after is None
    assert packet.decision_frames[-1].terminal_event["status"] == "observed_before_next_evaluated_observation"
    assert packet.decision_frames[-1].rationale_status == "not_recorded"
    first, second, terminal = packet.decision_frames
    assert first.state_after is None
    assert first.next_observed_state == second.state_before
    assert first.next_observation_trace_line == 6
    assert first.intervening_action_count == 1
    assert first.intervening_trace_lines == (5,)
    assert second.state_after is None
    assert second.next_observed_state == terminal.state_before
    assert second.next_observation_trace_line == 10
    assert second.transition_scope == (
        "next_observation_after_intervening_agent_actions"
    )
    assert second.intervening_action_count == 1
    assert second.intervening_trace_lines == (9,)
    assert terminal.state_after is None
    assert terminal.next_observed_state is None
    assert terminal.next_observation_trace_line is None
    assert terminal.intervening_action_count == 0
    assert packet.direct_state_after_count == 0
    assert packet.intervened_transition_count == 2
    assert packet.terminal_without_state_after_count == 1
    with pytest.raises(TypeError):
        packet.decision_frames[0].state_before["observation"]["camp"] = 1
    with pytest.raises(TypeError):
        packet.decision_frames[0].chosen_action["command"]["operation_type"] = "move"


def test_immediate_evaluated_observation_is_a_direct_state_after(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    immediate = events.pop(5)
    events.insert(3, immediate)
    _rebind_trace(evidence_root, manifest_path, manifest, events)
    packet = real.audit_real_judge_replay_candidate(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    ).packet
    frame = packet.decision_frames[0]
    assert frame.state_after == packet.decision_frames[1].state_before
    assert frame.next_observed_state is None
    assert frame.next_observation_trace_line == 4
    assert frame.transition_scope == "direct_until_next_evaluated_observation"
    assert frame.intervening_action_count == 0
    assert frame.intervening_trace_lines == ()


def test_multiple_opponent_actions_are_noncausal_and_line_bound(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    events = _trace_events()
    events[9:9] = [
        _judge_observation(1, 2, 40),
        _operation(1, "rank02", "endround", 2, None),
    ]
    _rebind_trace(evidence_root, manifest_path, manifest, events)
    packet = real.audit_real_judge_replay_candidate(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    ).packet
    frame = packet.decision_frames[1]
    assert frame.state_after is None
    assert frame.next_observed_state == packet.decision_frames[2].state_before
    assert frame.next_observation_trace_line == 12
    assert frame.intervening_action_count == 2
    assert frame.intervening_trace_lines == (9, 11)


def test_legacy_opponent_is_reported_and_excluded_without_polluting_evaluated_camp(tmp_path):
    manifest_root, evidence_root, manifest_path, manifest = candidate_fixture(tmp_path)
    candidate = real.audit_real_judge_replay_candidate(
        manifest_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
    )
    report = candidate.packet.opponent_compatibility
    assert report["compatible"] is False
    assert report["command_rejected"] == 2
    assert len(candidate.packet.decision_frames) == 3

    manifest["decision_scope"] = "all_camps"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="opponent|all.camp|incompatible"):
        real.audit_real_judge_replay_candidate(
            manifest_path,
            manifest_root=manifest_root,
            evidence_root=evidence_root,
        )


def test_h08_real_sample_produces_399_complete_evaluated_frames():
    assert H08_FINAL_MANIFEST.is_file()
    candidate = real.audit_real_judge_replay_candidate(
        H08_FINAL_MANIFEST,
        manifest_root=H08_FINAL_MANIFEST.parent,
        evidence_root=H08_ROOT,
    )
    packet = candidate.packet
    assert candidate.authoritative is False
    assert packet.role == "train"
    assert packet.evaluated_camp == 0
    assert len(packet.decision_frames) == 399
    assert packet.support_complete_count == 399
    assert packet.chosen_in_support_count == 399
    assert packet.authoritative is False
    assert packet.evidence_scope == "candidate_unapproved_real_judge_evidence"
    assert packet.direct_state_after_count == 348
    assert packet.intervened_transition_count == 50
    assert packet.terminal_without_state_after_count == 1
    assert packet.decision_frames[-1].state_after is None
    assert sum(
        frame.intervening_action_count for frame in packet.decision_frames
    ) == 1510
    assert packet.rationale_status == "not_recorded"
    assert packet.opponent_compatibility == {
        "camp": 1,
        "name": "rank02",
        "operations": 1510,
        "command_canonical": 1460,
        "command_rejected": 50,
        "support_complete": 1460,
        "chosen_in_support": 403,
        "chosen_outside_support": 1057,
        "compatible": False,
        "handling": "excluded_from_evaluated_camp_only_packet",
    }


def test_h08_approved_production_flow_and_final_skill_candidate():
    from agentbench_frame.games import miracle
    from agentbench_frame.games.miracle import iteration_protocol

    manifest_bytes = H08_FINAL_MANIFEST.read_bytes()
    adapter_identity = real.current_adapter_identity()
    manifest = real._parse_json_object(
        manifest_bytes, "H08 final manifest", canonical=True
    )
    assert manifest["adapter_identity"] == adapter_identity
    assert _sha(manifest_bytes) == H08_MANIFEST_SHA256
    assert approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS == frozenset(
        {H08_MANIFEST_SHA256}
    )

    context = real.preflight_real_judge_replay(
        H08_FINAL_MANIFEST,
        manifest_root=H08_FINAL_MANIFEST.parent,
        evidence_root=H08_ROOT,
    )
    packet = real.open_real_judge_replay(context)
    assert real.require_authoritative_real_replay_packet(packet) is packet
    assert packet.authoritative is True
    assert packet.evidence_scope == "approved_real_judge_training_replay"
    assert packet.manifest_sha256 == H08_MANIFEST_SHA256
    assert packet.artifact_sha256 == {
        "replay": "6aacfb48ab3806633ebdb9d9c8d665cab703ccbcb62ca67be9336207a16a19e6",
        "trace": "e0e280338ae3b06818a923733fb4a04a3730d931ea800e144ecd0aadf4241948",
        "result": "71e1bca160b2289edbe18f6a7b67a63a154ada7a01b22d3b5d8e8176b3cf60b8",
        "run_intent": "c86530b7628ca4c0fb3da6259c36d8d0347d2c06e56a3db083220d6a1e082200",
    }
    assert len(packet.decision_frames) == 399
    assert packet.support_complete_count == 399
    assert packet.chosen_in_support_count == 399
    assert packet.direct_state_after_count == 348
    assert packet.intervened_transition_count == 50
    assert packet.terminal_without_state_after_count == 1
    assert sum(
        frame.intervening_action_count for frame in packet.decision_frames
    ) == 1510
    assert packet.opponent_compatibility["compatible"] is False
    assert {frame.rationale_status for frame in packet.decision_frames} == {
        "not_recorded"
    }

    audit = real._parse_json_object(
        H08_PRODUCTION_PREFLIGHT.read_bytes(),
        "H08 production replay preflight",
        canonical=True,
    )
    assert audit["approval"]["manifest_sha256"] == H08_MANIFEST_SHA256
    assert audit["lifecycle"] == {
        "gate": "passed",
        "open": "passed",
        "packet_authoritative": True,
        "preflight": "passed",
        "preflight_context": "issuer_only",
    }
    assert audit["statistics"] == {
        "action_support_complete": 399,
        "chosen_in_support": 399,
        "decision_frames": 399,
        "direct_state_after": 348,
        "intervened_observation_intervals": 50,
        "intervening_opponent_operations": 1510,
        "terminal_without_state_after": 1,
    }

    skill = iteration_protocol.HumanReplaySkill.from_bytes(
        H08_HUMAN_SKILL_CANDIDATE.read_bytes()
    )
    assert _sha(H08_HUMAN_SKILL_CANDIDATE.read_bytes()) == (
        H08_HUMAN_SKILL_FILE_SHA256
    )
    assert skill.sha256 == H08_HUMAN_SKILL_SHA256
    assert iteration_protocol._load_approved_human_skill(
        H08_HUMAN_SKILL_CANDIDATE
    ) == skill
    assert "authoritative" not in skill.content
    assert skill.content["authority_policy"] == {
        "candidate_state": "content_finalized_awaiting_external_digest_approval",
        "external_registry": "APPROVED_HUMAN_REPLAY_SKILL_SHA256",
        "rule": "authoritative use is granted exclusively by presence of this exact canonical Skill digest in the external production approval registry; the Skill bytes do not change when authority is granted",
    }
    assert skill.content["conclusion"] == (
        "FINAL_CONTENT_AWAITING_EXTERNAL_DIGEST_APPROVAL"
    )
    assert skill.version == "reader-v6-final-candidate"
    assert skill.content["decision_assessment"]["human_confirmation"] == (
        "H03-B confirmed by project owner on 2026-08-01"
    )
    assert skill.content["turning_point_rubric"]["human_confirmation"] == (
        "H04-B confirmed by project owner on 2026-08-01"
    )
    assert skill.content["human_author_signoff"] == {
        "requirement": "H07",
        "status": "confirmed",
        "name": "龚恒",
        "date": "2026-08-01",
        "statement": "I reviewed the complete 24-miracle-read-replay rule set, evidence boundaries, H03-B decision criteria, and H04-B turning-point criteria; I accept the current content and sign it as its human author.",
    }
    assert [item["id"] for item in skill.content["human_required"]] == [
        "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08"
    ]
    assert skill.content["real_adapter_production_evidence"]["report_kind"] == (
        "approved_single_match_production_evidence_report"
    )
    assert skill.content["real_adapter_production_evidence"][
        "direct_state_after"
    ] == 348
    assert skill.content["real_adapter_production_evidence"][
        "intervened_observation_intervals"
    ] == 50
    assert skill.content["real_adapter_production_evidence"][
        "authoritative"
    ] is True
    assert skill.content["human_required"][-1]["status"] == "confirmed"
    baseline = json.loads(HUMAN_SKILL_V4_BASE.read_text(encoding="utf-8"))
    for key in (
        "evidence_labels",
        "workflow",
        "state_dimensions",
        "decision_assessment",
        "turning_point_rubric",
        "human_author_signoff",
        "output_sections",
    ):
        assert skill.content[key] == baseline["content"][key]
    for key, value in baseline["content"]["scope"].items():
        if key == "real_judge_adapter":
            continue
        assert skill.content["scope"][key] == value
    assert skill.content["forbidden_inferences"][:6] == baseline["content"][
        "forbidden_inferences"
    ]
    assert skill.content["human_required"][:7] == baseline["content"][
        "human_required"
    ][:7]
    assert iteration_protocol.APPROVED_HUMAN_REPLAY_SKILL_SHA256 == frozenset(
        {H08_HUMAN_SKILL_SHA256}
    )

    approval_audit = real._parse_json_object(
        H08_HUMAN_SKILL_APPROVAL.read_bytes(),
        "H08 HumanReplaySkill approval",
        canonical=True,
    )
    assert approval_audit["human_replay_skill"] == {
        "file_sha256": H08_HUMAN_SKILL_FILE_SHA256,
        "sha256": H08_HUMAN_SKILL_SHA256,
    }
    assert approval_audit["manifest_sha256"] == H08_MANIFEST_SHA256
    assert approval_audit["lifecycle"] == {
        "approved_loader": "passed",
        "real_judge_gate": "passed",
        "real_judge_open": "passed",
        "real_judge_preflight": "passed",
    }

    assert H08_FINAL_MANIFEST.read_bytes() == manifest_bytes
    assert real.current_adapter_identity() == adapter_identity
    assert packet.manifest_sha256 == H08_MANIFEST_SHA256

    assert miracle.RealJudgeReplayPacket is real.RealJudgeReplayPacket
    assert miracle.preflight_real_judge_replay is real.preflight_real_judge_replay


def test_h08_human_skill_approval_rejects_unapproved_tampered_and_revoked(
    tmp_path, monkeypatch
):
    from agentbench_frame.games.miracle import iteration_protocol

    signed_bytes = H08_HUMAN_SKILL_CANDIDATE.read_bytes()
    signed = iteration_protocol._load_approved_human_skill(
        H08_HUMAN_SKILL_CANDIDATE
    )
    assert signed.sha256 == H08_HUMAN_SKILL_SHA256

    unapproved = replace(
        signed,
        version="reader-v6-unapproved-test",
        sha256="",
    )
    unapproved_path = tmp_path / "unapproved-human-replay-skill.json"
    unapproved_path.write_bytes(unapproved.canonical_bytes())
    with pytest.raises(
        iteration_protocol.HumanAuthoredContentRequired,
        match="not approved",
    ):
        iteration_protocol._load_approved_human_skill(unapproved_path)

    tampered_path = tmp_path / "tampered-human-replay-skill.json"
    tampered_path.write_bytes(
        signed_bytes.replace(
            b'"reader-v6-final-candidate"',
            b'"reader-v6-final-candidatf"',
        )
    )
    with pytest.raises(
        iteration_protocol.IterationPreflightError,
        match="SHA mismatch",
    ):
        iteration_protocol._load_approved_human_skill(tampered_path)

    monkeypatch.setattr(
        iteration_protocol,
        "APPROVED_HUMAN_REPLAY_SKILL_SHA256",
        frozenset(),
    )
    with pytest.raises(
        iteration_protocol.HumanAuthoredContentRequired,
        match="not approved",
    ):
        iteration_protocol._load_approved_human_skill(
            H08_HUMAN_SKILL_CANDIDATE
        )


def test_code_identity_is_strict_and_lf_equivalent():
    lf = b"alpha\nbeta\n"
    assert real._canonical_source_sha256(lf) == real._canonical_source_sha256(
        lf.replace(b"\n", b"\r\n")
    )
    with pytest.raises(ValueError, match="BOM"):
        real._canonical_source_sha256(b"\xef\xbb\xbf" + lf)
    with pytest.raises(ValueError, match="carriage return"):
        real._canonical_source_sha256(b"alpha\rbeta\n")


def test_real_replay_context_is_issuer_only_and_forgery_fails_closed():
    with pytest.raises(TypeError, match="issuer-only"):
        real.RealJudgeReplayReadingContext()
    forged = object.__new__(real.RealJudgeReplayReadingContext)
    with pytest.raises(ValueError, match="trusted issued"):
        real.open_real_judge_replay(forged)


def test_synthetic_reader_contract_and_other_production_approvals_are_unchanged():
    from agentbench_frame.games.miracle import iteration_protocol

    assert replay.SYNTHETIC_REPLAY_SCHEMA_VERSION == "24-miracle-synthetic-replay-v1"
    assert replay.REPLAY_READING_MANIFEST_SCHEMA_VERSION == "24-miracle-replay-reading-manifest-v1"
    assert replay.APPROVED_TRAINING_REPLAY_MANIFESTS == frozenset()
    assert approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS == frozenset(
        {H08_MANIFEST_SHA256}
    )
    assert iteration_protocol.APPROVED_HUMAN_REPLAY_SKILL_SHA256 == frozenset(
        {H08_HUMAN_SKILL_SHA256}
    )
    assert iteration_protocol.APPROVED_MATCH_PLAN_MANIFEST_SHA256 == frozenset()
    assert iteration_protocol.APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256 == frozenset()
    assert (
        iteration_protocol.APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256
        == frozenset()
    )
    assert (
        iteration_protocol.APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256
        == frozenset()
    )
    assert (
        iteration_protocol.APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256
        == frozenset()
    )

"""Fail-closed adapter for approved real 24_miracle Judge replay triads.

The production entry point requires an independently approved canonical
manifest before parsing it.  Candidate audit is deliberately separate and
returns ``authoritative=False``; it exists so a digest can be reviewed before
being added to the empty production approval boundary.

The adapter never starts a Judge, AI, runner, Provider, session, or workspace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import ntpath
import os
import struct
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentbench_frame.eval.measurement import canonical_state_id
from agentbench_frame.games.miracle import real_replay_approvals_v1 as approvals
from agentbench_frame.games.miracle import research_protocol
from agentbench_frame.games.miracle import replay_reading_v1 as synthetic


REAL_JUDGE_MANIFEST_SCHEMA_VERSION = "24-miracle-real-judge-replay-manifest-v1"
REAL_JUDGE_PACKET_SCHEMA_VERSION = "24-miracle-real-judge-replay-packet-v1"
REAL_JUDGE_DECISION_FRAME_SCHEMA_VERSION = (
    "24-miracle-real-judge-decision-frame-v1"
)
REAL_JUDGE_RUN_INTENT_SCHEMA_VERSION = "24-miracle-h08-run-intent-v1"
RATIONALE_STATUS = "not_recorded"

MAX_REAL_JUDGE_MANIFEST_BYTES = 256 * 1024
MAX_REAL_JUDGE_RUN_INTENT_BYTES = 256 * 1024
MAX_REAL_JUDGE_REPLAY_BYTES = 32 * 1024 * 1024
MAX_REAL_JUDGE_TRACE_BYTES = 64 * 1024 * 1024
MAX_REAL_JUDGE_RESULT_BYTES = 1024 * 1024
MAX_REAL_JUDGE_TRACE_LINE_BYTES = 4 * 1024 * 1024

_EVENT_NAMES = (
    "",
    "TurnStart",
    "TurnEnd",
    "Spawn",
    "Move",
    "Attack",
    "Damage",
    "Death",
    "Heal",
    "ActivateArtifact",
    "GameEnd",
    "GameStart",
    "BuffAdd",
    "BuffRemove",
    "Attacking",
    "Attacked",
    "Leave",
    "Arrive",
    "Summon",
)
_PUBLIC_ACTIONS = frozenset(
    {"init", "move", "attack", "summon", "use", "endround", "surrender"}
)
_TRACE_KINDS = frozenset(
    {
        "match_start",
        "judge_frame",
        "ai_operation",
        "match_end",
        "ai_timeout",
        "ai_error",
        "stderr",
    }
)


def canonical_real_replay_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize an approval manifest as strict canonical UTF-8 JSON."""

    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_source_sha256(payload: bytes) -> str:
    if type(payload) is not bytes or payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("bound source must be strict UTF-8 without BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("bound source must be strict UTF-8") from exc
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise ValueError("bound source contains a bare carriage return")
    return _sha256(normalized.encode("utf-8"))


def _source_sha256(module: Any) -> str:
    return _canonical_source_sha256(Path(module.__file__).read_bytes())


def current_adapter_identity() -> dict[str, str]:
    """Return exact source identities required by a real-replay manifest."""

    return {
        "source_digest_mode": "strict_utf8_canonical_lf_v1",
        "adapter_schema_version": REAL_JUDGE_PACKET_SCHEMA_VERSION,
        "adapter_source_sha256": _source_sha256(
            __import__(__name__, fromlist=["unused"])
        ),
        "replay_reader_source_sha256": _source_sha256(synthetic),
        "research_protocol_source_sha256": _source_sha256(research_protocol),
    }


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(
    payload: bytes, label: str, *, canonical: bool = False
) -> dict[str, Any]:
    if type(payload) is not bytes or payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must be strict UTF-8 without BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token}")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    if canonical and payload != canonical_real_replay_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _strict_text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty exact string")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a strict boolean")
    return value


def _strict_int(value: Any, label: str, *, minimum=None, maximum=None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be a strict integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its allowed range")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its allowed range")
    return value


def _strict_number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a strict finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_git_sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-hex Git SHA")
    return value


def _captured_path_name(value: Any, label: str) -> str:
    path = _strict_text(value, label)
    if "\x00" in path:
        raise ValueError(f"{label} contains a NUL")
    name = ntpath.basename(path)
    if not name or name in {".", ".."}:
        raise ValueError(f"{label} has no exact final component")
    return name


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are incomplete or extra")
    return value


def _artifact_descriptor(
    value: Any, label: str, *, limit: int
) -> dict[str, Any]:
    value = _exact_fields(value, {"path", "bytes", "sha256"}, label)
    path = "/".join(synthetic._relative_parts(value["path"], f"{label} path"))
    size = _strict_int(value["bytes"], f"{label} bytes", minimum=1, maximum=limit)
    return {
        "path": path,
        "bytes": size,
        "sha256": _strict_sha256(value["sha256"], f"{label} SHA"),
    }


def _validate_players(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        raise ValueError("real replay players must contain camps 0 and 1 exactly")
    parsed = []
    for item in value:
        item = _exact_fields(
            item, {"camp", "name", "entry", "tree_sha256"}, "player identity"
        )
        parsed.append(
            {
                "camp": _strict_int(item["camp"], "player camp", minimum=0, maximum=1),
                "name": _strict_text(item["name"], "player name"),
                "entry": _strict_text(item["entry"], "player entry"),
                "tree_sha256": _strict_sha256(
                    item["tree_sha256"], "player tree identity"
                ),
            }
        )
    if [item["camp"] for item in parsed] != [0, 1]:
        raise ValueError("real replay player identities must be ordered camps 0,1")
    return parsed[0], parsed[1]


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "role",
        "case_id",
        "decision_scope",
        "evaluated_camp",
        "players",
        "capture_identity",
        "adapter_identity",
        "run_intent",
        "artifacts",
        "environment",
        "expected_result",
        "expected_adapter_output",
    }
    _exact_fields(value, fields, "real replay manifest")
    if value["schema_version"] != REAL_JUDGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("real replay manifest schema mismatch")
    if value["role"] != "train":
        raise ValueError("real Judge Replay Reading accepts only role=train")
    case_id = _strict_text(value["case_id"], "real replay case ID")
    if value["decision_scope"] not in {"evaluated_camp_only", "all_camps"}:
        raise ValueError("real replay decision scope is invalid")
    evaluated_camp = _strict_int(
        value["evaluated_camp"], "evaluated camp", minimum=0, maximum=1
    )
    players = _validate_players(value["players"])

    capture = _exact_fields(
        value["capture_identity"],
        {
            "framework_git_head",
            "framework_branch",
            "runner_sha256",
            "judge_repository_git_head",
            "judge_tree_sha256",
        },
        "capture identity",
    )
    capture = {
        "framework_git_head": _strict_git_sha(
            capture["framework_git_head"], "capture Framework Git identity"
        ),
        "framework_branch": _strict_text(
            capture["framework_branch"], "capture Framework branch"
        ),
        "runner_sha256": _strict_sha256(capture["runner_sha256"], "runner"),
        "judge_repository_git_head": _strict_git_sha(
            capture["judge_repository_git_head"], "Judge Git identity"
        ),
        "judge_tree_sha256": _strict_sha256(
            capture["judge_tree_sha256"], "Judge tree identity"
        ),
    }
    if value["adapter_identity"] != current_adapter_identity():
        raise ValueError("real replay adapter or enumerator code identity mismatch")

    run_intent = _artifact_descriptor(
        value["run_intent"],
        "run intent",
        limit=MAX_REAL_JUDGE_RUN_INTENT_BYTES,
    )
    artifacts = _exact_fields(
        value["artifacts"], {"replay", "trace", "result"}, "real replay artifacts"
    )
    artifacts = {
        "replay": _artifact_descriptor(
            artifacts["replay"], "replay", limit=MAX_REAL_JUDGE_REPLAY_BYTES
        ),
        "trace": _artifact_descriptor(
            artifacts["trace"], "trace", limit=MAX_REAL_JUDGE_TRACE_BYTES
        ),
        "result": _artifact_descriptor(
            artifacts["result"], "result", limit=MAX_REAL_JUDGE_RESULT_BYTES
        ),
    }
    environment = _exact_fields(
        value["environment"],
        {
            "external_deterministic_seed_supported",
            "seed_status",
            "realized_map_type",
            "realized_day_time",
        },
        "real replay environment",
    )
    if environment["external_deterministic_seed_supported"] is not False:
        raise ValueError("real Judge evidence must not claim external seed support")
    if environment["seed_status"] != "unsupported_not_recorded":
        raise ValueError("real Judge seed must remain unsupported_not_recorded")
    environment = {
        "external_deterministic_seed_supported": False,
        "seed_status": "unsupported_not_recorded",
        "realized_map_type": _strict_int(
            environment["realized_map_type"], "realized map type", minimum=0, maximum=1
        ),
        "realized_day_time": _strict_int(
            environment["realized_day_time"], "realized day time", minimum=0, maximum=1
        ),
    }

    expected_result = _exact_fields(
        value["expected_result"],
        {
            "status",
            "scores",
            "winner",
            "timeout",
            "ai_error",
            "runner_exception",
            "cleanup_succeeded",
        },
        "expected result",
    )
    if expected_result["status"] != "complete":
        raise ValueError("real replay expected result must be complete")
    scores = _exact_fields(expected_result["scores"], {"0", "1"}, "expected scores")
    expected_result = {
        "status": "complete",
        "scores": {
            "0": _strict_int(scores["0"], "camp 0 score"),
            "1": _strict_int(scores["1"], "camp 1 score"),
        },
        "winner": _strict_int(expected_result["winner"], "expected winner", minimum=0, maximum=1),
        "timeout": _strict_bool(expected_result["timeout"], "expected timeout"),
        "ai_error": _strict_bool(expected_result["ai_error"], "expected AI error"),
        "runner_exception": _strict_bool(
            expected_result["runner_exception"], "expected runner exception"
        ),
        "cleanup_succeeded": _strict_bool(
            expected_result["cleanup_succeeded"], "expected cleanup"
        ),
    }
    if any(
        (
            expected_result["timeout"],
            expected_result["ai_error"],
            expected_result["runner_exception"],
            not expected_result["cleanup_succeeded"],
        )
    ):
        raise ValueError("real replay complete result cannot contain execution failures")

    expected_output = _exact_fields(
        value["expected_adapter_output"],
        {
            "decision_frames",
            "support_complete",
            "chosen_in_support",
            "opponent_protocol_compatible",
        },
        "expected adapter output",
    )
    expected_output = {
        "decision_frames": _strict_int(
            expected_output["decision_frames"], "expected decision frames", minimum=1
        ),
        "support_complete": _strict_int(
            expected_output["support_complete"], "expected complete supports", minimum=1
        ),
        "chosen_in_support": _strict_int(
            expected_output["chosen_in_support"], "expected chosen-in-support", minimum=1
        ),
        "opponent_protocol_compatible": _strict_bool(
            expected_output["opponent_protocol_compatible"],
            "expected opponent compatibility",
        ),
    }
    return {
        "schema_version": REAL_JUDGE_MANIFEST_SCHEMA_VERSION,
        "role": "train",
        "case_id": case_id,
        "decision_scope": value["decision_scope"],
        "evaluated_camp": evaluated_camp,
        "players": players,
        "capture_identity": capture,
        "adapter_identity": dict(value["adapter_identity"]),
        "run_intent": run_intent,
        "artifacts": artifacts,
        "environment": environment,
        "expected_result": expected_result,
        "expected_adapter_output": expected_output,
    }


def _validate_run_intent(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "created_at",
        "purpose",
        "evidence_role",
        "case",
        "evaluated_camp",
        "authority",
        "seed_policy",
        "runtime",
        "judge",
        "players",
        "tree_digest_algorithm",
        "production_approval_claimed",
    }
    _exact_fields(value, fields, "run intent")
    if value["schema_version"] != REAL_JUDGE_RUN_INTENT_SCHEMA_VERSION:
        raise ValueError("run intent schema mismatch")
    if value["evidence_role"] != "train" or manifest["role"] != "train":
        raise ValueError("run intent and manifest must both be role=train")
    if value["case"] != manifest["case_id"]:
        raise ValueError("run intent case identity mismatch")
    if value["evaluated_camp"] != manifest["evaluated_camp"]:
        raise ValueError("run intent evaluated camp identity mismatch")
    _strict_text(value["created_at"], "run intent creation time")
    _strict_text(value["purpose"], "run intent purpose")
    _exact_fields(value["authority"], {"basis", "human_author", "date"}, "run authority")
    for key in ("basis", "human_author", "date"):
        _strict_text(value["authority"][key], f"run authority {key}")

    seed_policy = _exact_fields(
        value["seed_policy"],
        {"external_deterministic_seed_supported", "requirement"},
        "run seed policy",
    )
    if seed_policy["external_deterministic_seed_supported"] is not False:
        raise ValueError("run intent must declare unsupported external Judge seed")
    if "do not invent a seed" not in _strict_text(
        seed_policy["requirement"], "run seed requirement"
    ):
        raise ValueError("run intent must explicitly forbid invented seeds")

    runtime = _exact_fields(
        value["runtime"],
        {
            "python",
            "framework_git_head",
            "framework_branch",
            "framework_worktree_clean",
            "runner_sha256",
        },
        "run runtime identity",
    )
    capture = manifest["capture_identity"]
    if (
        runtime["framework_git_head"] != capture["framework_git_head"]
        or runtime["framework_branch"] != capture["framework_branch"]
        or runtime["runner_sha256"] != capture["runner_sha256"]
        or runtime["framework_worktree_clean"] is not True
    ):
        raise ValueError("run intent Framework or runner identity mismatch")
    _strict_text(runtime["python"], "run Python identity")

    judge = _exact_fields(
        value["judge"],
        {
            "path",
            "repository_git_head",
            "subtree_tracked_clean",
            "file_count",
            "tree_sha256",
        },
        "run Judge identity",
    )
    if (
        judge["repository_git_head"] != capture["judge_repository_git_head"]
        or judge["tree_sha256"] != capture["judge_tree_sha256"]
        or judge["subtree_tracked_clean"] is not True
    ):
        raise ValueError("run intent Judge identity mismatch")
    _strict_text(judge["path"], "run Judge path")
    _strict_int(judge["file_count"], "Judge file count", minimum=1)

    players = value["players"]
    if type(players) is not list or len(players) != 2:
        raise ValueError("run intent must bind both player identities")
    normalized = []
    for player in players:
        _exact_fields(
            player,
            {"camp", "name", "entry", "path", "file_count", "tree_sha256"},
            "run player identity",
        )
        normalized.append(
            {
                "camp": _strict_int(player["camp"], "run player camp", minimum=0, maximum=1),
                "name": _strict_text(player["name"], "run player name"),
                "entry": _strict_text(player["entry"], "run player entry"),
                "path": _strict_text(player["path"], "run player path"),
                "file_count": _strict_int(player["file_count"], "run player file count", minimum=1),
                "tree_sha256": _strict_sha256(
                    player["tree_sha256"], "run player tree identity"
                ),
            }
        )
    for actual, expected in zip(normalized, manifest["players"]):
        if {key: actual[key] for key in ("camp", "name", "entry", "tree_sha256")} != expected:
            raise ValueError("run intent player identity mismatch")
    if value["production_approval_claimed"] is not False:
        raise ValueError("run intent cannot self-claim production approval")
    _strict_text(value["tree_digest_algorithm"], "tree digest algorithm")
    return {
        "judge_path": judge["path"],
        "player_paths": tuple(player["path"] for player in normalized),
    }


def _validate_process(value: Any, role: str) -> None:
    fields = {
        "pid",
        "role",
        "started_at",
        "natural_exit",
        "natural_returncode",
        "termination_requested",
        "termination_reason",
        "final_returncode",
        "forced_kill",
        "cleanup_succeeded",
        "identity_confirmed",
    }
    value = _exact_fields(value, fields, f"{role} process result")
    _strict_int(value["pid"], f"{role} PID", minimum=1)
    _strict_number(value["started_at"], f"{role} started time")
    if value["role"] != role:
        raise ValueError(f"{role} process identity mismatch")
    for key in (
        "natural_exit",
        "termination_requested",
        "forced_kill",
        "cleanup_succeeded",
        "identity_confirmed",
    ):
        _strict_bool(value[key], f"{role} {key}")
    if (
        value["natural_exit"] is not True
        or value["termination_requested"] is not True
        or value["termination_reason"] != "match-end"
        or value["forced_kill"] is not False
        or value["cleanup_succeeded"] is not True
        or value["identity_confirmed"] is not True
    ):
        raise ValueError(f"{role} process did not close cleanly with confirmed identity")
    _strict_int(value["natural_returncode"], f"{role} natural return code")
    _strict_int(value["final_returncode"], f"{role} final return code")
    if role == "judge" and (
        value["natural_returncode"] != 0 or value["final_returncode"] != 0
    ):
        raise ValueError("Judge did not exit successfully")


def _validate_result(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent_identity: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "tag",
        "started_at",
        "finished_at",
        "duration_s",
        "judge_dir_resolved",
        "p0",
        "p1",
        "judge",
        "ai0",
        "ai1",
        "end_info_received",
        "end_info",
        "scores",
        "raw_winner",
        "score_tie",
        "judge_tiebreak_applied",
        "timeout",
        "ai_error",
        "trace_path",
        "replay_path",
        "cleanup_all_succeeded",
        "exception",
        "run_match_returncode",
    }
    _exact_fields(value, fields, "real Judge result")
    if value["schema_version"] != 1:
        raise ValueError("real Judge result schema mismatch")
    started = _strict_number(value["started_at"], "result started time")
    finished = _strict_number(value["finished_at"], "result finished time")
    duration = _strict_number(value["duration_s"], "result duration")
    if finished < started or duration < 0:
        raise ValueError("real Judge result time order is invalid")
    _strict_text(value["tag"], "result tag")
    if value["judge_dir_resolved"] != intent_identity["judge_path"]:
        raise ValueError("result Judge path identity mismatch")
    for camp, key in enumerate(("p0", "p1")):
        player = _exact_fields(value[key], {"name", "dir"}, f"result {key}")
        if (
            player["name"] != manifest["players"][camp]["name"]
            or player["dir"] != intent_identity["player_paths"][camp]
        ):
            raise ValueError("result player identity mismatch")
    _validate_process(value["judge"], "judge")
    _validate_process(value["ai0"], "ai0")
    _validate_process(value["ai1"], "ai1")
    if value["end_info_received"] is not True:
        raise ValueError("real Judge result is missing end_info")
    end_info = _exact_fields(value["end_info"], {"0", "1"}, "result end_info")
    scores = _exact_fields(value["scores"], {"0", "1"}, "result scores")
    normalized_scores = {
        "0": _strict_int(scores["0"], "result camp 0 score"),
        "1": _strict_int(scores["1"], "result camp 1 score"),
    }
    if dict(end_info) != normalized_scores:
        raise ValueError("result end_info and scores disagree")
    expected = manifest["expected_result"]
    if normalized_scores != expected["scores"]:
        raise ValueError("result scores disagree with approved manifest")
    winner = 0 if normalized_scores["0"] > normalized_scores["1"] else 1
    tie = normalized_scores["0"] == normalized_scores["1"]
    if (
        value["raw_winner"] != winner
        or winner != expected["winner"]
        or value["score_tie"] is not tie
        or value["judge_tiebreak_applied"] is not tie
    ):
        raise ValueError("result winner or tie identity mismatch")
    for label in ("timeout", "ai_error"):
        flags = _exact_fields(value[label], {"ai0", "ai1"}, f"result {label}")
        if flags != {"ai0": False, "ai1": False}:
            raise ValueError(f"real Judge result contains {label}")
    if (
        value["exception"] is not None
        or value["cleanup_all_succeeded"] is not True
        or value["run_match_returncode"] != 0
    ):
        raise ValueError("real Judge runner exception or cleanup failure")
    if _captured_path_name(value["trace_path"], "result trace path") != (
        _captured_path_name(
            manifest["artifacts"]["trace"]["path"], "manifest trace path"
        )
    ):
        raise ValueError("result trace path identity mismatch")
    if _captured_path_name(value["replay_path"], "result replay path") != (
        _captured_path_name(
            manifest["artifacts"]["replay"]["path"], "manifest replay path"
        )
    ):
        raise ValueError("result replay path identity mismatch")
    evaluated = manifest["evaluated_camp"]
    return {
        "scores": normalized_scores,
        "winner": winner,
        "outcome": "win" if winner == evaluated else "loss",
        "duration_s": duration,
    }


def _zeros(values: tuple[int, ...], label: str) -> None:
    if any(value != 0 for value in values):
        raise ValueError(f"replay {label} has invalid non-zero padding")


def _encoded_camp_item(value: int, maximum: int, label: str) -> int:
    if value in range(1, maximum + 1):
        return 0
    if value in range(11, 11 + maximum):
        return 1
    raise ValueError(f"replay {label} encoding is outside its source-defined range")


def _map_position(x: int, y: int, label: str) -> None:
    if not (-8 <= x <= 8 and -8 <= y <= 8 and -14 <= -(x + y) <= 14):
        raise ValueError(f"replay {label} position is outside the Miracle map")


def _validate_event_record(record: tuple[int, ...]) -> None:
    round_number, code, a, b, c, d, e = record
    if not 0 <= round_number <= 1_000_000:
        raise ValueError("replay event round is outside its allowed range")
    if not 1 <= code <= 18:
        raise ValueError("replay event code is outside 1..18")
    if code in (1, 2):
        if a not in (0, 1):
            raise ValueError("replay turn event camp is invalid")
        _zeros((b, c, d, e), "turn event")
    elif code == 3:
        _encoded_camp_item(a, 7, "spawn creature")
        if b not in (1, 2, 3):
            raise ValueError("replay spawn level is invalid")
        _map_position(c, d, "spawn")
        if e < 0:
            raise ValueError("replay spawn ID is invalid")
    elif code in (4, 16, 17):
        if a < 0:
            raise ValueError("replay movement event ID is invalid")
        _map_position(b, c, "movement")
        _zeros((d, e), "movement event")
    elif code in (5, 14, 15):
        if a < 0 or b < 0:
            raise ValueError("replay attack event identity is invalid")
        _zeros((c, d, e), "attack event")
    elif code == 6:
        if a < 0 or b < 0 or c < 0 or d not in (1, 2, 3, 4):
            raise ValueError("replay damage event fields are invalid")
        _zeros((e,), "damage event")
    elif code == 7:
        if a < 0:
            raise ValueError("replay death ID is invalid")
        _zeros((b, c, d, e), "death event")
    elif code == 8:
        if a < 0 or b < 0 or c < 0:
            raise ValueError("replay heal event fields are invalid")
        _zeros((d, e), "heal event")
    elif code == 9:
        if a not in (0, 1) or _encoded_camp_item(b, 4, "artifact") != a:
            raise ValueError("replay artifact camp identity is invalid")
        if not (-8 <= c <= 1_000_000 and -8 <= d <= 8 and e >= 0):
            raise ValueError("replay artifact target fields are invalid")
    elif code == 10:
        if a not in (0, 1):
            raise ValueError("replay GameEnd winner is invalid")
        _zeros((b, c, d, e), "GameEnd")
    elif code == 11:
        if a not in (0, 1) or _encoded_camp_item(b, 4, "GameStart artifact") != a:
            raise ValueError("replay GameStart identity is invalid")
        for creature in (c, d, e):
            if _encoded_camp_item(creature, 7, "GameStart creature") != a:
                raise ValueError("replay GameStart creature camp is invalid")
    elif code in (12, 13):
        if a < 0 or b not in (0, 1, 2, 3, 4):
            raise ValueError("replay buff event fields are invalid")
        _zeros((c, d, e), "buff event")
    elif code == 18:
        _encoded_camp_item(a, 7, "summon creature")
        if b not in (1, 2, 3):
            raise ValueError("replay summon level is invalid")
        _map_position(c, d, "summon")
        _zeros((e,), "summon event")


def _parse_binary_replay(
    payload: bytes, environment: Mapping[str, Any]
) -> dict[str, Any]:
    if len(payload) < 32 or len(payload) % 4:
        raise ValueError("binary replay is shorter than header/trailer or misaligned")
    integers = struct.unpack(f">{len(payload) // 4}i", payload)
    header = integers[:7]
    if header[:3] != (0, 0, 0) or header[5:] != (0, 0):
        raise ValueError("binary replay header fixed fields are invalid")
    if header[3] not in (0, 1) or header[4] not in (0, 1):
        raise ValueError("binary replay header map/day fields are invalid")
    if (
        header[3] != environment["realized_map_type"]
        or header[4] != environment["realized_day_time"]
    ):
        raise ValueError("binary replay header disagrees with approved environment")
    if integers[-1] != -1:
        raise ValueError("binary replay is missing its -1 trailer")
    body = integers[7:-1]
    if not body or len(body) % 7:
        raise ValueError("binary replay event body is not aligned to seven integers")
    records = tuple(
        tuple(body[index : index + 7]) for index in range(0, len(body), 7)
    )
    for record in records:
        _validate_event_record(record)
    if any(left[0] > right[0] for left, right in zip(records, records[1:])):
        raise ValueError("binary replay event rounds are out of order")
    game_end = tuple(record for record in records if record[1] == 10)
    if len(game_end) != 1 or records[-1] != game_end[0]:
        raise ValueError("binary replay must end with exactly one GameEnd event")
    if sum(record[1] == 11 for record in records) != 2:
        raise ValueError("binary replay must contain exactly two GameStart events")
    return {
        "header": {
            "map_type": header[3],
            "day_time": header[4],
            "raw": tuple(header),
        },
        "event_count": len(records),
        "max_round": max(record[0] for record in records),
        "game_end_round": game_end[0][0],
        "game_end_winner": game_end[0][2],
    }


def _decode_observation_frame(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not str or len(value) < 6 or not value[:6].isdigit():
        raise ValueError(f"{label} is not a six-digit-length Judge observation")
    declared = int(value[:6])
    body = value[6:]
    if declared != len(body):
        raise ValueError(f"{label} Judge observation length mismatch")
    return _parse_json_object(body.encode("utf-8"), label)


def _state(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(observation))
    return {
        "canonical_state_id": canonical_state_id(copied),
        "observation": copied,
    }


def _support(observation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        support = research_protocol.build_action_support(
            research_protocol.enumerate_legal_commands(observation)
        )
    except research_protocol.IncompleteActionSupportError as exc:
        raise ValueError(f"complete ActionSupport is unavailable: {exc}") from exc
    return {
        "schema_version": support.schema_version,
        "support_id": support.support_id,
        "actions": [
            {"action_id": item.action_id, "command": item.action}
            for item in support.actions
        ],
    }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class RealJudgeDecisionFrame:
    schema_version: str
    decision_step: int
    trace_line: int
    state_before: Mapping[str, Any]
    action_support: Mapping[str, Any]
    chosen_action: Mapping[str, Any]
    acting_identity_refs: Mapping[str, Any]
    state_after: Mapping[str, Any] | None
    next_observed_state: Mapping[str, Any] | None
    next_observation_trace_line: int | None
    transition_scope: str
    intervening_action_count: int
    intervening_trace_lines: tuple[int, ...]
    terminal_event: Mapping[str, Any] | None
    rationale_status: str = RATIONALE_STATUS


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class RealJudgeReplayPacket:
    schema_version: str
    authoritative: bool
    evidence_scope: str
    manifest_sha256: str
    role: str
    case_id: str
    evaluated_camp: int
    players: tuple[Mapping[str, Any], Mapping[str, Any]]
    capture_identity: Mapping[str, Any]
    adapter_identity: Mapping[str, Any]
    environment: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]
    decision_frames: tuple[RealJudgeDecisionFrame, ...]
    terminal: Mapping[str, Any]
    opponent_compatibility: Mapping[str, Any]
    support_complete_count: int
    chosen_in_support_count: int
    direct_state_after_count: int
    intervened_transition_count: int
    terminal_without_state_after_count: int
    rationale_status: str = RATIONALE_STATUS


@dataclass(frozen=True)
class RealJudgeReplayCandidate:
    authoritative: bool
    manifest_sha256: str
    packet: RealJudgeReplayPacket


_AUTHORIZED_REAL_REPLAY_PACKETS: weakref.WeakSet[RealJudgeReplayPacket] = (
    weakref.WeakSet()
)


def _parse_trace_events(payload: bytes) -> list[dict[str, Any]]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("trace must be strict UTF-8 without BOM")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("trace must be strict UTF-8 JSONL") from exc
    lines = payload.splitlines()
    if not lines:
        raise ValueError("trace JSONL is empty")
    events = []
    for line_number, line in enumerate(lines, 1):
        if not line or len(line) > MAX_REAL_JUDGE_TRACE_LINE_BYTES:
            raise ValueError("trace JSONL contains an empty or oversized line")
        event = _parse_json_object(line, f"trace line {line_number}")
        event["_line_number"] = line_number
        events.append(event)
    return events


def _validate_trace_event_shape(event: Mapping[str, Any]) -> str:
    kind = event.get("kind")
    if kind not in _TRACE_KINDS:
        raise ValueError("trace contains an unsupported event kind")
    expected = {
        "match_start": {"kind", "players", "replay", "judge_dir", "_line_number"},
        "judge_frame": {"kind", "goal", "frame", "_line_number"},
        "ai_operation": {"kind", "player", "name", "operation", "_line_number"},
        "match_end": {"kind", "end_info", "_line_number"},
        "ai_timeout": {"kind", "player", "state", "_line_number"},
        "ai_error": {"kind", "player", "state", "error", "_line_number"},
        "stderr": {"kind", "who", "stderr", "_line_number"},
    }[kind]
    if set(event) != expected:
        raise ValueError(f"trace {kind} fields are incomplete or extra")
    return kind


def _terminal_scores(value: Any, label: str) -> dict[str, int]:
    if type(value) is not str:
        raise ValueError(f"{label} must be encoded Judge end_info text")
    parsed = _parse_json_object(value.encode("utf-8"), label)
    _exact_fields(parsed, {"0", "1"}, label)
    return {
        "0": _strict_int(parsed["0"], f"{label} camp 0 score"),
        "1": _strict_int(parsed["1"], f"{label} camp 1 score"),
    }


def _build_frame(
    *,
    step: int,
    line: int,
    observation: Mapping[str, Any],
    operation: Mapping[str, Any],
    player_identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        canonical = research_protocol.canonical_command(operation)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evaluated command is not canonical: {exc}") from exc
    if canonical["operation_type"] not in _PUBLIC_ACTIONS:
        raise ValueError("evaluated trace contains a non-agent internal operation")
    supplied = _support(observation)
    action_id = research_protocol.command_action_id(canonical)
    by_id = {item["action_id"]: item["command"] for item in supplied["actions"]}
    if by_id.get(action_id) != canonical:
        raise ValueError("evaluated chosen action is outside trusted ActionSupport")
    return {
        "schema_version": REAL_JUDGE_DECISION_FRAME_SCHEMA_VERSION,
        "decision_step": step,
        "trace_line": line,
        "state_before": _freeze(_state(observation)),
        "action_support": _freeze(supplied),
        "chosen_action": _freeze(
            {"action_id": action_id, "command": canonical}
        ),
        "acting_identity_refs": _freeze(
            {
                "camp": player_identity["camp"],
                "name": player_identity["name"],
                "entry": player_identity["entry"],
                "tree_sha256": player_identity["tree_sha256"],
            }
        ),
    }


def _audit_opponent_operation(
    stats: dict[str, int], observation: Mapping[str, Any], operation: Any
) -> None:
    stats["operations"] += 1
    try:
        canonical = research_protocol.canonical_command(operation)
    except (TypeError, ValueError):
        stats["command_rejected"] += 1
        return
    stats["command_canonical"] += 1
    try:
        support = research_protocol.build_action_support(
            research_protocol.enumerate_legal_commands(observation)
        )
    except research_protocol.IncompleteActionSupportError:
        return
    stats["support_complete"] += 1
    if research_protocol.command_action_id(canonical) in support.action_ids:
        stats["chosen_in_support"] += 1
    else:
        stats["chosen_outside_support"] += 1


def _build_packet_from_trace(
    payload: bytes,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    replay_info: Mapping[str, Any],
    result_info: Mapping[str, Any],
    intent_identity: Mapping[str, Any],
    *,
    authoritative: bool,
) -> RealJudgeReplayPacket:
    events = _parse_trace_events(payload)
    if events[0].get("kind") != "match_start":
        raise ValueError("trace order must begin with match_start")
    if sum(event.get("kind") == "match_start" for event in events) != 1:
        raise ValueError("trace order requires exactly one match_start")
    if sum(event.get("kind") == "match_end" for event in events) != 1:
        raise ValueError("trace order requires exactly one match_end")

    evaluated = manifest["evaluated_camp"]
    opponent = evaluated ^ 1
    players = manifest["players"]
    observations: dict[int, Mapping[str, Any]] = {}
    observation_ready = {0: False, 1: False}
    pending: dict[str, Any] | None = None
    intervening_trace_lines: list[int] = []
    frames: list[RealJudgeDecisionFrame] = []
    match_end_scores: dict[str, int] | None = None
    terminal_frame_scores: dict[str, int] | None = None
    terminal_frame_seen = False
    match_end_seen = False
    opponent_stats = {
        "operations": 0,
        "command_canonical": 0,
        "command_rejected": 0,
        "support_complete": 0,
        "chosen_in_support": 0,
        "chosen_outside_support": 0,
    }

    for index, event in enumerate(events):
        kind = _validate_trace_event_shape(event)
        if match_end_seen and kind != "stderr":
            raise ValueError("trace order contains events after match_end")
        if terminal_frame_seen and kind not in {"stderr", "match_end"}:
            raise ValueError("trace order contains events after terminal Judge frame")
        if kind == "match_start":
            if index != 0 or event["players"] != [item["name"] for item in players]:
                raise ValueError("trace match_start player identity or order mismatch")
            if _captured_path_name(event["replay"], "trace replay path") != (
                _captured_path_name(
                    manifest["artifacts"]["replay"]["path"],
                    "manifest replay path",
                )
            ):
                raise ValueError("trace match_start replay identity mismatch")
            if event["judge_dir"] != intent_identity["judge_path"]:
                raise ValueError("trace match_start Judge identity mismatch")
            continue
        if kind in {"ai_timeout", "ai_error"}:
            raise ValueError("trace contains an AI timeout or error")
        if kind == "stderr":
            _strict_text(event["who"], "trace stderr source")
            if type(event["stderr"]) is not str:
                raise ValueError("trace stderr payload must be text")
            continue
        if kind == "match_end":
            match_end_scores = _terminal_scores(event["end_info"], "trace match_end")
            match_end_seen = True
            continue
        if kind == "judge_frame":
            if event["goal"] != -1 or type(event["frame"]) is not dict:
                raise ValueError("trace Judge frame identity is invalid")
            frame = event["frame"]
            state_number = frame.get("state")
            if type(state_number) is not int:
                raise ValueError("trace Judge frame state must be a strict integer")
            if state_number == -1:
                if set(frame) != {"state", "end_info"}:
                    raise ValueError("terminal Judge frame fields are invalid")
                if terminal_frame_seen:
                    raise ValueError("trace contains duplicate terminal Judge frames")
                terminal_frame_scores = _terminal_scores(
                    frame["end_info"], "terminal Judge frame"
                )
                terminal_frame_seen = True
                continue
            if "content" not in frame:
                if set(frame) not in ({"state", "time", "length"}, {"state"}):
                    raise ValueError("trace non-observation Judge frame fields are invalid")
                continue
            if set(frame) != {"state", "listen", "player", "content"}:
                raise ValueError("trace observation Judge frame fields are invalid")
            raw_players = frame["player"]
            contents = frame["content"]
            if (
                type(raw_players) is not list
                or type(contents) is not list
                or not raw_players
                or len(raw_players) != len(contents)
            ):
                raise ValueError("trace observation player/content alignment is invalid")
            seen_players: set[int] = set()
            for raw_player, raw_content in zip(raw_players, contents):
                camp = _strict_int(raw_player, "trace observation player", minimum=0, maximum=1)
                if camp in seen_players or observation_ready[camp]:
                    raise ValueError("trace contains a duplicate unconsumed observation")
                seen_players.add(camp)
                observation = _decode_observation_frame(
                    raw_content, f"trace line {event['_line_number']} observation"
                )
                if observation.get("camp") != camp:
                    raise ValueError("trace observation camp does not match Judge player")
                if camp == evaluated and pending is not None:
                    after = _state(observation)
                    if intervening_trace_lines:
                        state_after = None
                        next_observed_state = _freeze(after)
                        transition_scope = (
                            "next_observation_after_intervening_agent_actions"
                        )
                    else:
                        state_after = _freeze(after)
                        next_observed_state = None
                        transition_scope = "direct_until_next_evaluated_observation"
                    frames.append(
                        RealJudgeDecisionFrame(
                            **pending,
                            state_after=state_after,
                            next_observed_state=next_observed_state,
                            next_observation_trace_line=event["_line_number"],
                            transition_scope=transition_scope,
                            intervening_action_count=len(intervening_trace_lines),
                            intervening_trace_lines=tuple(intervening_trace_lines),
                            terminal_event=None,
                        )
                    )
                    pending = None
                    intervening_trace_lines = []
                observations[camp] = copy.deepcopy(observation)
                observation_ready[camp] = True
            continue

        # ai_operation
        camp = _strict_int(event["player"], "trace operation player", minimum=0, maximum=1)
        if event["name"] != players[camp]["name"]:
            raise ValueError("trace operation player name identity mismatch")
        if camp not in observations or not observation_ready[camp]:
            raise ValueError("trace operation is missing its preceding observation")
        operation = event["operation"]
        if camp == evaluated:
            if pending is not None:
                raise ValueError("evaluated operations are out of observation order")
            pending = _build_frame(
                step=len(frames) + 1,
                line=event["_line_number"],
                observation=observations[camp],
                operation=operation,
                player_identity=players[camp],
            )
        else:
            if pending is not None:
                intervening_trace_lines.append(event["_line_number"])
            _audit_opponent_operation(opponent_stats, observations[camp], operation)
        observation_ready[camp] = False

    if not match_end_seen or match_end_scores is None or terminal_frame_scores is None:
        raise ValueError("trace is missing a proved terminal Judge event")
    if match_end_scores != terminal_frame_scores or match_end_scores != result_info["scores"]:
        raise ValueError("trace terminal scores disagree with result")
    if replay_info["game_end_winner"] != result_info["winner"]:
        raise ValueError("binary replay GameEnd winner disagrees with result")
    if pending is None:
        raise ValueError("trace terminal boundary has no final evaluated decision")
    terminal_event = {
        "status": "observed_before_next_evaluated_observation",
        "outcome": result_info["outcome"],
        "winner": result_info["winner"],
        "scores": result_info["scores"],
        "game_end_round": replay_info["game_end_round"],
        "state_after_recorded": False,
    }
    frames.append(
        RealJudgeDecisionFrame(
            **pending,
            state_after=None,
            next_observed_state=None,
            next_observation_trace_line=None,
            transition_scope="terminal_before_next_evaluated_observation",
            intervening_action_count=len(intervening_trace_lines),
            intervening_trace_lines=tuple(intervening_trace_lines),
            terminal_event=_freeze(terminal_event),
        )
    )
    for expected_step, frame in enumerate(frames, 1):
        if frame.decision_step != expected_step:
            raise ValueError("real DecisionFrame steps are not continuous")
    for left, right in zip(frames, frames[1:]):
        observed_successor = (
            left.state_after
            if left.state_after is not None
            else left.next_observed_state
        )
        if observed_successor != right.state_before:
            raise ValueError("real DecisionFrame observation chain is not continuous")

    direct_state_after_count = sum(
        frame.state_after is not None for frame in frames
    )
    intervened_transition_count = sum(
        frame.next_observed_state is not None for frame in frames
    )
    terminal_without_state_after_count = sum(
        frame.terminal_event is not None and frame.state_after is None
        for frame in frames
    )

    opponent_compatible = (
        opponent_stats["command_rejected"] == 0
        and opponent_stats["support_complete"] == opponent_stats["operations"]
        and opponent_stats["chosen_outside_support"] == 0
        and opponent_stats["chosen_in_support"] == opponent_stats["operations"]
    )
    opponent_report = {
        "camp": opponent,
        "name": players[opponent]["name"],
        **opponent_stats,
        "compatible": opponent_compatible,
        "handling": "excluded_from_evaluated_camp_only_packet",
    }
    if manifest["decision_scope"] != "evaluated_camp_only":
        reason = "opponent protocol is incompatible" if not opponent_compatible else "all-camp scope is unsupported"
        raise ValueError(f"all-camp DecisionFrame conversion fails closed: {reason}")

    expected = manifest["expected_adapter_output"]
    actual = {
        "decision_frames": len(frames),
        "support_complete": len(frames),
        "chosen_in_support": len(frames),
        "opponent_protocol_compatible": opponent_compatible,
    }
    if actual != expected:
        raise ValueError("real adapter output disagrees with approved manifest")

    artifact_sha = {
        name: manifest["artifacts"][name]["sha256"]
        for name in ("replay", "trace", "result")
    }
    artifact_sha["run_intent"] = manifest["run_intent"]["sha256"]
    return RealJudgeReplayPacket(
        schema_version=REAL_JUDGE_PACKET_SCHEMA_VERSION,
        authoritative=authoritative,
        evidence_scope=(
            "approved_real_judge_training_replay"
            if authoritative
            else "candidate_unapproved_real_judge_evidence"
        ),
        manifest_sha256=manifest_sha256,
        role="train",
        case_id=manifest["case_id"],
        evaluated_camp=evaluated,
        players=tuple(_freeze(player) for player in players),
        capture_identity=_freeze(manifest["capture_identity"]),
        adapter_identity=_freeze(manifest["adapter_identity"]),
        environment=_freeze(manifest["environment"]),
        artifact_sha256=_freeze(artifact_sha),
        decision_frames=tuple(frames),
        terminal=_freeze(terminal_event),
        opponent_compatibility=_freeze(opponent_report),
        support_complete_count=len(frames),
        chosen_in_support_count=len(frames),
        direct_state_after_count=direct_state_after_count,
        intervened_transition_count=intervened_transition_count,
        terminal_without_state_after_count=terminal_without_state_after_count,
    )


def _read_exact_artifact(
    root: Any,
    descriptor: Mapping[str, Any],
    label: str,
    limit: int,
) -> bytes:
    try:
        payload = root.read(descriptor["path"], f"{label} path", limit)
    except OSError as exc:
        raise ValueError(f"{label} is unsafe or unavailable") from exc
    if len(payload) != descriptor["bytes"]:
        raise ValueError(f"{label} bytes mismatch")
    if _sha256(payload) != descriptor["sha256"]:
        raise ValueError(f"{label} SHA mismatch")
    return payload


def _bind_root(root: Path, label: str) -> Path:
    raw = synthetic._bind_exact_path(root, label, allow_string=False)
    synthetic._validate_lexical_filesystem_path(raw, label)
    if not os.path.isabs(raw):
        raise ValueError(f"{label} must be absolute")
    return Path(os.path.abspath(raw))


def _read_and_validate(
    manifest_root: Path,
    manifest_relative: str,
    evidence_root: Path,
    *,
    require_approval: bool,
    packet_authoritative: bool = False,
    expected_manifest_bytes: bytes | None = None,
    expected_manifest_root_identity: tuple[Any, ...] | None = None,
    expected_evidence_root_identity: tuple[Any, ...] | None = None,
) -> tuple[
    RealJudgeReplayPacket,
    bytes,
    str,
    tuple[Any, ...],
    tuple[Any, ...],
]:
    with synthetic._open_approved_root(
        manifest_root, expected_manifest_root_identity
    ) as manifest_handle:
        manifest_bytes = manifest_handle.read(
            manifest_relative,
            "real replay manifest",
            MAX_REAL_JUDGE_MANIFEST_BYTES,
        )
        if expected_manifest_bytes is not None and manifest_bytes != expected_manifest_bytes:
            raise ValueError("real replay manifest bytes changed after preflight")
        manifest_sha = _sha256(manifest_bytes)
        if (
            require_approval
            and manifest_sha
            not in approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS
        ):
            raise ValueError("real replay manifest is not independently approved")
        manifest = _validate_manifest(
            _parse_json_object(
                manifest_bytes, "real replay manifest", canonical=True
            )
        )
        with synthetic._open_approved_root(
            evidence_root, expected_evidence_root_identity
        ) as evidence_handle:
            intent_bytes = _read_exact_artifact(
                evidence_handle,
                manifest["run_intent"],
                "run intent",
                MAX_REAL_JUDGE_RUN_INTENT_BYTES,
            )
            replay_bytes = _read_exact_artifact(
                evidence_handle,
                manifest["artifacts"]["replay"],
                "replay",
                MAX_REAL_JUDGE_REPLAY_BYTES,
            )
            trace_bytes = _read_exact_artifact(
                evidence_handle,
                manifest["artifacts"]["trace"],
                "trace",
                MAX_REAL_JUDGE_TRACE_BYTES,
            )
            result_bytes = _read_exact_artifact(
                evidence_handle,
                manifest["artifacts"]["result"],
                "result",
                MAX_REAL_JUDGE_RESULT_BYTES,
            )
            intent = _parse_json_object(intent_bytes, "run intent")
            intent_identity = _validate_run_intent(intent, manifest)
            replay_info = _parse_binary_replay(
                replay_bytes, manifest["environment"]
            )
            result_info = _validate_result(
                _parse_json_object(result_bytes, "real Judge result"),
                manifest,
                intent_identity,
            )
            packet = _build_packet_from_trace(
                trace_bytes,
                manifest,
                manifest_sha,
                replay_info,
                result_info,
                intent_identity,
                authoritative=packet_authoritative,
            )
            return (
                packet,
                manifest_bytes,
                manifest_sha,
                manifest_handle.identity,
                evidence_handle.identity,
            )


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class RealJudgeReplayReadingContext:
    """Issuer-only proof for an approved real Judge replay triad."""

    _manifest_root: str
    _manifest_root_identity: tuple[Any, ...]
    _manifest_relative_path: str
    _evidence_root: str
    _evidence_root_identity: tuple[Any, ...]
    _manifest_bytes: bytes
    _manifest_sha256: str

    def __new__(cls):
        raise TypeError("RealJudgeReplayReadingContext is issuer-only")

    @property
    def manifest(self) -> Mapping[str, Any]:
        return _freeze(
            _parse_json_object(
                self._manifest_bytes, "context real replay manifest", canonical=True
            )
        )


def _context_snapshot(context: RealJudgeReplayReadingContext) -> tuple[Any, ...]:
    if type(context._manifest_root) is not str or type(context._evidence_root) is not str:
        raise ValueError("real replay context root fields are invalid")
    manifest_identity = synthetic._strict_root_identity(
        context._manifest_root_identity
    )
    evidence_identity = synthetic._strict_root_identity(
        context._evidence_root_identity
    )
    relative = "/".join(
        synthetic._relative_parts(
            context._manifest_relative_path, "context real replay manifest path"
        )
    )
    if type(context._manifest_bytes) is not bytes:
        raise ValueError("real replay context manifest bytes are invalid")
    digest = _strict_sha256(context._manifest_sha256, "context real replay manifest")
    return (
        context._manifest_root,
        manifest_identity,
        relative,
        context._evidence_root,
        evidence_identity,
        context._manifest_bytes,
        digest,
    )


def _build_context_authority():
    registry: weakref.WeakKeyDictionary[
        RealJudgeReplayReadingContext, tuple[Any, ...]
    ] = weakref.WeakKeyDictionary()

    def preflight(
        manifest_path: Path | str,
        *,
        manifest_root: Path,
        evidence_root: Path,
    ) -> RealJudgeReplayReadingContext:
        bound_manifest_root, relative = synthetic._manifest_location(
            manifest_root, manifest_path
        )
        bound_evidence_root = _bind_root(evidence_root, "real replay evidence root")
        (
            packet,
            payload,
            digest,
            manifest_identity,
            evidence_identity,
        ) = _read_and_validate(
            bound_manifest_root,
            relative,
            bound_evidence_root,
            require_approval=True,
        )
        if packet.role != "train":
            raise ValueError("real replay role changed during preflight")
        context = object.__new__(RealJudgeReplayReadingContext)
        object.__setattr__(context, "_manifest_root", os.fspath(bound_manifest_root))
        object.__setattr__(
            context,
            "_manifest_root_identity",
            synthetic._strict_root_identity(manifest_identity),
        )
        object.__setattr__(context, "_manifest_relative_path", relative)
        object.__setattr__(context, "_evidence_root", os.fspath(bound_evidence_root))
        object.__setattr__(
            context,
            "_evidence_root_identity",
            synthetic._strict_root_identity(evidence_identity),
        )
        object.__setattr__(context, "_manifest_bytes", bytes(payload))
        object.__setattr__(context, "_manifest_sha256", digest)
        registry[context] = _context_snapshot(context)
        return context

    def issued(context: Any) -> bool:
        if type(context) is not RealJudgeReplayReadingContext:
            return False
        try:
            snapshot = registry.get(context)
            return snapshot is not None and snapshot == _context_snapshot(context)
        except (AttributeError, TypeError, ValueError):
            return False

    return preflight, issued


preflight_real_judge_replay, _is_issued_context = _build_context_authority()
del _build_context_authority


def open_real_judge_replay(
    context: RealJudgeReplayReadingContext,
) -> RealJudgeReplayPacket:
    if not _is_issued_context(context):
        raise ValueError("trusted issued real replay context is required")
    if (
        context._manifest_sha256
        not in approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS
    ):
        raise ValueError("real replay context digest is no longer approved")
    packet, payload, digest, manifest_identity, evidence_identity = _read_and_validate(
        Path(context._manifest_root),
        context._manifest_relative_path,
        Path(context._evidence_root),
        require_approval=True,
        packet_authoritative=True,
        expected_manifest_bytes=context._manifest_bytes,
        expected_manifest_root_identity=context._manifest_root_identity,
        expected_evidence_root_identity=context._evidence_root_identity,
    )
    if (
        payload != context._manifest_bytes
        or digest != context._manifest_sha256
        or not synthetic._root_identities_match(
            manifest_identity, context._manifest_root_identity
        )
        or not synthetic._root_identities_match(
            evidence_identity, context._evidence_root_identity
        )
        or packet.role != "train"
    ):
        raise ValueError("real replay context identity changed")
    _AUTHORIZED_REAL_REPLAY_PACKETS.add(packet)
    return packet


def require_authoritative_real_replay_packet(
    packet: RealJudgeReplayPacket,
) -> RealJudgeReplayPacket:
    """Production lifecycle gate for real Judge replay packets."""

    if (
        type(packet) is not RealJudgeReplayPacket
        or packet.authoritative is not True
        or packet.evidence_scope != "approved_real_judge_training_replay"
        or packet not in _AUTHORIZED_REAL_REPLAY_PACKETS
        or packet.manifest_sha256
        not in approvals.APPROVED_REAL_JUDGE_REPLAY_MANIFESTS
    ):
        raise ValueError("an issuer-opened authoritative real replay packet is required")
    return packet


def audit_real_judge_replay_candidate(
    manifest_path: Path | str,
    *,
    manifest_root: Path,
    evidence_root: Path,
) -> RealJudgeReplayCandidate:
    """Structurally validate an unapproved candidate without authorizing it."""

    bound_manifest_root, relative = synthetic._manifest_location(
        manifest_root, manifest_path
    )
    bound_evidence_root = _bind_root(evidence_root, "real replay evidence root")
    packet, _payload, digest, _manifest_identity, _evidence_identity = (
        _read_and_validate(
            bound_manifest_root,
            relative,
            bound_evidence_root,
            require_approval=False,
        )
    )
    return RealJudgeReplayCandidate(False, digest, packet)


__all__ = [
    "MAX_REAL_JUDGE_MANIFEST_BYTES",
    "MAX_REAL_JUDGE_REPLAY_BYTES",
    "MAX_REAL_JUDGE_RESULT_BYTES",
    "MAX_REAL_JUDGE_RUN_INTENT_BYTES",
    "MAX_REAL_JUDGE_TRACE_BYTES",
    "RATIONALE_STATUS",
    "REAL_JUDGE_DECISION_FRAME_SCHEMA_VERSION",
    "REAL_JUDGE_MANIFEST_SCHEMA_VERSION",
    "REAL_JUDGE_PACKET_SCHEMA_VERSION",
    "RealJudgeDecisionFrame",
    "RealJudgeReplayCandidate",
    "RealJudgeReplayPacket",
    "RealJudgeReplayReadingContext",
    "audit_real_judge_replay_candidate",
    "canonical_real_replay_json_bytes",
    "current_adapter_identity",
    "open_real_judge_replay",
    "preflight_real_judge_replay",
    "require_authoritative_real_replay_packet",
]

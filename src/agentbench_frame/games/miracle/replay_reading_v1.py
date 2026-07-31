"""Train-only, synthetic-only structured Replay Reading for 24_miracle.

The module parses approved canonical bytes into an ordered timeline. It never
starts a Judge, policy, Provider, runner, workspace, store, log, or session.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

from agentbench_frame.eval.measurement import canonical_state_id
from agentbench_frame.games.miracle.research_protocol import (
    SEED_MAX,
    SEED_MIN,
    build_action_support,
    canonical_command,
    command_action_id,
    enumerate_legal_commands,
)


REPLAY_READING_MANIFEST_SCHEMA_VERSION = "24-miracle-replay-reading-manifest-v1"
SYNTHETIC_REPLAY_SCHEMA_VERSION = "24-miracle-synthetic-replay-v1"
RATIONALE_STATUS = "not_recorded"

# Production is intentionally empty. Tests may temporarily monkeypatch it.
APPROVED_TRAINING_REPLAY_MANIFESTS: frozenset[str] = frozenset()
_CONTEXT_ISSUER = object()


def canonical_replay_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must be strict UTF-8 without BOM")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        canonical = canonical_replay_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid JSON data") from exc
    if payload != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _strict_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_seed(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not SEED_MIN <= value <= SEED_MAX
    ):
        raise ValueError(f"{label} must be a strict frozen-range seed")
    return value


def _strict_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an int or float, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _is_symlink(path: Path) -> bool:
    return path.is_symlink()


def _validated_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_dir() or _is_symlink(root):
        raise ValueError("approved replay root must be a real directory")
    return root.resolve()


def _safe_relative_file(root: Path, relative: Any, label: str) -> Path:
    resolved_root = _validated_root(root)
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a relative path without escape")
    candidate = resolved_root
    for part in path.parts:
        candidate = candidate / part
        if _is_symlink(candidate):
            raise ValueError(f"{label} contains a symlink component")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes approved root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is missing or not a file")
    return resolved


def _supplied_manifest(root: Path, path: Path) -> tuple[Path, str]:
    resolved_root = _validated_root(root)
    if not isinstance(path, Path):
        raise ValueError("manifest path must be a Path")
    try:
        relative = path.absolute().relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError("manifest path escapes approved root") from exc
    return _safe_relative_file(resolved_root, relative, "manifest path"), relative


def _validate_case(value: Any) -> dict[str, Any]:
    fields = {
        "case_id", "opponent", "evaluated_agent_camp", "map_type", "day_time", "repeat"
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("case identity fields are incomplete or extra")
    result = copy.deepcopy(dict(value))
    _strict_text(result["case_id"], "case ID")
    _strict_text(result["opponent"], "opponent")
    for field in ("evaluated_agent_camp", "map_type", "day_time"):
        if not isinstance(result[field], int) or isinstance(result[field], bool) or result[field] not in (0, 1):
            raise ValueError(f"case {field} must be strict integer 0 or 1")
    if not isinstance(result["repeat"], int) or isinstance(result["repeat"], bool) or result["repeat"] not in (1, 2, 3):
        raise ValueError("case repeat must be strict integer 1, 2, or 3")
    return result


def _validate_seeds(value: Any) -> dict[str, int]:
    fields = ("logic_seed", "evaluated_agent_seed", "opponent_seed")
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError("seed bundle is incomplete or extra")
    return {field: _strict_seed(value[field], field) for field in fields}


def _validate_policy(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"version", "source_sha256"}:
        raise ValueError("acting policy identity is incomplete or extra")
    return {
        "version": _strict_text(value["version"], "policy version"),
        "source_sha256": _strict_sha(value["source_sha256"], "policy source"),
    }


def _validate_champion(value: Any) -> dict[str, str]:
    fields = {"logical_id", "version", "descriptor_sha256", "artifact_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("champion identity is incomplete or extra")
    return {
        "logical_id": _strict_text(value["logical_id"], "champion logical ID"),
        "version": _strict_text(value["version"], "champion version"),
        "descriptor_sha256": _strict_sha(value["descriptor_sha256"], "champion descriptor"),
        "artifact_sha256": _strict_sha(value["artifact_sha256"], "champion artifact"),
    }


def _validate_state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"canonical_state_id", "observation"}:
        raise ValueError(f"{label} must contain state identity and observation")
    observation = value["observation"]
    if not isinstance(observation, Mapping):
        raise ValueError(f"{label} observation must be an object")
    if value["canonical_state_id"] != canonical_state_id(observation):
        raise ValueError(f"{label} canonical state identity mismatch")
    return copy.deepcopy(dict(value))


def _validate_evaluated_camp(
    observation: Mapping[str, Any], expected_camp: int, label: str
) -> None:
    camp = observation.get("camp")
    if type(camp) is not int or camp not in (0, 1):
        raise ValueError(f"{label} observation camp must be strict integer 0 or 1")
    if camp != expected_camp:
        raise ValueError(f"{label} observation camp does not match evaluated agent camp")


@dataclass(frozen=True)
class DecisionFrame:
    decision_step: int
    state_before: Mapping[str, Any]
    action_support: Mapping[str, Any]
    chosen_action: Mapping[str, Any]
    acting_identity_refs: Mapping[str, str]
    state_after: Mapping[str, Any]
    reward: float
    outcome: str
    terminated: bool
    truncated: bool
    case_identity_ref: str
    seed_bundle_ref: str
    rationale_status: str = RATIONALE_STATUS


@dataclass(frozen=True)
class ReplayPacket:
    manifest_sha256: str
    replay_sha256: str
    match_plan_sha256: str
    case_identity: Mapping[str, Any]
    role: str
    seeds: Mapping[str, int]
    acting_policy: Mapping[str, str]
    champion: Mapping[str, str]
    decision_frames: tuple[DecisionFrame, ...]
    terminal: Mapping[str, Any]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class ReplayReadingContext:
    """Issuer-only immutable preflight proof; it never stores a ReplayPacket."""

    _issuer: object
    _approved_root: str
    _manifest_relative_path: str
    _manifest_bytes: bytes
    _manifest_sha256: str

    def __new__(cls, issuer: object = None):
        if issuer is not _CONTEXT_ISSUER:
            raise TypeError("ReplayReadingContext can only be issued by trusted preflight")
        return object.__new__(cls)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return _deep_freeze(_strict_object_bytes(self._manifest_bytes, "context manifest"))


def _issue_context(root: Path, relative: str, payload: bytes, digest: str) -> ReplayReadingContext:
    context = ReplayReadingContext.__new__(ReplayReadingContext, _CONTEXT_ISSUER)
    object.__setattr__(context, "_issuer", _CONTEXT_ISSUER)
    object.__setattr__(context, "_approved_root", str(root.resolve()))
    object.__setattr__(context, "_manifest_relative_path", relative)
    object.__setattr__(context, "_manifest_bytes", bytes(payload))
    object.__setattr__(context, "_manifest_sha256", digest)
    return context


def _identity_refs(policy: Mapping[str, str], champion: Mapping[str, str]) -> dict[str, str]:
    return {
        "acting_policy_version": policy["version"],
        "policy_source_sha256": policy["source_sha256"],
        "champion_logical_id": champion["logical_id"],
        "champion_version": champion["version"],
        "champion_descriptor_sha256": champion["descriptor_sha256"],
        "champion_artifact_sha256": champion["artifact_sha256"],
    }


def _validate_frame(value: Any, expected_step: int, *, case, seeds, policy, champion) -> DecisionFrame:
    fields = {
        "decision_step", "state_before", "action_support", "chosen_action",
        "acting_identity_refs", "state_after", "reward", "outcome", "terminated",
        "truncated", "case_identity_ref", "seed_bundle_ref", "rationale_status",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DecisionFrame fields are incomplete or extra")
    step = value["decision_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step != expected_step:
        raise ValueError("decision_step must be strict integers 1..N")
    before = _validate_state(value["state_before"], "state_before")
    after = _validate_state(value["state_after"], "state_after")
    expected_camp = case["evaluated_agent_camp"]
    _validate_evaluated_camp(
        before["observation"], expected_camp, "state_before"
    )
    _validate_evaluated_camp(
        after["observation"], expected_camp, "state_after"
    )
    support = build_action_support(enumerate_legal_commands(before["observation"]))
    actions = [{"action_id": item.action_id, "command": item.action} for item in support.actions]
    supplied = value["action_support"]
    trusted = {"schema_version": support.schema_version, "support_id": support.support_id, "actions": actions}
    if (
        type(supplied) is not dict
        or set(supplied) != set(trusted)
        or canonical_replay_json_bytes(supplied)
        != canonical_replay_json_bytes(trusted)
    ):
        raise ValueError("DecisionFrame does not contain trusted ActionSupport")
    chosen = value["chosen_action"]
    if not isinstance(chosen, Mapping) or set(chosen) != {"action_id", "command"}:
        raise ValueError("chosen action is incomplete")
    command = canonical_command(chosen["command"])
    by_id = {item["action_id"]: item["command"] for item in actions}
    if command_action_id(command) != chosen["action_id"] or by_id.get(chosen["action_id"]) != command:
        raise ValueError("chosen action is outside trusted ActionSupport")
    if value["acting_identity_refs"] != _identity_refs(policy, champion):
        raise ValueError("acting identity refs mismatch")
    if value["case_identity_ref"] != case["case_id"]:
        raise ValueError("DecisionFrame case identity ref mismatch")
    seed_ref = hashlib.sha256(canonical_replay_json_bytes(dict(seeds))).hexdigest()
    if value["seed_bundle_ref"] != seed_ref:
        raise ValueError("DecisionFrame seed bundle ref mismatch")
    if value["rationale_status"] != RATIONALE_STATUS:
        raise ValueError("rationale must remain not_recorded")
    if not isinstance(value["terminated"], bool) or not isinstance(value["truncated"], bool):
        raise ValueError("terminal flags must be booleans")
    if value["outcome"] not in {"win", "loss", "draw", "ongoing"}:
        raise ValueError("DecisionFrame outcome is invalid")
    return DecisionFrame(
        step, before, copy.deepcopy(dict(supplied)), copy.deepcopy(dict(chosen)),
        copy.deepcopy(dict(value["acting_identity_refs"])), after,
        _strict_number(value["reward"], "DecisionFrame reward"), value["outcome"],
        value["terminated"], value["truncated"], value["case_identity_ref"],
        value["seed_bundle_ref"],
    )


def _validate_replay(value: Mapping[str, Any], manifest: Mapping[str, Any], manifest_sha: str) -> ReplayPacket:
    fields = {
        "schema_version", "match_plan_sha256", "case_identity", "role", "seeds",
        "acting_policy", "champion", "rationale_status", "decision_count",
        "terminal_recorded", "decision_frames", "terminal",
    }
    if set(value) != fields or value.get("schema_version") != SYNTHETIC_REPLAY_SCHEMA_VERSION:
        raise ValueError("synthetic replay schema or fields mismatch")
    case = _validate_case(value["case_identity"])
    seeds = _validate_seeds(value["seeds"])
    policy = _validate_policy(value["acting_policy"])
    champion = _validate_champion(value["champion"])
    if any((
        value["match_plan_sha256"] != manifest["match_plan_sha256"],
        case != manifest["case_identity"], value["role"] != manifest["role"],
        seeds != manifest["seeds"], policy != manifest["acting_policy"],
        champion != manifest["champion"],
    )):
        raise ValueError("replay and approved manifest identity mismatch")
    frames = value["decision_frames"]
    count = value["decision_count"]
    if not isinstance(frames, list) or not frames or not isinstance(count, int) or isinstance(count, bool) or count != len(frames):
        raise ValueError("synthetic replay decision count mismatch")
    if value["terminal_recorded"] is not True or value["rationale_status"] != RATIONALE_STATUS:
        raise ValueError("synthetic replay terminal/rationale marker is invalid")
    parsed = tuple(
        _validate_frame(frame, index, case=case, seeds=seeds, policy=policy, champion=champion)
        for index, frame in enumerate(frames, start=1)
    )
    for left, right in zip(parsed, parsed[1:]):
        if left.state_after != right.state_before:
            raise ValueError("DecisionFrame state chain is not continuous")
        if left.terminated or left.truncated or left.outcome != "ongoing":
            raise ValueError("only the last DecisionFrame may be terminal")
    terminal = value["terminal"]
    terminal_fields = {"outcome", "reward", "termination_reason", "terminated", "truncated"}
    if not isinstance(terminal, Mapping) or set(terminal) != terminal_fields:
        raise ValueError("terminal record is incomplete or extra")
    terminal_reward = _strict_number(terminal["reward"], "terminal reward")
    _strict_text(terminal["termination_reason"], "termination reason")
    if terminal["outcome"] not in {"win", "loss", "draw"} or not isinstance(terminal["terminated"], bool) or not isinstance(terminal["truncated"], bool):
        raise ValueError("terminal record is invalid")
    last = parsed[-1]
    if not (last.terminated or last.truncated) or (last.outcome, last.reward, last.terminated, last.truncated) != (terminal["outcome"], terminal_reward, terminal["terminated"], terminal["truncated"]):
        raise ValueError("terminal record disagrees with the last DecisionFrame")
    return ReplayPacket(
        manifest_sha, manifest["replay_sha256"], manifest["match_plan_sha256"],
        case, "train", seeds, policy, champion, parsed, copy.deepcopy(dict(terminal)),
    )


def _read_and_validate(root: Path, manifest_relative: str, expected_bytes: bytes | None = None) -> tuple[ReplayPacket, bytes, str]:
    manifest_file = _safe_relative_file(root, manifest_relative, "manifest path")
    manifest_bytes = manifest_file.read_bytes()
    if expected_bytes is not None and manifest_bytes != expected_bytes:
        raise ValueError("manifest bytes changed after preflight")
    manifest = _strict_object_bytes(manifest_bytes, "replay manifest")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest not in APPROVED_TRAINING_REPLAY_MANIFESTS:
        raise ValueError("replay manifest is not independently approved")
    fields = {
        "schema_version", "replay_artifact_path", "replay_sha256", "match_plan_sha256",
        "case_identity", "role", "seeds", "acting_policy", "champion",
    }
    if set(manifest) != fields or manifest.get("schema_version") != REPLAY_READING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("replay manifest schema or fields mismatch")
    manifest["replay_sha256"] = _strict_sha(manifest["replay_sha256"], "replay")
    manifest["match_plan_sha256"] = _strict_sha(manifest["match_plan_sha256"], "match plan")
    manifest["case_identity"] = _validate_case(manifest["case_identity"])
    manifest["seeds"] = _validate_seeds(manifest["seeds"])
    manifest["acting_policy"] = _validate_policy(manifest["acting_policy"])
    manifest["champion"] = _validate_champion(manifest["champion"])
    if manifest["role"] != "train":
        raise ValueError("Replay Reading accepts only independently approved role=train")
    replay_file = _safe_relative_file(root, manifest["replay_artifact_path"], "replay path")
    replay_bytes = replay_file.read_bytes()
    if hashlib.sha256(replay_bytes).hexdigest() != manifest["replay_sha256"]:
        raise ValueError("replay artifact SHA mismatch")
    document = _strict_object_bytes(replay_bytes, "synthetic replay")
    return _validate_replay(document, manifest, digest), manifest_bytes, digest


def preflight_replay_reading(manifest_path: Path, *, approved_root: Path) -> ReplayReadingContext:
    manifest_file, relative = _supplied_manifest(approved_root, manifest_path)
    packet, payload, digest = _read_and_validate(approved_root, relative)
    if packet.role != "train" or manifest_file.read_bytes() != payload:
        raise ValueError("replay manifest changed during preflight")
    return _issue_context(approved_root, relative, payload, digest)


def open_replay_reading(context: ReplayReadingContext) -> ReplayPacket:
    try:
        trusted = (
            type(context) is ReplayReadingContext
            and context._issuer is _CONTEXT_ISSUER
            and isinstance(context._manifest_bytes, bytes)
            and hashlib.sha256(context._manifest_bytes).hexdigest() == context._manifest_sha256
        )
    except (AttributeError, TypeError):
        trusted = False
    if not trusted:
        raise ValueError("trusted issued Replay Reading context is required")
    _strict_object_bytes(context._manifest_bytes, "context manifest")
    if context._manifest_sha256 not in APPROVED_TRAINING_REPLAY_MANIFESTS:
        raise ValueError("context manifest digest is no longer approved")
    root = Path(context._approved_root)
    packet, payload, digest = _read_and_validate(
        root, context._manifest_relative_path, context._manifest_bytes
    )
    if payload != context._manifest_bytes or digest != context._manifest_sha256 or packet.role != "train":
        raise ValueError("context manifest bytes, digest, or role changed")
    return packet


def render_replay_timeline(context: ReplayReadingContext) -> str:
    """Reopen issuer-created context and render only the revalidated packet."""

    if type(context) is not ReplayReadingContext:
        raise TypeError("renderer requires an issued Replay Reading context")
    packet = open_replay_reading(context)
    case = packet.case_identity
    seeds = packet.seeds

    def encoded_text(value: Any, label: str) -> str:
        return quote(_strict_text(value, label), safe="-._~")

    def encoded_command(value: Mapping[str, Any]) -> str:
        canonical = canonical_replay_json_bytes(value).decode("utf-8")
        return quote(canonical, safe="-._~")

    lines = [
        " | ".join((
            f"case={encoded_text(case['case_id'], 'case ID')}",
            f"role={encoded_text(packet.role, 'role')}",
            f"logic_seed={seeds['logic_seed']}",
            f"evaluated_agent_seed={seeds['evaluated_agent_seed']}",
            f"opponent_seed={seeds['opponent_seed']}",
        ))
    ]
    for frame in packet.decision_frames:
        command = encoded_command(frame.chosen_action["command"])
        lines.append(" | ".join((
            f"step={frame.decision_step}",
            f"state={encoded_text(frame.state_before['canonical_state_id'], 'state ID')}",
            f"legal_actions={len(frame.action_support['actions'])}",
            f"chosen={encoded_text(frame.chosen_action['action_id'], 'action ID')}:{command}",
            "policy="
            f"{encoded_text(packet.acting_policy['version'], 'policy version')}@"
            f"{encoded_text(packet.acting_policy['source_sha256'], 'policy source')}",
            "champion="
            f"{encoded_text(packet.champion['logical_id'], 'champion logical ID')}@"
            f"{encoded_text(packet.champion['version'], 'champion version')}",
            f"reward={frame.reward}",
            f"outcome={encoded_text(frame.outcome, 'frame outcome')}",
            "rationale_status="
            f"{encoded_text(frame.rationale_status, 'rationale status')}",
        )))
    terminal = packet.terminal
    lines.append(" | ".join((
        "terminal",
        f"outcome={encoded_text(terminal['outcome'], 'terminal outcome')}",
        f"reward={terminal['reward']}",
        f"terminated={str(terminal['terminated']).lower()}",
        f"truncated={str(terminal['truncated']).lower()}",
        "termination_reason="
        f"{encoded_text(terminal['termination_reason'], 'termination reason')}",
    )))
    return "\n".join(lines) + "\n"


__all__ = [
    "APPROVED_TRAINING_REPLAY_MANIFESTS",
    "RATIONALE_STATUS",
    "REPLAY_READING_MANIFEST_SCHEMA_VERSION",
    "SYNTHETIC_REPLAY_SCHEMA_VERSION",
    "DecisionFrame",
    "ReplayPacket",
    "ReplayReadingContext",
    "canonical_replay_json_bytes",
    "preflight_replay_reading",
    "open_replay_reading",
    "render_replay_timeline",
]

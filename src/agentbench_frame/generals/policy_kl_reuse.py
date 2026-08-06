"""Verify and materialize immutable inputs from a completed policy-KL run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

from .measurement_state import measurement_state_id
from .models import PolicyKLExtensionConfig
from .policy_kl_math import (
    CanonicalAction,
    compute_controlled_policy_kl,
)


EXPECTED_ACTION_SPACE_SPEC_ID = (
    "a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e"
)
EVENT_METADATA = frozenset({
    "schema_version",
    "event_id",
    "event_type",
    "event",
    "run_id",
    "created_at",
    "timestamp",
})
QUALITY_DEFECT_FIELDS = (
    "malformed_lines",
    "invalid_events",
    "unknown_event_types",
    "duplicate_event_ids",
    "missing_event_ids",
    "missing_run_ids",
)


class PolicyKLSourceError(ValueError):
    """The completed source run does not match its frozen reuse contract."""


@dataclass(frozen=True)
class VerifiedPolicyKLSource:
    root: Path
    tree_hash: str
    reference_records: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    actions: dict[tuple[str, str], CanonicalAction]
    prior_facts: tuple[dict[str, Any], ...]
    reuse_events: tuple[dict[str, Any], ...]
    kl_source_events: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ]
    prior_metric: dict[str, Any]


@dataclass(frozen=True)
class ArtifactReceipt:
    source_relative_path: str
    destination_relative_path: str
    semantic_role: str
    sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyKLSourceError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyKLSourceError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PolicyKLSourceError(f"cannot read JSONL artifact {path}: {exc}") from exc
    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolicyKLSourceError(
                f"malformed JSONL artifact {path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PolicyKLSourceError(
                f"JSONL artifact {path} line {line_number} must be an object"
            )
        records.append(value)
    return tuple(records)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_tree_hash(root: Path) -> str:
    """Hash regular files using the frozen sorted sha256sum stream format."""

    root = Path(root)
    if not root.is_dir():
        raise PolicyKLSourceError(f"source run directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise PolicyKLSourceError("source path contains a newline")
        digest.update(
            f"{_file_hash(path)}  ./{relative}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _scientific_payload(
    event: dict[str, Any],
    *,
    preserve_run_id: bool = False,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in event.items()
        if key not in EVENT_METADATA or (preserve_run_id and key == "run_id")
    }


def _reused_scientific_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = _scientific_payload(event)
    for key in (
        "source_event_id", "source_run_id", "reuse_status",
        "materialization_measurement_id",
    ):
        payload.pop(key, None)
    return payload


def _require_clean_quality(
    root: Path,
    summary: dict[str, Any],
) -> None:
    computed = inspect_event_file(root / "events.jsonl").to_dict()
    saved = _read_json(root / "quality.json")
    summary_quality = summary.get("event_quality")
    if saved != computed or summary_quality != computed:
        raise PolicyKLSourceError("source event quality receipts do not match")
    if (
        computed.get("total_lines") != computed.get("valid_events")
        or any(computed.get(field) != 0 for field in QUALITY_DEFECT_FIELDS)
        or computed.get("warnings") != []
    ):
        raise PolicyKLSourceError("source event quality is not clean")


def _reference_records(
    root: Path,
    config: PolicyKLExtensionConfig,
) -> tuple[dict[str, Any], ...]:
    spec = _read_json(root / "benchmark/reference-state-spec.json")
    if (
        not isinstance(spec.get("measurement_id"), str)
        or not spec.get("measurement_id")
        or spec.get("opponent_id") != config.opponent_id
        or tuple(spec.get("seeds") or ()) != config.seeds
        or tuple(spec.get("seats") or ()) != config.seats
        or tuple(spec.get("decision_numbers") or ())
        != config.decision_numbers
    ):
        raise PolicyKLSourceError("source reference-state specification changed")
    state_ids = spec.get("state_ids")
    if (
        not isinstance(state_ids, list)
        or len(state_ids) != 12
        or len(set(state_ids)) != 12
        or not all(isinstance(item, str) and item for item in state_ids)
    ):
        raise PolicyKLSourceError("source reference state IDs must be 12 unique strings")

    records = []
    coordinates = set()
    for state_id in state_ids:
        record = _read_json(root / "reference/states" / f"{state_id}.json")
        snapshot = record.get("snapshot")
        coordinate = (
            record.get("seed"),
            record.get("seat"),
            record.get("decision_number"),
        )
        if (
            record.get("state_id") != state_id
            or not isinstance(snapshot, dict)
            or measurement_state_id(snapshot) != state_id
            or snapshot.get("actor") != record.get("seat")
        ):
            raise PolicyKLSourceError(f"source reference state changed: {state_id}")
        coordinates.add(coordinate)
        records.append(record)
    expected_coordinates = {
        (seed, seat, decision)
        for seed in config.seeds
        for seat in config.seats
        for decision in config.decision_numbers
    }
    if coordinates != expected_coordinates:
        raise PolicyKLSourceError("source reference state coordinates changed")
    return tuple(records)


def _exact_counts(
    root: Path,
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    counts = {}
    raw_records = {}
    for reference in records:
        state_id = reference["state_id"]
        record = _read_json(root / "action-space/counts" / f"{state_id}.json")
        support = record.get("support_size")
        if (
            record.get("measurement_state_id") != state_id
            or record.get("seed") != reference["seed"]
            or record.get("seat") != reference["seat"]
            or record.get("decision_number") != reference["decision_number"]
            or record.get("action_space_spec_id")
            != EXPECTED_ACTION_SPACE_SPEC_ID
            or record.get("status") != "complete"
            or not isinstance(support, str)
            or not support.isdecimal()
            or int(support) < 1
        ):
            raise PolicyKLSourceError(f"source exact count changed: {state_id}")
        counts[state_id] = int(support)
        raw_records[state_id] = record
    if len(list((root / "action-space/counts").glob("*.json"))) != 12:
        raise PolicyKLSourceError("source exact count directory must contain 12 records")
    return counts, raw_records


def _policy_actions(
    root: Path,
    config: PolicyKLExtensionConfig,
    records: tuple[dict[str, Any], ...],
) -> tuple[
    dict[tuple[str, str], CanonicalAction],
    dict[tuple[str, str], dict[str, Any]],
]:
    actions = {}
    raw_records = {}
    snapshotter = LocalWorkspaceSnapshotter()
    for policy in config.history[:-1]:
        policy_root = root / "policies" / policy.version
        manifest = _read_json(policy_root / "manifest.json")
        actual = snapshotter.capture(policy_root / "source")
        if (
            manifest.get("content_hash") != policy.content_hash
            or actual.content_hash != policy.content_hash
            or manifest.get("files") != actual.files
        ):
            raise PolicyKLSourceError(
                f"source {policy.version} policy manifest changed"
            )
        for reference in records:
            state_id = reference["state_id"]
            record = _read_json(policy_root / f"{state_id}.json")
            raw_action = record.get("canonical_action")
            if (
                record.get("version") != policy.version
                or record.get("run_id") != policy.run_id
                or record.get("content_hash") != policy.content_hash
                or record.get("measurement_state_id") != state_id
                or record.get("action_space_spec_id")
                != EXPECTED_ACTION_SPACE_SPEC_ID
                or record.get("status") != "complete"
                or record.get("deterministic") is not True
                or not isinstance(raw_action, list)
                or not raw_action
                or not all(
                    isinstance(command, list)
                    and command
                    and all(type(item) is int for item in command)
                    for command in raw_action
                )
            ):
                raise PolicyKLSourceError(
                    f"source {policy.version} policy action changed: {state_id}"
                )
            action = tuple(tuple(command) for command in raw_action)
            actions[(policy.version, state_id)] = action
            raw_records[(policy.version, state_id)] = record
    expected_action_count = 12 * (len(config.history) - 1)
    if len(actions) != expected_action_count:
        raise PolicyKLSourceError(
            "source policy action set has the wrong record count"
        )
    return actions, raw_records


def _source_events(
    root: Path,
    config: PolicyKLExtensionConfig,
    references: tuple[dict[str, Any], ...],
    count_records: dict[str, dict[str, Any]],
    action_records: dict[tuple[str, str], dict[str, Any]],
    facts: tuple[dict[str, Any], ...],
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[tuple[str, str, str, str], dict[str, Any]],
]:
    events = _read_jsonl(root / "events.jsonl")
    source_version_count = len(config.history) - 1
    source_event_counts = {
        "reference_state_selected": 12,
        "action_space_count": 12,
        "historical_policy_action": 12 * source_version_count,
        "controlled_reference_policy_kl": 48 * (source_version_count - 1),
    }
    grouped: dict[str, list[dict[str, Any]]] = {
        event_type: [] for event_type in source_event_counts
    }
    for record in events:
        event_type = record.get("event_type")
        if event_type not in grouped:
            continue
        if (
            event_type != "historical_policy_action"
            and record.get("run_id") != config.source_run_id
        ):
            raise PolicyKLSourceError("source event run ID changed")
        grouped[event_type].append(record)
    if {
        event_type: len(records)
        for event_type, records in grouped.items()
    } != source_event_counts:
        raise PolicyKLSourceError("source scientific event counts changed")

    reference_measurement_id = _read_json(
        root / "benchmark/reference-state-spec.json"
    )["measurement_id"]
    source_summary = _read_json(root / "summary.json")
    accepted_reference_measurements = {
        reference_measurement_id,
        source_summary.get("source_measurement_id"),
    }
    reference_by_id = {record["state_id"]: record for record in references}
    for record in grouped["reference_state_selected"]:
        state_id = record.get("measurement_state_id")
        reference = reference_by_id.get(state_id)
        expected_path = root / "reference/states" / f"{state_id}.json"
        if (
            reference is None
            or record.get("measurement_id")
            not in accepted_reference_measurements
            or record.get("seed") != reference["seed"]
            or record.get("seat") != reference["seat"]
            or record.get("decision_number") != reference["decision_number"]
            or Path(str(record.get("artifact_ref"))).resolve()
            != expected_path.resolve()
        ):
            raise PolicyKLSourceError("source reference event changed")

    for record in grouped["action_space_count"]:
        state_id = record.get("measurement_state_id")
        if _reused_scientific_payload(record) != count_records.get(state_id):
            raise PolicyKLSourceError("source count event changed")
    for record in grouped["historical_policy_action"]:
        key = (record.get("version"), record.get("measurement_state_id"))
        payload = _reused_scientific_payload(record)
        policy_run_id = payload.pop("policy_run_id", None)
        payload["run_id"] = (
            policy_run_id if policy_run_id is not None else record.get("run_id")
        )
        expected_action = action_records.get(key)
        if expected_action is not None and "artifact_ref" not in expected_action:
            payload.pop("artifact_ref", None)
        if payload != expected_action:
            raise PolicyKLSourceError("source policy action event changed")

    fact_by_key = {
        (
            fact["version_before"],
            fact["version_after"],
            fact["measurement_state_id"],
            fact["epsilon"],
        ): fact
        for fact in facts
    }
    kl_events = {}
    for record in grouped["controlled_reference_policy_kl"]:
        key = (
            record.get("version_before"),
            record.get("version_after"),
            record.get("measurement_state_id"),
            record.get("epsilon"),
        )
        if (
            key in kl_events
            or _reused_scientific_payload(record) != fact_by_key.get(key)
        ):
            raise PolicyKLSourceError("source policy KL event changed")
        kl_events[key] = record
    if set(kl_events) != set(fact_by_key):
        raise PolicyKLSourceError("source policy KL event coordinates changed")
    reuse_events = tuple(
        record
        for event_type in (
            "reference_state_selected",
            "action_space_count",
            "historical_policy_action",
        )
        for record in grouped[event_type]
    )
    return reuse_events, kl_events


def verify_policy_kl_source(
    source_run_dir: Path,
    config: PolicyKLExtensionConfig,
) -> VerifiedPolicyKLSource:
    """Fail closed unless a complete v1 run matches every frozen input."""

    root = Path(source_run_dir).resolve()
    tree_hash = canonical_tree_hash(root)
    if tree_hash != config.source_tree_hash:
        raise PolicyKLSourceError("source run tree hash changed")
    summary = _read_json(root / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("run_id") != config.source_run_id
        or summary.get("measurement_id") != config.source_measurement_id
    ):
        raise PolicyKLSourceError("source summary is not the completed v1 run")
    _require_clean_quality(root, summary)

    action_spec = _read_json(root / "benchmark/action-space-spec.json")
    if action_spec.get("spec_id") != EXPECTED_ACTION_SPACE_SPEC_ID:
        raise PolicyKLSourceError("source action-space specification changed")
    references = _reference_records(root, config)
    counts, count_records = _exact_counts(root, references)
    actions, action_records = _policy_actions(root, config, references)
    versions = tuple(policy.version for policy in config.history[:-1])
    computed = compute_controlled_policy_kl(
        versions=versions,
        reference_state_ids=tuple(
            reference["state_id"] for reference in references
        ),
        counts=counts,
        actions=actions,
        missing_actions={},
        epsilons=config.epsilons,
        primary_epsilon=config.primary_epsilon,
        action_space_spec_id=EXPECTED_ACTION_SPACE_SPEC_ID,
    )
    facts = _read_jsonl(root / "measurement/per-state-kl.jsonl")
    expected_fact_count = 48 * (len(config.history) - 2)
    if len(facts) != expected_fact_count:
        raise PolicyKLSourceError(
            "source measurement has the wrong KL fact count"
        )
    if facts != computed.facts:
        raise PolicyKLSourceError("source per-state policy KL facts changed")
    transition_metric = _read_json(
        root / "measurement/transition-summary.json"
    )
    if (
        transition_metric != computed.metric
        or summary.get("controlled_reference_policy_kl") != computed.metric
    ):
        raise PolicyKLSourceError("source transition summary changed")
    reuse_events, kl_events = _source_events(
        root,
        config,
        references,
        count_records,
        action_records,
        facts,
    )
    return VerifiedPolicyKLSource(
        root=root,
        tree_hash=tree_hash,
        reference_records=references,
        counts=counts,
        actions=actions,
        prior_facts=facts,
        reuse_events=reuse_events,
        kl_source_events=kl_events,
        prior_metric=computed.metric,
    )


def _copy_pairs(source: VerifiedPolicyKLSource) -> list[tuple[Path, Path, str]]:
    root = source.root
    pairs = [
        (
            root / "benchmark/action-space-spec.json",
            Path("benchmark/action-space-spec.json"),
            "action_space_spec",
        ),
    ]
    for directory, role in (
        ("reference/states", "reference_state"),
        ("action-space/counts", "exact_support_count"),
    ):
        for path in sorted((root / directory).glob("*.json")):
            pairs.append((path, path.relative_to(root), role))
    for version in sorted({key[0] for key in source.actions}):
        policy_root = root / "policies" / version
        for path in sorted(
            item
            for item in policy_root.rglob("*")
            if item.is_file() and not item.is_symlink()
        ):
            pairs.append((path, path.relative_to(root), "historical_policy"))
    provenance = {
        "benchmark/action-space-spec.json": "action-space-spec.json",
        "benchmark/reference-state-spec.json": "reference-state-spec.json",
        "events.jsonl": "events.jsonl",
        "measurement/per-state-kl.jsonl": "per-state-kl.jsonl",
        "measurement/transition-summary.json": "transition-summary.json",
        "quality.json": "quality.json",
        "summary.json": "summary.json",
    }
    for source_relative, destination_name in provenance.items():
        pairs.append(
            (
                root / source_relative,
                Path("provenance/source-run") / destination_name,
                "source_run_receipt",
            )
        )
    return pairs


def _materialize_file(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise PolicyKLSourceError(
                f"materialized artifact changed: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def materialize_policy_kl_source(
    source: VerifiedPolicyKLSource,
    destination_run_dir: Path,
) -> tuple[ArtifactReceipt, ...]:
    """Copy the fixed scientific subset and persist byte-level receipts."""

    destination_root = Path(destination_run_dir)
    receipts = []
    for source_path, destination_relative, role in _copy_pairs(source):
        destination_path = destination_root / destination_relative
        _materialize_file(source_path, destination_path)
        digest = _file_hash(source_path)
        if _file_hash(destination_path) != digest:
            raise PolicyKLSourceError(
                f"materialized artifact digest changed: {destination_relative}"
            )
        receipts.append(
            ArtifactReceipt(
                source_relative_path=source_path.relative_to(
                    source.root
                ).as_posix(),
                destination_relative_path=destination_relative.as_posix(),
                semantic_role=role,
                sha256=digest,
            )
        )
    receipts.sort(key=lambda item: item.destination_relative_path)
    receipt_path = destination_root / "provenance/imported-artifacts.jsonl"
    receipt_payload = "".join(
        json.dumps(asdict(receipt), sort_keys=True) + "\n"
        for receipt in receipts
    ).encode("utf-8")
    if receipt_path.exists():
        if receipt_path.read_bytes() != receipt_payload:
            raise PolicyKLSourceError("imported artifact receipt changed")
    else:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(receipt_payload)
    return tuple(receipts)

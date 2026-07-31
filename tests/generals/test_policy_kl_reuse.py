from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agentbench_frame.generals.measurement_state import measurement_state_id
from agentbench_frame.generals.models import (
    HistoricalPolicyConfig,
    PolicyKLExtensionConfig,
)
from agentbench_frame.generals.policy_kl_math import (
    compute_controlled_policy_kl,
)
from agentbench_frame.generals.policy_kl_reuse import (
    PolicyKLSourceError,
    canonical_tree_hash,
    materialize_policy_kl_source,
    verify_policy_kl_source,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


ACTION_SPEC_ID = (
    "a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e"
)
EPSILONS = ("0.001", "0.01", "0.05", "0.1")
SEEDS = (289101, 289202, 289303)
SEATS = (0, 1)
DECISIONS = (2, 10)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _test_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink()
    ):
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{file_hash}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def event(event_type: str, index: int, payload: dict) -> dict:
    record = {
        **payload,
        "schema_version": "1.0",
        "event_id": f"evt_source_{index:04d}",
        "event_type": event_type,
        "event": event_type,
        "created_at": "2026-07-30T00:00:00+00:00",
        "timestamp": float(index),
    }
    record["run_id"] = payload.get("run_id", "source-run")
    return record


def build_complete_source(tmp_path: Path):
    root = tmp_path / "source-run"
    root.mkdir()
    snapshotter = LocalWorkspaceSnapshotter()
    history = []
    state_records = []
    reference_events = []
    count_events = []
    action_events = []
    counts = {}
    actions = {}

    for seed in SEEDS:
        for seat in SEATS:
            for decision in DECISIONS:
                snapshot = {
                    "schema": "generals-measurement-state-v1",
                    "actor": seat,
                    "state": {"seed": seed, "decision": decision},
                }
                state_id = measurement_state_id(snapshot)
                record = {
                    "seed": seed,
                    "seat": seat,
                    "decision_number": decision,
                    "state_id": state_id,
                    "snapshot": snapshot,
                }
                state_records.append(record)
                state_path = root / "reference/states" / f"{state_id}.json"
                write_json(state_path, record)
                reference_events.append(
                    {
                        "measurement_id": "generals-policy-kl-reference-v1",
                        "seed": seed,
                        "seat": seat,
                        "decision_number": decision,
                        "measurement_state_id": state_id,
                        "artifact_ref": str(state_path),
                    }
                )
                support = 13 + len(state_records)
                counts[state_id] = support
                count_record = {
                    "measurement_state_id": state_id,
                    "seed": seed,
                    "seat": seat,
                    "decision_number": decision,
                    "action_space_spec_id": ACTION_SPEC_ID,
                    "status": "complete",
                    "support_size": str(support),
                    "expanded_states": support,
                    "legal_edges": support - 1,
                    "cache_hits": 0,
                    "elapsed_time_s": 0.01,
                    "error": None,
                }
                write_json(
                    root / "action-space/counts" / f"{state_id}.json",
                    count_record,
                )
                count_events.append(count_record)

    for version_index in range(7):
        version = f"v{version_index}"
        run_id = f"run-{version_index}"
        source = root / "policies" / version / "source"
        source.mkdir(parents=True)
        (source / "main.py").write_text(
            f"VERSION = {version_index}\n",
            encoding="utf-8",
        )
        manifest = snapshotter.capture(source)
        snapshotter.write_manifest(
            manifest,
            root / "policies" / version / "manifest.json",
        )
        history.append(
            HistoricalPolicyConfig(
                version=version,
                run_id=run_id,
                content_hash=manifest.content_hash,
            )
        )
        for state_index, state_record in enumerate(state_records):
            state_id = state_record["state_id"]
            canonical = ((5, 1 + (version_index + state_index) % 2), (8,))
            actions[(version, state_id)] = canonical
            action_record = {
                "version": version,
                "run_id": run_id,
                "content_hash": manifest.content_hash,
                "measurement_state_id": state_id,
                "status": "complete",
                "deterministic": True,
                "raw_actions": [
                    [list(command) for command in canonical],
                    [list(command) for command in canonical],
                ],
                "canonical_action": [
                    list(command) for command in canonical
                ],
                "elapsed_time_s": 0.01,
                "stdout": "{}\n{}\n",
                "stderr": "",
                "error": None,
                "action_space_spec_id": ACTION_SPEC_ID,
            }
            write_json(
                root / "policies" / version / f"{state_id}.json",
                action_record,
            )
            action_events.append(action_record)

    computation = compute_controlled_policy_kl(
        versions=tuple(item.version for item in history),
        reference_state_ids=tuple(
            item["state_id"] for item in state_records
        ),
        counts=counts,
        actions=actions,
        missing_actions={},
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        action_space_spec_id=ACTION_SPEC_ID,
    )
    write_json(
        root / "benchmark/action-space-spec.json",
        {
            "schema": "generals-macro-action-space-v1",
            "engine_hash": "e" * 64,
            "measurement_state_schema": "generals-measurement-state-v1",
            "canonicalization_version": "generals-canonical-macro-v1",
            "spec_id": ACTION_SPEC_ID,
        },
    )
    write_json(
        root / "benchmark/reference-state-spec.json",
        {
            "measurement_id": "generals-policy-kl-reference-v1",
            "opponent_id": "advanced-rank02-robinliu-v18",
            "seeds": list(SEEDS),
            "seats": list(SEATS),
            "decision_numbers": list(DECISIONS),
            "state_ids": [item["state_id"] for item in state_records],
        },
    )
    measurement = root / "measurement"
    measurement.mkdir()
    (measurement / "per-state-kl.jsonl").write_text(
        "".join(
            json.dumps(fact, sort_keys=True) + "\n"
            for fact in computation.facts
        ),
        encoding="utf-8",
    )
    write_json(
        measurement / "transition-summary.json",
        computation.metric,
    )

    raw_events = []
    for event_type, payloads in (
        ("reference_state_selected", reference_events),
        ("action_space_count", count_events),
        ("historical_policy_action", action_events),
        ("controlled_reference_policy_kl", computation.facts),
    ):
        for payload in payloads:
            raw_events.append(event(event_type, len(raw_events), payload))
    assert len(raw_events) == 396
    (root / "events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in raw_events),
        encoding="utf-8",
    )
    quality = {
        "total_lines": 396,
        "valid_events": 396,
        "malformed_lines": 0,
        "invalid_events": 0,
        "unknown_event_types": 0,
        "duplicate_event_ids": 0,
        "missing_event_ids": 0,
        "missing_run_ids": 0,
        "warnings": [],
    }
    write_json(root / "quality.json", quality)
    write_json(
        root / "summary.json",
        {
            "run_id": "source-run",
            "status": "complete",
            "measurement_id": "generals-policy-kl-reference-v1",
            "event_quality": quality,
            "controlled_reference_policy_kl": computation.metric,
        },
    )
    (root / "run.toml").write_text('run_id = "source-run"\n')
    (root / "action-space/cache.sqlite").write_bytes(b"not imported")
    (root / "action-space/official-probes.jsonl").write_text("{}\n")
    prepared = root / "prepared-opponents/opponent"
    prepared.mkdir(parents=True)
    (prepared / "binary").write_bytes(b"not imported")

    history.append(
        HistoricalPolicyConfig(
            version="v7",
            run_id="run-7",
            content_hash="7" * 64,
        )
    )
    config = PolicyKLExtensionConfig(
        measurement_id="generals-policy-kl-reference-v2",
        source_measurement_id="generals-policy-kl-reference-v1",
        source_run_id="source-run",
        source_tree_hash=_test_tree_hash(root),
        opponent_id="advanced-rank02-robinliu-v18",
        seeds=SEEDS,
        seats=SEATS,
        decision_numbers=DECISIONS,
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        history=tuple(history),
    )
    return root, config


def current_hash(config, root):
    return replace(config, source_tree_hash=_test_tree_hash(root))


def test_canonical_tree_hash_matches_the_frozen_stream_format(tmp_path):
    root = tmp_path / "tree"
    (root / "b").mkdir(parents=True)
    (root / "a.txt").write_text("one", encoding="utf-8")
    (root / "b/z.txt").write_text("two", encoding="utf-8")

    assert canonical_tree_hash(root) == (
        "3b28391b2a9e4640b4626a99d83ff334abfe31450cd9b116744f03d54b9e3060"
    )


def test_verify_accepts_one_complete_source_run(tmp_path):
    root, config = build_complete_source(tmp_path)

    verified = verify_policy_kl_source(root, config)

    assert verified.tree_hash == config.source_tree_hash
    assert len(verified.reference_records) == 12
    assert len(verified.counts) == 12
    assert len(verified.actions) == 84
    assert len(verified.prior_facts) == 288
    assert len(verified.reuse_events) == 108
    assert len(verified.kl_source_events) == 288
    assert len(verified.prior_metric["transitions"]) == 6
    action_event = next(
        item
        for item in verified.reuse_events
        if item["event_type"] == "historical_policy_action"
    )
    assert action_event["run_id"] == "run-0"


def test_verify_rejects_a_wrong_frozen_tree_hash(tmp_path):
    root, config = build_complete_source(tmp_path)

    with pytest.raises(PolicyKLSourceError, match="tree hash"):
        verify_policy_kl_source(
            root,
            replace(config, source_tree_hash="0" * 64),
        )


def test_verify_rejects_a_mutated_count_after_tree_reapproval(tmp_path):
    root, config = build_complete_source(tmp_path)
    count_path = next((root / "action-space/counts").glob("*.json"))
    count = json.loads(count_path.read_text())
    count["status"] = "broken"
    write_json(count_path, count)

    with pytest.raises(PolicyKLSourceError, match="count"):
        verify_policy_kl_source(root, current_hash(config, root))


def test_verify_rejects_dirty_source_event_quality(tmp_path):
    root, config = build_complete_source(tmp_path)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["event_quality"]["unknown_event_types"] = 1
    write_json(summary_path, summary)

    with pytest.raises(PolicyKLSourceError, match="event quality"):
        verify_policy_kl_source(root, current_hash(config, root))


def test_verify_rejects_a_missing_per_state_coordinate(tmp_path):
    root, config = build_complete_source(tmp_path)
    facts_path = root / "measurement/per-state-kl.jsonl"
    facts = facts_path.read_text().splitlines()
    facts_path.write_text("\n".join(facts[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(PolicyKLSourceError, match="288"):
        verify_policy_kl_source(root, current_hash(config, root))


def test_materialization_is_idempotent_and_leaves_source_unchanged(tmp_path):
    root, config = build_complete_source(tmp_path)
    before = canonical_tree_hash(root)
    verified = verify_policy_kl_source(root, config)
    destination = tmp_path / "new-run"

    first = materialize_policy_kl_source(verified, destination)
    second = materialize_policy_kl_source(verified, destination)

    assert second == first
    assert canonical_tree_hash(root) == before
    assert (destination / "benchmark/action-space-spec.json").is_file()
    assert len(list((destination / "reference/states").glob("*.json"))) == 12
    assert len(list((destination / "action-space/counts").glob("*.json"))) == 12
    assert len(list((destination / "policies").glob("v[0-6]"))) == 7
    assert (destination / "provenance/source-run/events.jsonl").is_file()
    assert (destination / "provenance/source-run/per-state-kl.jsonl").is_file()
    assert not (destination / "action-space/cache.sqlite").exists()
    assert not (destination / "prepared-opponents").exists()
    assert len(first) > 100
    receipt_lines = (
        destination / "provenance/imported-artifacts.jsonl"
    ).read_text().splitlines()
    assert len(receipt_lines) == len(first)


def test_real_completed_v1_tree_hash_matches_the_frozen_contract():
    root = Path(
        "agentbench_data/runs/28_generals/generals-policy-kl/"
        "20260730_1126_8d123b55"
    )
    if not root.is_dir():
        pytest.skip("real controlled policy-KL run is unavailable")

    assert canonical_tree_hash(root) == (
        "6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225"
    )

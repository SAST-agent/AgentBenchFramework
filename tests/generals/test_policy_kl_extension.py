import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentbench_frame.generals.historical_policy import PolicyProbeResult
from agentbench_frame.generals.policy_kl_extension import (
    GeneralsPolicyKLExtensionPipeline,
)
from tests.generals.test_policy_kl_reuse import (
    ACTION_SPEC_ID,
    build_complete_source,
)


class FakeActionSpace:
    spec_id = ACTION_SPEC_ID

    def spec_payload(self):
        return {
            "schema": "generals-macro-action-space-v1",
            "engine_hash": "e" * 64,
            "measurement_state_schema": "generals-measurement-state-v1",
            "canonicalization_version": "generals-canonical-macro-v1",
            "spec_id": ACTION_SPEC_ID,
        }

    def canonicalize(self, state, action):
        return tuple(tuple(command) for command in action)


class FakeExtensionPipeline(GeneralsPolicyKLExtensionPipeline):
    probe_mode = "complete"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probes = []

    def _new_action_space(self, run_dir):
        return FakeActionSpace()

    def _resolve_v7(self, run_dir):
        policy = self.reference.history[-1]
        source = Path(run_dir) / "policies/v7/source"
        source.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            version=policy.version,
            run_id=policy.run_id,
            content_hash=policy.content_hash,
            source=source,
        )

    def _probe_policy(self, policy, reference):
        assert policy.version == "v7"
        self.probes.append((policy.version, reference.state_id))
        if self.probe_mode == "raise":
            self.probe_mode = "complete"
            raise RuntimeError("simulated interruption before v7 actions")
        old = ((5, 1), (8,))
        new = ((5, 2), (8,))
        if self.probe_mode == "nondeterministic" and len(self.probes) == 1:
            return PolicyProbeResult(
                version="v7",
                status="nondeterministic",
                deterministic=False,
                raw_actions=(old, new),
                elapsed_time_s=0.01,
                stdout="",
                stderr="",
                error="policy emitted different actions across repeats",
            )
        return PolicyProbeResult(
            version="v7",
            status="complete",
            deterministic=True,
            raw_actions=(new, new),
            elapsed_time_s=0.01,
            stdout="{}\n{}\n",
            stderr="",
        )


def make_pipeline(tmp_path, *, probe_mode="complete"):
    source, reference = build_complete_source(tmp_path)
    pipeline = FakeExtensionPipeline(
        config=SimpleNamespace(benchmark_id="generals-hl-pilot-v1"),
        reference=reference,
        assets=SimpleNamespace(
            engine_root=tmp_path,
            engine_hash="e" * 64,
            root=tmp_path,
        ),
        data_dir=tmp_path / "data",
        source_run_dir=source,
    )
    pipeline.probe_mode = probe_mode
    return pipeline, source


def read_events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_extension_probes_only_v7_and_emits_complete_projection(tmp_path):
    pipeline, source = make_pipeline(tmp_path)

    result = pipeline.run()

    metric = result.summary["controlled_reference_policy_kl"]
    state_ids = json.loads(
        (source / "benchmark/reference-state-spec.json").read_text()
    )["state_ids"]
    assert result.status == "complete"
    assert pipeline.probes == [("v7", state_id) for state_id in state_ids]
    assert len(metric["transitions"]) == 7
    assert metric["transitions"][:6] == json.loads(
        (source / "measurement/transition-summary.json").read_text()
    )["transitions"]
    assert metric["transitions"][-1]["version_before"] == "v6"
    assert metric["transitions"][-1]["version_after"] == "v7"
    assert all(
        item["coverage"] == {"complete": 12, "total": 12}
        for item in metric["transitions"]
    )
    assert len(
        (result.run_dir / "measurement/per-state-kl.jsonl")
        .read_text()
        .splitlines()
    ) == 336
    assert result.summary["source_run_id"] == "source-run"
    assert result.summary["source_tree_hash_before"] == (
        result.summary["source_tree_hash_after"]
    )

    events = read_events(result.run_dir)
    reused = [
        item for item in events if item.get("reuse_status") == "verified_reuse"
    ]
    new = [item for item in events if item.get("reuse_status") == "new"]
    assert len(reused) == 396
    assert len(new) == 60
    assert all(item["source_event_id"] for item in reused)
    assert all(item["source_run_id"] == "source-run" for item in reused)
    assert sum(
        item["event_type"] == "historical_policy_action"
        and item.get("version") == "v7"
        for item in new
    ) == 12
    assert sum(
        item["event_type"] == "controlled_reference_policy_kl"
        and item.get("version_before") == "v6"
        and item.get("version_after") == "v7"
        for item in new
    ) == 48
    assert result.summary["event_quality"]["unknown_event_types"] == 0
    assert result.summary["event_quality"]["duplicate_event_ids"] == 0


def test_one_nondeterministic_v7_state_keeps_new_aggregate_null(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, probe_mode="nondeterministic")

    result = pipeline.run()

    transition = result.summary["controlled_reference_policy_kl"][
        "transitions"
    ][-1]
    assert result.status == "incomplete_policy_measurement"
    assert transition["mean_kl_nats"] is None
    assert transition["coverage"] == {"complete": 11, "total": 12}
    rows = [
        json.loads(line)
        for line in (result.run_dir / "measurement/per-state-kl.jsonl")
        .read_text()
        .splitlines()
    ]
    missing = [
        item
        for item in rows
        if item["version_before"] == "v6" and item["status"] == "missing"
    ]
    assert len(missing) == 4
    assert all("different actions" in item["missing_reasons"][0] for item in missing)


def test_recovery_appends_without_duplicate_import_events(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, probe_mode="raise")
    failed = pipeline.run()
    before = (failed.run_dir / "events.jsonl").read_bytes()

    recovered = pipeline.recover(failed.run_dir)
    after = (failed.run_dir / "events.jsonl").read_bytes()

    assert failed.status == "failed"
    assert recovered.status == "complete"
    assert recovered.run_dir == failed.run_dir
    assert after.startswith(before)
    events = read_events(failed.run_dir)
    reused = [
        item for item in events if item.get("reuse_status") == "verified_reuse"
    ]
    assert len(reused) == 396
    assert len({item["event_id"] for item in events}) == len(events)
    assert recovered.summary["event_quality"]["duplicate_event_ids"] == 0


def test_recovery_rejects_a_complete_extension(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    complete = pipeline.run()

    with pytest.raises(ValueError, match="complete"):
        pipeline.recover(complete.run_dir)

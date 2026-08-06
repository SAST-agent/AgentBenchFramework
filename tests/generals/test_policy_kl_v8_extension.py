import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentbench_frame.generals.historical_policy import PolicyProbeResult
from agentbench_frame.generals.models import HistoricalPolicyConfig
from agentbench_frame.generals.policy_kl_reuse import canonical_tree_hash
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter
from agentbench_frame.generals.policy_kl_v8_extension import (
    GeneralsPolicyKLV8ExtensionPipeline,
)
from tests.generals.test_policy_kl_extension import (
    FakeActionSpace,
    make_pipeline as make_v7_pipeline,
)


class FakeV8Pipeline(GeneralsPolicyKLV8ExtensionPipeline):
    probe_mode = "complete"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probes = []

    def _new_action_space(self, run_dir):
        return FakeActionSpace()

    def _resolve_v8(self, run_dir):
        source = Path(run_dir) / "policies/v8/source"
        source.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            version="v8",
            run_id=self.target_run_dir.name,
            content_hash=self.expected_target_hash,
            source=source,
        )

    def _probe_policy(self, policy, reference):
        self.probes.append((policy.version, reference.state_id))
        action = ((5, 3), (8,))
        if self.probe_mode == "nondeterministic" and len(self.probes) == 1:
            return PolicyProbeResult(
                version="v8", status="nondeterministic",
                deterministic=False,
                raw_actions=(action, ((5, 4), (8,))),
                elapsed_time_s=0.01, stdout="", stderr="",
                error="policy emitted different actions across repeats",
            )
        return PolicyProbeResult(
            version="v8", status="complete", deterministic=True,
            raw_actions=(action, action), elapsed_time_s=0.01,
            stdout="{}\n{}\n", stderr="",
        )


def make_pipeline(tmp_path, *, probe_mode="complete"):
    v7_pipeline, _ = make_v7_pipeline(tmp_path)
    v7_result = v7_pipeline.run()
    v7_root = v7_result.run_dir / "policies/v7"
    v7_source = v7_root / "source"
    (v7_source / "strategy.py").write_text("# v7\n", encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    v7_manifest = snapshotter.capture(v7_source)
    snapshotter.write_manifest(v7_manifest, v7_root / "manifest.json")
    for path in v7_root.glob("*.json"):
        if path.name == "manifest.json":
            continue
        record = json.loads(path.read_text())
        record["content_hash"] = v7_manifest.content_hash
        path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    events = []
    for line in (v7_result.run_dir / "events.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") == "historical_policy_action" and record.get("version") == "v7":
            record["content_hash"] = v7_manifest.content_hash
        events.append(record)
    (v7_result.run_dir / "events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events)
    )
    target = tmp_path / "target-v8"
    source = target / "versions/v8/source"
    source.mkdir(parents=True)
    (source / "strategy.py").write_text("# v8\n", encoding="utf-8")
    target_hash = "8" * 64
    (target / "versions/v8/manifest.json").write_text(json.dumps({
        "version": "v8", "content_hash": target_hash,
        "files": [{"path": "strategy.py"}],
    }), encoding="utf-8")
    reference = replace(
        v7_pipeline.reference,
        measurement_id="generals-policy-kl-reference-v3",
        source_measurement_id=v7_pipeline.reference.measurement_id,
        source_run_id=v7_result.run_dir.name,
        source_tree_hash=canonical_tree_hash(v7_result.run_dir),
        history=v7_pipeline.reference.history[:-1] + (
            replace(
                v7_pipeline.reference.history[-1],
                content_hash=v7_manifest.content_hash,
            ),
            HistoricalPolicyConfig("v8", target.name, target_hash),
        ),
    )
    pipeline = FakeV8Pipeline(
        config=SimpleNamespace(benchmark_id="generals-hl-pilot-v1"),
        reference=reference,
        assets=SimpleNamespace(
            engine_root=tmp_path, engine_hash="e" * 64, root=tmp_path,
        ),
        data_dir=tmp_path / "v8-data",
        source_run_dir=v7_result.run_dir,
        target_run_dir=target,
        expected_target_hash=target_hash,
    )
    pipeline.probe_mode = probe_mode
    return pipeline, v7_result


def test_v8_extension_reuses_v0_v7_and_probes_only_v8(tmp_path):
    pipeline, source = make_pipeline(tmp_path)
    result = pipeline.run()
    metric = result.summary["controlled_reference_policy_kl"]
    state_ids = json.loads(
        (source.run_dir / "benchmark/reference-state-spec.json").read_text()
    )["state_ids"]
    assert result.status == "complete"
    assert pipeline.probes == [("v8", state_id) for state_id in state_ids]
    assert len(metric["transitions"]) == 8
    assert metric["transitions"][:7] == source.summary[
        "controlled_reference_policy_kl"
    ]["transitions"]
    assert metric["transitions"][-1]["version_after"] == "v8"
    assert len((result.run_dir / "measurement/per-state-kl.jsonl").read_text().splitlines()) == 384


def test_v8_extension_marks_nondeterministic_probe_incomplete(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, probe_mode="nondeterministic")
    result = pipeline.run()
    assert result.status == "incomplete_policy_measurement"
    assert result.summary["controlled_reference_policy_kl"]["transitions"][-1][
        "coverage"
    ] == {"complete": 11, "total": 12}


def test_v8_extension_rejects_target_hash_mismatch(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    pipeline.expected_target_hash = "9" * 64
    with pytest.raises(ValueError, match="target v8"):
        GeneralsPolicyKLV8ExtensionPipeline._resolve_v8(
            pipeline, Path(tmp_path / "out")
        )

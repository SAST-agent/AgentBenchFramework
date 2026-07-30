import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentbench_frame.generals.historical_policy import PolicyProbeResult
from agentbench_frame.generals.macro_counter import (
    ExactCountIncomplete,
    ExactCountResult,
)
from agentbench_frame.generals.models import (
    HistoricalPolicyConfig,
    PolicyKLReferenceConfig,
)


def reference_config():
    return PolicyKLReferenceConfig(
        measurement_id="generals-policy-kl-reference-v1",
        opponent_id="advanced-rank02-robinliu-v18",
        seeds=(289101, 289202, 289303),
        seats=(0, 1),
        decision_numbers=(2, 10),
        epsilons=("0.001", "0.01", "0.05", "0.1"),
        primary_epsilon="0.01",
        history=tuple(
            HistoricalPolicyConfig(
                version=f"v{index}",
                run_id=f"run-{index}",
                content_hash=str(index) * 64,
            )
            for index in range(7)
        ),
    )


class FakeActionSpace:
    spec_id = "fake-action-space-v1"

    def spec_payload(self):
        return {
            "schema": "fake",
            "engine_hash": "e" * 64,
            "measurement_state_schema": "fake-state-v1",
            "canonicalization_version": "fake-v1",
            "spec_id": self.spec_id,
        }

    def canonicalize(self, state, action):
        if action == ((99,), (8,)):
            raise ValueError("illegal")
        return tuple(tuple(command) for command in action)


class FakePipelineMixin:
    fail_count_once = False
    missing_probe = None

    def _collect_reference_states(self, run, run_dir):
        from agentbench_frame.generals.policy_kl_pipeline import ReferenceState

        records = []
        for seed in self.reference.seeds:
            for seat in self.reference.seats:
                for decision in self.reference.decision_numbers:
                    snapshot = {
                        "schema": "fake-state-v1",
                        "actor": seat,
                        "state": {
                            "round": decision,
                            "seed": seed,
                        },
                    }
                    records.append(
                        ReferenceState.from_snapshot(
                            seed=seed,
                            seat=seat,
                            decision_number=decision,
                            snapshot=snapshot,
                        )
                    )
        return tuple(records)

    def _new_action_space(self, run_dir):
        return FakeActionSpace()

    def _count_reference_state(self, action_space, reference, cache_path):
        if self.fail_count_once:
            self.fail_count_once = False
            raise ExactCountIncomplete(
                ExactCountResult(
                    status="incomplete_max_expanded_states",
                    support_size=None,
                    expanded_states=2,
                    legal_edges=3,
                    cache_hits=0,
                    elapsed_time_s=0.01,
                    root_state_id=reference.state_id,
                    error="guard",
                )
            )
        support = 100 + reference.decision_number + reference.seat
        return ExactCountResult(
            status="complete",
            support_size=support,
            expanded_states=4,
            legal_edges=8,
            cache_hits=1,
            elapsed_time_s=0.01,
            root_state_id=reference.state_id,
        )

    def _resolve_policies(self, run_dir):
        return tuple(
            SimpleNamespace(
                version=item.version,
                run_id=item.run_id,
                content_hash=item.content_hash,
                source=Path(run_dir) / "policies" / item.version / "source",
            )
            for item in self.reference.history
        )

    def _probe_policy(self, policy, reference):
        if self.missing_probe == (policy.version, reference.state_id):
            return PolicyProbeResult(
                version=policy.version,
                status="worker_error",
                deterministic=False,
                raw_actions=(),
                elapsed_time_s=0.01,
                stdout="",
                stderr="broken",
                error="broken",
            )
        index = int(policy.version[1:])
        action = ((5, 1 + index % 2), (8,))
        return PolicyProbeResult(
            version=policy.version,
            status="complete",
            deterministic=True,
            raw_actions=(action, action),
            elapsed_time_s=0.01,
            stdout="",
            stderr="",
        )


def make_pipeline(tmp_path):
    from agentbench_frame.generals.policy_kl_pipeline import (
        GeneralsPolicyKLPipeline,
    )

    class FakePipeline(FakePipelineMixin, GeneralsPolicyKLPipeline):
        pass

    return FakePipeline(
        config=SimpleNamespace(benchmark_id="generals-hl-pilot-v1"),
        reference=reference_config(),
        assets=SimpleNamespace(
            engine_root=tmp_path,
            engine_hash="e" * 64,
            root=tmp_path,
        ),
        data_dir=tmp_path,
        count_wall_time_s=1,
        count_max_states=100,
    )


def test_pipeline_emits_complete_six_transition_measurement(tmp_path):
    pipeline = make_pipeline(tmp_path)

    result = pipeline.run()

    metric = result.summary["controlled_reference_policy_kl"]
    assert result.status == "complete"
    assert len(metric["transitions"]) == 6
    assert all(
        point["coverage"] == {"complete": 12, "total": 12}
        for point in metric["transitions"]
    )
    assert metric["primary_epsilon"] == "0.01"
    assert len(
        (result.run_dir / "measurement/per-state-kl.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 288

    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(
        event["event_type"] == "reference_state_selected"
        for event in events
    ) == 12
    assert sum(
        event["event_type"] == "historical_policy_action"
        for event in events
    ) == 84
    assert sum(
        event["event_type"] == "controlled_reference_policy_kl"
        for event in events
    ) == 288
    assert result.summary["event_quality"]["unknown_event_types"] == 0


def test_one_missing_policy_action_keeps_affected_aggregates_null(tmp_path):
    pipeline = make_pipeline(tmp_path)
    references = pipeline._collect_reference_states(None, tmp_path)
    pipeline.missing_probe = ("v3", references[0].state_id)

    result = pipeline.run()

    transitions = result.summary["controlled_reference_policy_kl"][
        "transitions"
    ]
    affected = {
        (item["version_before"], item["version_after"]): item
        for item in transitions
        if "v3" in (item["version_before"], item["version_after"])
    }
    assert result.status == "incomplete_policy_measurement"
    assert set(affected) == {("v2", "v3"), ("v3", "v4")}
    assert all(item["mean_kl_nats"] is None for item in affected.values())
    assert all(
        item["coverage"] == {"complete": 11, "total": 12}
        for item in affected.values()
    )


def test_incomplete_count_is_resumable_in_same_append_only_run(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.fail_count_once = True

    failed = pipeline.run()
    before = (failed.run_dir / "events.jsonl").read_bytes()
    recovered = pipeline.recover(failed.run_dir)
    after = (failed.run_dir / "events.jsonl").read_bytes()

    assert failed.status == "incomplete_exact_count"
    assert recovered.run_dir == failed.run_dir
    assert recovered.status == "complete"
    assert after.startswith(before)
    assert len(after) > len(before)
    events = [
        json.loads(line)
        for line in after.decode("utf-8").splitlines()
    ]
    assert any(
        event["event_type"] == "pipeline_resumed"
        for event in events
    )


def test_recovery_rejects_a_complete_measurement(tmp_path):
    pipeline = make_pipeline(tmp_path)
    complete = pipeline.run()

    with pytest.raises(ValueError, match="complete"):
        pipeline.recover(complete.run_dir)

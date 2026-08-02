import json
from pathlib import Path

from agentbench_frame.games.rollman.research_measurement import (
    RollmanMeasurementRunner,
)
from agentbench_frame.hl.codebase import VersionStore
from agentbench_frame.hl.evaluator import CandidateEvaluation


def _trace(path, states):
    with path.open("w", encoding="utf-8") as handle:
        for index, state in enumerate(states):
            handle.write(
                json.dumps(
                    {
                        "type": "action",
                        "player": 0,
                        "decision": {
                            "state": state,
                            "state_id": f"rollout-{index}",
                            "action": 0,
                            "memory_id": None,
                        },
                    }
                )
                + "\n"
            )


def test_measurement_uses_frozen_origin_contexts_and_separate_occupancy(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("OLD\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    old = store.snapshot(parent_version_id=None, act_id="origin")
    (workspace / "ai.py").write_text("NEW\n", encoding="utf-8")
    new = store.snapshot(parent_version_id=old.version_id, act_id="act-1")
    states = [{"level": 1, "round": 0}, {"level": 1, "round": 1}]
    origin_trace = tmp_path / "origin.trace.jsonl"
    new_trace = tmp_path / "new.trace.jsonl"
    _trace(origin_trace, states)
    _trace(new_trace, list(reversed(states)))
    old_eval = CandidateEvaluation(
        status="complete",
        score=0.5,
        matches=({"status": "complete", "trace": str(origin_trace)},),
    )
    new_eval = CandidateEvaluation(
        status="complete",
        score=0.5,
        matches=({"status": "complete", "trace": str(new_trace)},),
    )

    def fake_probe(*, workspace, states, **kwargs):
        is_new = (Path(workspace) / "ai.py").read_text(encoding="utf-8") == "NEW\n"
        return tuple(
            {
                "reference_index": index,
                "state_id": f"visible-{index}",
                "action": 4 if is_new and index == 1 else 0,
                "memory_id": None,
            }
            for index, _ in enumerate(states)
        )

    runner = RollmanMeasurementRunner(
        root=tmp_path / "measurement",
        sdk_root=tmp_path,
        epsilon=0.05,
        probe_runner=fake_probe,
    )
    runner.freeze_reference(old_eval)
    result = runner.measure(
        new_version=new,
        old_version=old,
        version_store=store,
        new_evaluation=new_eval,
        old_evaluation=old_eval,
    )

    assert len(result["local_policy_kl_trace"]) == 2
    assert result["local_policy_kl_trace"][0] == 0
    assert result["local_policy_kl_trace"][1] > 0
    assert result["occupancy_shift"] == 0


def test_activation_measurement_counts_exact_action_changes_on_parent_traces(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("OLD\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    old = store.snapshot(parent_version_id=None, act_id="origin")
    (workspace / "ai.py").write_text("NEW\n", encoding="utf-8")
    new = store.snapshot(parent_version_id=old.version_id, act_id="candidate")
    states = [{"level": 1, "round": 0}, {"level": 1, "round": 1}]
    parent_trace = tmp_path / "parent.trace.jsonl"
    _trace(parent_trace, states)
    parent_evaluation = CandidateEvaluation(
        status="complete",
        score=0.0,
        matches=(
            {
                "status": "complete",
                "opponent": "rank15",
                "seed": 101,
                "trace": str(parent_trace),
            },
        ),
    )

    def fake_probe(*, workspace, states, **kwargs):
        is_new = (Path(workspace) / "ai.py").read_text(encoding="utf-8") == "NEW\n"
        return tuple(
            {
                "reference_index": index,
                "state_id": f"visible-{index}",
                "action": 4 if is_new and index == 1 else 0,
                "memory_id": None,
            }
            for index, _ in enumerate(states)
        )

    runner = RollmanMeasurementRunner(
        root=tmp_path / "measurement",
        sdk_root=tmp_path,
        epsilon=0.05,
        probe_runner=fake_probe,
    )

    result = runner.measure_activation(
        new_version=new,
        old_version=old,
        version_store=store,
        parent_evaluation=parent_evaluation,
    )

    assert result["status"] == "complete"
    assert result["decision_count"] == 2
    assert result["changed_action_count"] == 1
    assert result["changed_fraction"] == 0.5
    assert result["episodes"][0]["changed_reference_indices"] == [1]


def test_activation_measurement_only_uses_requested_quick_screen_seeds(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("OLD\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    old = store.snapshot(parent_version_id=None, act_id="origin")
    (workspace / "ai.py").write_text("NEW\n", encoding="utf-8")
    new = store.snapshot(parent_version_id=old.version_id, act_id="candidate")
    seed101 = tmp_path / "seed101.trace.jsonl"
    seed102 = tmp_path / "seed102.trace.jsonl"
    _trace(seed101, [{"level": 1, "round": 0}])
    _trace(seed102, [{"level": 1, "round": 1}])
    parent_evaluation = CandidateEvaluation(
        status="complete",
        score=0.0,
        matches=(
            {"status": "complete", "seed": 101, "trace": str(seed101)},
            {"status": "complete", "seed": 102, "trace": str(seed102)},
        ),
    )

    def fake_probe(*, workspace, states, **kwargs):
        is_new = (Path(workspace) / "ai.py").read_text(encoding="utf-8") == "NEW\n"
        return (
            {
                "reference_index": 0,
                "state_id": "visible",
                "action": 4 if is_new and states[0]["round"] == 1 else 0,
                "memory_id": None,
            },
        )

    runner = RollmanMeasurementRunner(
        root=tmp_path / "measurement",
        sdk_root=tmp_path,
        epsilon=0.05,
        probe_runner=fake_probe,
    )
    result = runner.measure_activation(
        new_version=new,
        old_version=old,
        version_store=store,
        parent_evaluation=parent_evaluation,
        seeds=(101,),
    )

    assert result["decision_count"] == 1
    assert result["changed_action_count"] == 0
    assert [episode["seed"] for episode in result["episodes"]] == [101]


def test_activation_measurement_reuses_cached_parent_probe(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("OLD\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    old = store.snapshot(parent_version_id=None, act_id="origin")
    (workspace / "ai.py").write_text("NEW-ONE\n", encoding="utf-8")
    new_one = store.snapshot(parent_version_id=old.version_id, act_id="one")
    (workspace / "ai.py").write_text("NEW-TWO\n", encoding="utf-8")
    new_two = store.snapshot(parent_version_id=old.version_id, act_id="two")
    trace = tmp_path / "parent.trace.jsonl"
    _trace(trace, [{"level": 1, "round": 0}])
    parent_evaluation = CandidateEvaluation(
        status="complete",
        score=0.0,
        matches=({"status": "complete", "seed": 101, "trace": str(trace)},),
    )
    calls = []

    def fake_probe(*, workspace, states, artifact_path, **kwargs):
        label = (Path(workspace) / "ai.py").read_text(encoding="utf-8").strip()
        calls.append(label)
        result = (
            {
                "reference_index": 0,
                "state_id": "visible",
                "action": 0 if label == "OLD" else 4,
                "memory_id": None,
            },
        )
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        Path(artifact_path).write_text(json.dumps(list(result)), encoding="utf-8")
        return result

    runner = RollmanMeasurementRunner(
        root=tmp_path / "measurement",
        sdk_root=tmp_path,
        epsilon=0.05,
        probe_runner=fake_probe,
    )
    for candidate in (new_one, new_two):
        runner.measure_activation(
            new_version=candidate,
            old_version=old,
            version_store=store,
            parent_evaluation=parent_evaluation,
            seeds=(101,),
        )

    assert calls.count("OLD") == 1
    assert calls.count("NEW-ONE") == 1
    assert calls.count("NEW-TWO") == 1

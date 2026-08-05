import json

import pytest

from agentbench_frame.doto.lifecycle import (
    begin_iteration,
    close_iteration,
    init_run,
    finalize_run,
    record_build,
    record_evaluation,
    record_ig,
)
from agentbench_frame.doto.population import BuiltPolicy, PopulationBundle
from agentbench_frame.doto.run_store import RunState
from agentbench_frame.doto.run_store import InvalidTransition
from tests.doto.test_run_store import make_inputs


def _complete_matrix(score=1.0, completion_rate=1.0):
    task_ids = [f"train:human-{index:02d}:seat{seat}" for index in range(15) for seat in (0, 1)]
    return {
        "schema_version": 1,
        "split": "train",
        "seed": 11,
        "seats": [0, 1],
        "expected_task_ids": task_ids,
        "episodes": [],
        "summary": {"formal_score": score, "completion_rate": completion_rate},
    }


def _ready_baseline(tmp_path):
    source, train, test, skills = make_inputs(tmp_path)
    run = init_run(tmp_path / "DotoResults", "codex", source, train, test, skills, run_id="r1")
    executable = tmp_path / "main.out"
    executable.write_bytes(b"native")
    record_build(run, 0, {"status": "ready", "executable": executable})
    record_evaluation(run, 0, _complete_matrix())
    record_ig(run, 0, {"status": "computed", "value": 0.0, "reason": None})
    close_iteration(run, 0)
    return run, source


def test_closed_iteration_cannot_be_overwritten(tmp_path):
    run, _ = _ready_baseline(tmp_path)
    with pytest.raises(InvalidTransition, match="closed"):
        record_build(run, 0, {"status": "failed", "error": "late write"})


def test_complete_closed_iteration_can_parent_a_child(tmp_path):
    run, source = _ready_baseline(tmp_path)
    child = begin_iteration(run, 0, source, "try a different targeting rule")
    assert child["iteration"] == 1
    assert child["parent_iteration"] == 0
    assert json.loads((run.run_dir / "iterations/iteration-0001/iteration.json").read_text())["analysis"]


def test_incomplete_closed_iteration_cannot_be_parent(tmp_path):
    source, train, test, skills = make_inputs(tmp_path)
    run = init_run(tmp_path / "DotoResults", "codex", source, train, test, skills, run_id="r1")
    record_build(run, 0, {"status": "failed", "error": "compile"})
    close_iteration(run, 0)
    with pytest.raises(InvalidTransition, match="complete evaluated parent"):
        begin_iteration(run, 0, source, "must not branch from failure")


def test_close_aligns_parent_score_and_ig_versions(tmp_path):
    run, source = _ready_baseline(tmp_path)
    child = begin_iteration(run, 0, source, "measure an explicit child")
    executable = tmp_path / "child.out"
    executable.write_bytes(b"child-native")
    record_build(run, 1, {"status": "ready", "executable": executable})
    record_evaluation(run, 1, _complete_matrix(score=2.0))
    record_ig(run, 1, {"status": "computed", "value": 0.0, "reason": None})
    close_iteration(run, 1)
    score = json.loads((run.run_dir / "score_curve.json").read_text())["points"][1]
    ig = json.loads((run.run_dir / "ig_curve.json").read_text())["points"][1]
    assert score["parent_iteration"] == 0
    assert score["evo"] == 2.0
    assert ig["version"] == score["version"] == child["version"]


def test_29_cells_closes_with_null_score(tmp_path):
    run, source = _ready_baseline(tmp_path)
    begin_iteration(run, 0, source, "retain an incomplete formal evaluation")
    executable = tmp_path / "child.out"
    executable.write_bytes(b"child-native")
    record_build(run, 1, {"status": "ready", "executable": executable})
    record_evaluation(run, 1, _complete_matrix(score=None, completion_rate=29 / 30))
    record_ig(run, 1, {"status": "missing", "value": None,
                       "reason": "one formal trace is missing"})
    close_iteration(run, 1)
    point = json.loads((run.run_dir / "score_curve.json").read_text())["points"][1]
    assert point["evo"] is None
    assert point["completion_rate"] == 29 / 30


def _sealed_bundle(run):
    test_hash = json.loads((run.run_dir / "pools/test.json").read_text())["sha256"]
    policies = tuple(
        BuiltPolicy(f"hidden-{index:02d}", f"Hidden {index}",
                    run.run_dir / "fixture.out", f"{index:064x}")
        for index in range(28)
    )
    return PopulationBundle("fixture-v1", "test", 11, (0, 1), policies,
                            bundle_sha256=test_hash)


def _fake_final_evaluator(spec, candidate, scheduler, output_dir, **kwargs):
    return {
        "schema_version": 1,
        "benchmark_version": spec.bundle.benchmark_version,
        "split": "test",
        "seed": 11,
        "seats": [0, 1],
        "expected_task_ids": [task.task_id for task in spec.tasks],
        "episodes": [],
        "summary": {
            "expected_matches": 56,
            "attempted_matches": 56,
            "completion_rate": 1.0,
            "formal_score": 1.0,
            "win_rate": 0.7,
            "defeated_policy_count": 17,
            "policy_count": 28,
        },
        "scheduler": {"cpu_history": [70.0], "worker_history": [8],
                      "cancelled_task_ids": []},
    }


def test_finalize_runs_56_and_seals(tmp_path):
    run, _ = _ready_baseline(tmp_path)
    summary = finalize_run(run, 0, _sealed_bundle(run), object(),
                           evaluation_runner=_fake_final_evaluator)
    assert summary["test_policy_count"] == 28
    assert summary["attempted_test_matches"] == 56
    assert run.reload().state == RunState.FINALIZED


def test_interruption_cannot_change_selected_candidate(tmp_path):
    run, _ = _ready_baseline(tmp_path)

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        finalize_run(run, 0, _sealed_bundle(run), object(), evaluation_runner=interrupted)
    assert run.reload().state == RunState.FINAL_TEST_STARTED
    with pytest.raises(InvalidTransition, match="selected candidate"):
        finalize_run(run, 1, _sealed_bundle(run), object(), evaluation_runner=interrupted)

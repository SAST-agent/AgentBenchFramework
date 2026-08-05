from pathlib import Path

from agentbench_frame.doto.evaluation import (
    EpisodeResult,
    MatrixSpec,
    aggregate_matrix,
    evaluate_matrix,
)
from agentbench_frame.doto.population import BuiltPolicy, PopulationBundle
from agentbench_frame.doto.scheduler import AdaptiveScheduler, SchedulerConfig


def _bundle(split: str, count: int) -> PopulationBundle:
    policies = tuple(
        BuiltPolicy(f"human-{index:02d}", f"Human {index}", Path(f"/runtime/{index}/main.out"),
                    f"{index:064x}")
        for index in range(count)
    )
    return PopulationBundle("fixture-v1", split, 11, (0, 1), policies)


def test_matrix_sizes_and_seat_balance_are_complete():
    train = MatrixSpec.formal_train(_bundle("train", 15), seed=11)
    sealed = MatrixSpec.final_test(_bundle("test", 28), seed=11)
    assert len(train.tasks) == 30
    assert len(sealed.tasks) == 56
    assert {(task.seed, task.candidate_seat) for task in train.tasks} == {(11, 0), (11, 1)}
    assert all(sum(task.opponent_id == policy.policy_id for task in train.tasks) == 2
               for policy in train.bundle.policies)


def _episode(task_id: str, opponent_id: str, seat: int,
             diff: float | None, error: str | None = None) -> EpisodeResult:
    scores = None if diff is None else ((diff, 0.0) if seat == 0 else (0.0, diff))
    return EpisodeResult(
        task_id=task_id,
        opponent_id=opponent_id,
        opponent_name=opponent_id,
        seed=11,
        candidate_seat=seat,
        terminated_by="normal" if error is None else "ai_timeout",
        scores=scores,
        score_diff=diff,
        replay_path=None,
        trace_path=None,
        error=error,
    )


def test_incomplete_cell_nulls_formal_score_and_defeat_count():
    rows = [
        _episode("human:0", "human", 0, 2.0),
        _episode("human:1", "human", 1, None, "ai_timeout"),
    ]
    summary = aggregate_matrix(rows, expected_task_ids=("human:0", "human:1"))
    assert summary["formal_score"] is None
    assert summary["completion_rate"] == 0.5
    assert summary["defeated_policy_count"] == 0
    assert summary["policies"][0]["status"] == "incomplete"


def test_policy_requires_both_normal_seats_and_positive_mean():
    rows = [
        _episode("won:0", "won", 0, 4.0),
        _episode("won:1", "won", 1, -1.0),
        _episode("draw:0", "draw", 0, 1.0),
        _episode("draw:1", "draw", 1, -1.0),
    ]
    summary = aggregate_matrix(rows, expected_task_ids=tuple(row.task_id for row in rows))
    assert summary["formal_score"] == 0.75
    assert summary["defeated_policy_count"] == 1
    assert [row["defeated"] for row in summary["policies"]] == [False, True]


def test_fake_server_executes_all_30_cells_with_unique_artifacts(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    fake_ai = fixtures / "fake_ai.py"
    policies = tuple(
        BuiltPolicy(f"human-{index:02d}", f"Human {index}", fake_ai, f"{index:064x}")
        for index in range(15)
    )
    bundle = PopulationBundle("fixture-v1", "train", 11, (0, 1), policies)
    scheduler = AdaptiveScheduler(
        SchedulerConfig(max_workers=8, sample_seconds=0.001), cpu_sampler=lambda: 60.0
    )
    document = evaluate_matrix(
        MatrixSpec.formal_train(bundle), fake_ai, scheduler, tmp_path,
        server_dir=fixtures / "fake_server", test_only=True,
    )
    assert document["summary"]["expected_matches"] == 30
    assert document["summary"]["completion_rate"] == 1.0
    assert document["summary"]["formal_score"] == 0.0
    replays = [row["replay_path"] for row in document["episodes"]]
    assert len(replays) == len(set(replays)) == 30
    assert all(Path(path).is_file() for path in replays)
    replay_mtimes = {path: Path(path).stat().st_mtime_ns for path in replays}
    resumed = evaluate_matrix(
        MatrixSpec.formal_train(bundle), fake_ai, scheduler, tmp_path,
        server_dir=fixtures / "fake_server", test_only=True,
    )
    assert resumed["summary"]["attempted_matches"] == 30
    assert {path: Path(path).stat().st_mtime_ns for path in replays} == replay_mtimes

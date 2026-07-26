from pathlib import Path
import json

from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.evaluator import (
    GeneralsEvaluator,
    build_evaluation_spec,
    build_learning_cases,
)
from agentbench_frame.generals.models import MatchResult
from agentbench_frame.tracking.run import Run


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"


def test_evaluation_matrix_is_exactly_three_by_three_by_two():
    config = load_pilot_config(FIXTURE)
    spec = build_evaluation_spec(config)
    assert spec.version == "generals-hl-pilot-v1"
    assert len(spec.cases) == 18
    assert len({case.case_id for case in spec.cases}) == 18
    assert {case.seed for case in spec.cases} == {280101, 280202, 280303}
    assert {case.first_player for case in spec.cases} == {0, 1}


def test_learning_cases_exclude_high_and_evaluation_seeds():
    config = load_pilot_config(FIXTURE)
    cases = build_learning_cases(config)
    assert len(cases) == 12
    assert {case.opponent for case in cases} == {
        "advanced-rank08-nashjunheng-v20",
        "popular-rank16-xiaoaojianghu-v1",
    }
    assert {case.seed for case in cases}.isdisjoint({280101, 280202, 280303})


class FakeMatches:
    def __init__(self, outcomes, invalid=None):
        self.outcomes = iter(outcomes)
        self.invalid = invalid

    def __call__(self, case, workspace, version, artifact_dir):
        outcome = next(self.outcomes)
        winner = case.first_player if outcome == "win" else 1 - case.first_player
        if outcome == "draw":
            winner = -1
        valid = case.case_id != self.invalid
        return MatchResult(
            case_id=case.case_id,
            valid=valid,
            winner=winner if valid else None,
            termination_type="normal" if valid else "process_error",
            seed=case.seed,
            evaluated_seat=case.first_player,
            turns=(),
            elapsed_time_s=0.01,
            engine_hash="hash",
            error=None if valid else "boom",
        )


def test_complete_evaluation_scores_draw_as_half(tmp_path):
    config = load_pilot_config(FIXTURE)
    run = Run.start("28_generals", "baseline", data_dir=str(tmp_path))
    executor = FakeMatches(["win"] * 8 + ["draw"] * 2 + ["loss"] * 8)
    result = GeneralsEvaluator(config, executor).evaluate(
        workspace=Path("/agent"), version="v0", phase="evaluation", run=run
    )
    assert result.status == "complete"
    assert result.score == 0.5
    assert set(result.per_tier) == {"high", "medium", "low"}
    assert run.budget_snapshot()["evaluation_episodes"] == 18
    assert run.budget_snapshot()["evaluation_game_agent_decision_steps"] == 0
    run.writer.flush()
    events = [
        json.loads(line)
        for line in (Path(run.run_dir) / "events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event_type"] == "dense_trajectory" for event in events) == 18
    assert sum(event["event_type"] == "dense_episode_summary" for event in events) == 18
    run.finish()


def test_one_invalid_case_keeps_raw_results_but_removes_score(tmp_path):
    config = load_pilot_config(FIXTURE)
    spec = build_evaluation_spec(config)
    run = Run.start("28_generals", "baseline", data_dir=str(tmp_path))
    executor = FakeMatches(["win"] * 18, invalid=spec.cases[0].case_id)
    result = GeneralsEvaluator(config, executor).evaluate(
        workspace=Path("/agent"), version="v0", phase="evaluation", run=run
    )
    assert len(result.results) == 18
    assert result.status == "incomplete"
    assert result.score is None
    run.finish()

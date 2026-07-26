from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import (
    AssetValidationError,
    load_calibration_config,
    load_calibration_selection,
    resolve_calibration_source,
)
from agentbench_frame.generals.calibration import (
    CalibrationEvaluator,
    build_calibration_cases,
    select_calibration_candidate,
)
from agentbench_frame.generals.evaluator import build_evaluation_spec
from agentbench_frame.generals.models import CalibrationEvaluation, MatchResult
from agentbench_frame.generals.process import build_calibration_process
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.tracking.run import Run


FIXTURE = Path(__file__).parent / "fixtures" / "calibration-v1.toml"
SELECTION = Path(__file__).parent / "fixtures" / "calibration-v1-selection.toml"
PILOT_FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"


def development_result(mode, score, status="complete"):
    return CalibrationEvaluation(
        mode=mode,
        split="development",
        status=status,
        score=score,
        wins=0,
        losses=0,
        draws=0,
        per_seat={0: score, 1: score},
        results=(),
        matches=(),
        in_target_range=None,
    )


def test_calibration_manifest_has_disjoint_frozen_splits():
    config = load_calibration_config(FIXTURE)

    assert config.benchmark_id == "generals-hl-calibration-v1"
    assert config.candidate_modes == (
        "passive",
        "local-expander",
        "resource-greedy",
    )
    assert len(build_calibration_cases(config, "development", "passive")) == 10
    assert len(build_calibration_cases(config, "heldout", "passive")) == 10
    assert set(config.development_seeds).isdisjoint(config.heldout_seeds)


def test_calibration_cases_never_enter_formal_spec():
    formal = build_evaluation_spec(load_pilot_config(PILOT_FIXTURE))
    calibration = build_calibration_cases(
        load_calibration_config(FIXTURE), "heldout", "passive"
    )

    assert formal.version == "generals-hl-pilot-v1"
    assert all(case.metadata["suite"] == "calibration" for case in calibration)
    assert {case.case_id for case in formal.cases}.isdisjoint(
        case.case_id for case in calibration
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("development_seeds", (282101, 282101), "development seeds"),
        (
            "heldout_seeds",
            (282101, 282702, 282803, 282904, 283005),
            "seed sets must be disjoint",
        ),
        ("candidate_modes", ("passive", "passive"), "candidate modes"),
        ("target_min", -0.1, "target range"),
        ("target_max", 1.1, "target range"),
        ("target_midpoint", 0.9, "target midpoint"),
    ],
)
def test_invalid_calibration_config_is_rejected(field, value, message):
    from agentbench_frame.generals.assets import validate_calibration_config

    config = load_calibration_config(FIXTURE)
    with pytest.raises(AssetValidationError, match=message):
        validate_calibration_config(replace(config, **{field: value}))


def test_selection_rejects_unknown_mode_and_bad_hash(tmp_path):
    config = load_calibration_config(FIXTURE)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        SELECTION.read_text().replace(
            'selected_mode = "local-expander"', 'selected_mode = "rush"'
        )
    )
    bad_hash = tmp_path / "bad-hash.toml"
    bad_hash.write_text(SELECTION.read_text().replace("a" * 64, "short"))

    with pytest.raises(AssetValidationError, match="selected mode"):
        load_calibration_selection(unknown, config)
    with pytest.raises(AssetValidationError, match="source_hash"):
        load_calibration_selection(bad_hash, config)


def test_calibration_source_must_match_frozen_selection_hash(tmp_path):
    source = tmp_path / "calibration"
    source.mkdir()
    content = b"print('calibration')\n"
    (source / "main.py").write_bytes(content)
    digest = hashlib.sha256()
    digest.update(b"main.py")
    digest.update(b"\0")
    digest.update(content)
    digest.update(b"\0")
    expected_hash = digest.hexdigest()
    config = replace(load_calibration_config(FIXTURE), source=Path("calibration"))
    selection = replace(
        load_calibration_selection(SELECTION, config),
        source_hash=expected_hash,
    )

    assert resolve_calibration_source(config, selection, tmp_path) == source
    with pytest.raises(AssetValidationError, match="source hash"):
        resolve_calibration_source(
            config, replace(selection, source_hash="0" * 64), tmp_path
        )


def test_candidate_selection_uses_distance_to_point_four_then_manifest_order():
    config = load_calibration_config(FIXTURE)
    results = {
        "passive": development_result("passive", 0.7),
        "local-expander": development_result("local-expander", 0.3),
        "resource-greedy": development_result("resource-greedy", 0.5),
    }

    assert select_calibration_candidate(config, results) == "local-expander"


def test_incomplete_candidate_cannot_be_selected():
    config = load_calibration_config(FIXTURE)
    results = {
        mode: development_result(mode, None, status="incomplete")
        for mode in config.candidate_modes
    }

    with pytest.raises(
        ValueError, match="all candidate evaluations must be complete"
    ):
        select_calibration_candidate(config, results)


def test_calibration_process_allows_only_declared_mode(tmp_path):
    source = tmp_path / "calibration"
    source.mkdir()
    (source / "main.py").write_text("")

    spec = build_calibration_process(
        source,
        tmp_path / "engine",
        Path("/python"),
        tmp_path / "sdk",
        "local-expander",
    )

    assert spec.agent_id == "calibration-local-expander"
    assert spec.env["AGENTBENCH_CALIBRATION_MODE"] == "local-expander"
    with pytest.raises(ValueError, match="unsupported calibration mode"):
        build_calibration_process(
            source, tmp_path / "engine", Path("/python"), tmp_path / "sdk", "rush"
        )


class FakeMatches:
    def __init__(self, invalid=None):
        self.invalid = invalid

    def __call__(self, case, workspace, version, artifact_dir):
        valid = case.case_id != self.invalid
        target_wins = case.seed in {282601, 282702}
        winner = case.first_player if target_wins else 1 - case.first_player
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


def test_heldout_calibration_score_and_seat_split_are_separate(tmp_path):
    config = load_calibration_config(FIXTURE)
    run = Run.start("28_generals", "calibration", data_dir=str(tmp_path))

    result = CalibrationEvaluator(config, FakeMatches()).evaluate(
        workspace=tmp_path,
        version="v0",
        split="heldout",
        mode="passive",
        run=run,
        budget_phase="evaluation",
    )

    assert result.status == "complete"
    assert result.score == 0.4
    assert result.wins == 4
    assert result.losses == 6
    assert result.per_seat == {0: 0.4, 1: 0.4}
    assert result.in_target_range is True
    assert run.budget_snapshot()["evaluation_episodes"] == 10
    assert run.budget_snapshot()["learning_episodes"] == 0
    first_artifact = (
        Path(run.run_dir)
        / "matches"
        / "v0"
        / result.matches[0].case_id
    )
    assert (first_artifact / "dense-trace.jsonl").is_file()
    assert (first_artifact / "dense-summary.json").is_file()
    run.finish()


def test_invalid_calibration_case_removes_aggregate_score(tmp_path):
    config = load_calibration_config(FIXTURE)
    cases = build_calibration_cases(config, "heldout", "passive")
    run = Run.start("28_generals", "calibration", data_dir=str(tmp_path))

    result = CalibrationEvaluator(
        config, FakeMatches(invalid=cases[0].case_id)
    ).evaluate(
        workspace=tmp_path,
        version="v0",
        split="heldout",
        mode="passive",
        run=run,
        budget_phase="evaluation",
    )

    assert len(result.results) == 10
    assert result.status == "incomplete"
    assert result.score is None
    assert result.in_target_range is None
    run.finish()

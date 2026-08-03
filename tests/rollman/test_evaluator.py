import dataclasses
from pathlib import Path

import pytest

from agentbench_frame.games.rollman.evaluator import (
    Opponent,
    RollmanEvaluator,
    load_human_pool,
)
from agentbench_frame.games.rollman.match import MatchError, ProcessSpec
from agentbench_frame.hl.codebase import Version


def _version() -> Version:
    return Version(
        version_id="v-test",
        content_hash="a" * 64,
        parent_version_id=None,
        act_id="act-test",
        edit_type="candidate",
        created_at="2026-07-31T00:00:00Z",
        files=("ai.py",),
    )


@dataclasses.dataclass(frozen=True)
class _Replay:
    raw_sha256: str = "raw"
    normalized_sha256: str = "normalized"


@dataclasses.dataclass(frozen=True)
class _Match:
    status: str
    seed: int
    rollman_score: int
    ghosts_score: int
    result: str
    end_state: tuple[str, str] = ("OK", "OK")
    replay: _Replay = _Replay()
    rollman_decisions: tuple = ()
    trace_path: Path = Path("trace.jsonl")


def test_human_pool_is_ranked_and_contains_all_16_opaque_archives():
    manifest = Path(
        "/Users/qingle/Code/SAST/AgentBench/"
        "top_algorithms/corpus/29_rollman_ghost_final/MANIFEST.tsv"
    )
    if not manifest.is_file():
        return

    pool = load_human_pool(manifest)

    assert len(pool) == 16
    assert pool[0].rank == 1
    assert pool[0].opponent_id == "rank01"
    assert pool[-1].rank == 16
    assert all(opponent.archive.suffix == ".zip" for opponent in pool)


def test_learning_evaluation_uses_rank1_and_fixed_seeds(tmp_path):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        seed = kwargs["seed"]
        return _Match(
            status="complete",
            seed=seed,
            rollman_score=10 if seed == 11 else 0,
            ghosts_score=0 if seed == 11 else 10,
            result="win" if seed == 11 else "loss",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate", version.version_id)),
        learning_opponent=Opponent(
            opponent_id="rank01",
            rank=1,
            archive=Path("rank01.zip"),
            process=ProcessSpec(("ghost",)),
        ),
        human_pool=(),
        fixed_gate_seeds=(11, 22),
        certification_seeds=(33,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "complete"
    assert result.score == 0.5
    assert [call["seed"] for call in calls] == [11, 22]
    assert all(call["ghosts"].argv == ("ghost",) for call in calls)
    assert len(result.matches) == 2


def test_learning_target_switches_without_rebuilding_evaluator(tmp_path):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["ghosts"].argv[0])
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    rank15 = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost-15",)),
    )
    rank14 = Opponent(
        opponent_id="rank14",
        rank=14,
        archive=Path("rank14.zip"),
        process=ProcessSpec(("ghost-14",)),
    )
    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=rank15,
        human_pool=(),
        fixed_gate_seeds=(11,),
        certification_seeds=(33,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    first = evaluator.evaluate(_version())
    evaluator.set_learning_opponent(rank14)
    second = evaluator.evaluate(_version())

    assert {match["opponent"] for match in first.matches} == {"rank15"}
    assert {match["opponent"] for match in second.matches} == {"rank14"}
    assert calls == ["ghost-15", "ghost-14"]


def test_any_invalid_fixed_case_makes_aggregate_score_missing(tmp_path):
    calls = 0

    def runner(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MatchError("player process timed out")
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=Opponent(
            opponent_id="rank01",
            rank=1,
            archive=Path("rank01.zip"),
            process=ProcessSpec(("ghost",)),
        ),
        human_pool=(),
        fixed_gate_seeds=(1, 2),
        certification_seeds=(3,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "incomplete"
    assert result.score is None
    assert result.matches[0]["result"] == "win"
    assert result.matches[1]["status"] == "incomplete"


def test_candidate_runtime_error_is_not_scored_as_a_valid_loss(tmp_path):
    def runner(**kwargs):
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=-1000,
            ghosts_score=1000,
            result="loss",
            end_state=("RE", "OK"),
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=Opponent(
            opponent_id="rank16",
            rank=16,
            archive=Path("rank16.zip"),
            process=ProcessSpec(("ghost",)),
        ),
        human_pool=(),
        fixed_gate_seeds=(101,),
        certification_seeds=(201,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "incomplete"
    assert result.score is None
    assert result.matches[0]["status"] == "incomplete"
    assert result.matches[0]["error"] == "candidate ended with RE"
    assert result.matches[0]["end_state"] == ["RE", "OK"]


def test_game_judger_opponent_timeout_is_a_valid_candidate_win(tmp_path):
    def runner(**kwargs):
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1000,
            ghosts_score=-1000,
            result="win",
            end_state=("OK", "TLE"),
        )

    opponent = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost",)),
    )
    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponent,
        human_pool=(opponent,),
        fixed_gate_seeds=(1,),
        certification_seeds=(2,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "complete"
    assert result.score == 1.0
    assert result.matches[0]["end_state"] == ["OK", "TLE"]
    assert result.matches[0]["result"] == "win"


def test_staged_evaluation_uses_one_quick_seed_then_remaining_finalist_seeds(tmp_path):
    calls = []

    def runner(**kwargs):
        calls.append((kwargs["seed"], str(kwargs["replay_path"])))
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    opponent = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost",)),
    )
    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponent,
        human_pool=(opponent,),
        fixed_gate_seeds=(101, 102, 103, 104),
        certification_seeds=(201,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    quick = evaluator.quick_screen(_version())
    finalist = evaluator.evaluate_finalist(_version())
    combined = evaluator.combine_stages(quick, finalist)

    assert [match["seed"] for match in quick.matches] == [101]
    assert [match["seed"] for match in finalist.matches] == [102, 103, 104]
    assert [match["seed"] for match in combined.matches] == [101, 102, 103, 104]
    assert combined.score == 1.0
    assert calls[0][0] == 101


def test_staged_evaluation_uses_configured_two_plus_two_seed_split(tmp_path):
    def runner(**kwargs):
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    opponent = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost",)),
    )
    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponent,
        human_pool=(opponent,),
        fixed_gate_seeds=(101, 102, 103, 104),
        certification_seeds=(201,),
        artifact_root=tmp_path,
        match_runner=runner,
        quick_screen_seed_count=2,
        finalist_seed_count=2,
    )

    quick = evaluator.quick_screen(_version())
    finalist = evaluator.evaluate_finalist(_version())
    combined = evaluator.combine_stages(quick, finalist)

    assert [match["seed"] for match in quick.matches] == [101, 102]
    assert [match["seed"] for match in finalist.matches] == [103, 104]
    assert [match["seed"] for match in combined.matches] == [101, 102, 103, 104]


def test_staged_evaluation_deduplicates_recovered_finalist_matches():
    from agentbench_frame.hl.evaluator import CandidateEvaluation

    quick = CandidateEvaluation(
        status="complete",
        score=0.25,
        matches=tuple(
            {
                "opponent": "rank15",
                "phase": "learning",
                "seed": seed,
                "result": result,
            }
            for seed, result in (
                (101, "loss"),
                (102, "loss"),
                (103, "loss"),
                (104, "win"),
            )
        ),
    )
    finalist = CandidateEvaluation(
        status="complete",
        score=0.5,
        matches=quick.matches[2:],
    )

    combined = RollmanEvaluator.combine_stages(quick, finalist)

    assert len(combined.matches) == 4
    assert combined.score == 0.25


@pytest.mark.parametrize(
    ("quick_count", "finalist_count"),
    ((0, 2), (2, 0), (3, 2)),
)
def test_staged_evaluation_rejects_invalid_seed_counts(
    tmp_path, quick_count, finalist_count
):
    opponent = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost",)),
    )

    with pytest.raises(ValueError, match="seed count"):
        RollmanEvaluator(
            logic=ProcessSpec(("logic",)),
            candidate_factory=lambda version: ProcessSpec(("candidate",)),
            learning_opponent=opponent,
            human_pool=(opponent,),
            fixed_gate_seeds=(101, 102, 103, 104),
            certification_seeds=(201,),
            artifact_root=tmp_path,
            quick_screen_seed_count=quick_count,
            finalist_seed_count=finalist_count,
        )


def test_fixed_cases_can_run_in_parallel_but_preserve_seed_order(tmp_path):
    import threading

    barrier = threading.Barrier(4)

    def runner(**kwargs):
        barrier.wait(timeout=2)
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=kwargs["seed"],
            ghosts_score=0,
            result="win",
        )

    opponent = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost",)),
    )
    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponent,
        human_pool=(opponent,),
        fixed_gate_seeds=(4, 1, 3, 2),
        certification_seeds=(8,),
        artifact_root=tmp_path,
        match_runner=runner,
        max_parallel_matches=4,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "complete"
    assert [match["seed"] for match in result.matches] == [4, 1, 3, 2]


def test_reporting_panel_uses_every_valid_opponent_on_requested_seed(tmp_path):
    opponents = tuple(
        Opponent(
            opponent_id=f"rank{rank:02d}",
            rank=rank,
            archive=Path(f"rank{rank:02d}.zip"),
            process=ProcessSpec((f"ghost-{rank}",)),
        )
        for rank in (14, 15)
    )

    def runner(**kwargs):
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponents[0],
        human_pool=opponents,
        fixed_gate_seeds=(101,),
        certification_seeds=(201,),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.evaluate_reporting_panel(_version(), seeds=(301,))

    assert result.status == "complete"
    assert {(match["opponent"], match["seed"]) for match in result.matches} == {
        ("rank14", 301),
        ("rank15", 301),
    }


def test_certification_runs_every_human_on_same_seed_set(tmp_path):
    opponents = tuple(
        Opponent(
            opponent_id=f"rank{rank:02d}",
            rank=rank,
            archive=Path(f"rank{rank:02d}.zip"),
            process=ProcessSpec((f"ghost-{rank}",)),
        )
        for rank in range(1, 17)
    )
    calls = []

    def runner(**kwargs):
        calls.append((kwargs["ghosts"].argv[0], kwargs["seed"]))
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=opponents[0],
        human_pool=opponents,
        fixed_gate_seeds=(1,),
        certification_seeds=(7, 8),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.certify(_version())

    assert result.status == "complete"
    assert result.score == 1.0
    assert len(calls) == 32
    assert calls[:2] == [("ghost-1", 7), ("ghost-1", 8)]
    assert calls[-2:] == [("ghost-16", 7), ("ghost-16", 8)]


def test_generalizable_certification_runs_only_sealed_hard_opponents(tmp_path):
    opponents = tuple(
        Opponent(
            opponent_id=f"rank{rank:02d}",
            rank=rank,
            archive=Path(f"rank{rank:02d}.zip"),
            process=ProcessSpec((f"ghost-{rank}",)),
        )
        for rank in range(1, 17)
    )
    hard = (opponents[14], opponents[15])
    calls = []

    def runner(**kwargs):
        calls.append((kwargs["ghosts"].argv[0], kwargs["seed"]))
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=hard[0],
        learning_opponents=hard,
        human_pool=opponents,
        certification_opponents=hard,
        fixed_gate_seeds=(1,),
        certification_seeds=(91, 92, 93, 94, 95),
        artifact_root=tmp_path,
        match_runner=runner,
    )

    result = evaluator.certify(_version())

    assert result.status == "complete"
    assert len(calls) == 10
    assert {opponent for opponent, _ in calls} == {"ghost-15", "ghost-16"}


def test_dual_opponent_stages_share_rotating_training_and_validation_cases(tmp_path):
    rank15 = Opponent(
        opponent_id="rank15",
        rank=15,
        archive=Path("rank15.zip"),
        process=ProcessSpec(("ghost-15",)),
    )
    rank16 = Opponent(
        opponent_id="rank16",
        rank=16,
        archive=Path("rank16.zip"),
        process=ProcessSpec(("ghost-16",)),
    )
    calls = []

    def runner(**kwargs):
        calls.append((kwargs["ghosts"].argv[0], kwargs["seed"]))
        return _Match(
            status="complete",
            seed=kwargs["seed"],
            rollman_score=1,
            ghosts_score=0,
            result="win",
        )

    evaluator = RollmanEvaluator(
        logic=ProcessSpec(("logic",)),
        candidate_factory=lambda version: ProcessSpec(("candidate",)),
        learning_opponent=rank15,
        learning_opponents=(rank15, rank16),
        human_pool=(rank15, rank16),
        fixed_gate_seeds=(101, 102, 103, 201, 202),
        training_seeds=(101, 102, 103),
        validation_seeds=(201, 202),
        certification_seeds=(91, 92, 93, 94, 95),
        artifact_root=tmp_path,
        match_runner=runner,
        quick_screen_seed_count=1,
        finalist_seed_count=2,
        training_rotation_stride=1,
    )
    evaluator.set_training_cycle(2)

    assert evaluator.current_learning_cases() == (
        ("rank15", 102),
        ("rank15", 201),
        ("rank15", 202),
        ("rank16", 102),
        ("rank16", 201),
        ("rank16", 202),
    )

    evaluator.quick_screen(_version())
    assert calls == [("ghost-15", 102), ("ghost-16", 102)]

    calls.clear()
    evaluator.evaluate_finalist(_version())
    assert calls == [
        ("ghost-15", 201),
        ("ghost-15", 202),
        ("ghost-16", 201),
        ("ghost-16", 202),
    ]
    assert all(seed < 90_000 for _, seed in calls)

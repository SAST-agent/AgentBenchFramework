import dataclasses
from pathlib import Path

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

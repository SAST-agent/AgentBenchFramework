import dataclasses
import sys
from pathlib import Path


def _version():
    from agentbench_frame.hl.codebase import Version

    return Version(
        version_id="v-test",
        content_hash="a" * 64,
        parent_version_id=None,
        act_id="act-test",
        edit_type="candidate",
        created_at="2026-08-04T00:00:00Z",
        files=("ai.py", "main.py"),
    )


def _record(*, role, seed, result):
    from agentbench_frame.hl.match_record import MatchRecord

    won = result == "win"
    return MatchRecord.from_mapping(
        {
            "schema_version": "1.0",
            "game": "30_antwar2",
            "candidate": "v-test",
            "opponent": "rank01",
            "candidate_role": role,
            "seed": seed,
            "status": "complete",
            "result": result,
            "points": 1.0 if won else 0.0,
            "candidate_score": 10.0 if won else 0.0,
            "opponent_score": 0.0 if won else 10.0,
            "dense_margin": 10.0 if won else -10.0,
            "terminal_metrics": {"p0_camp_hp": 10, "p1_camp_hp": 0},
            "rounds": 12,
            "replay": "replay.json",
            "trace": "trace.jsonl",
            "faults": [],
            "live_opponent": True,
        }
    )


@dataclasses.dataclass(frozen=True)
class _Artifacts:
    record: object


def test_evaluation_covers_both_roles_and_uses_generic_match_records(tmp_path):
    from agentbench_frame.games.antwar2.evaluator import (
        AntWar2Evaluator,
        AntWarOpponent,
    )
    from agentbench_frame.games.antwar2.match import ProcessSpec

    calls = []

    def runner(**kwargs):
        calls.append((kwargs["candidate_role"], kwargs["seed"]))
        result = "win" if kwargs["candidate_role"] == "P0" else "loss"
        return _Artifacts(
            _record(role=kwargs["candidate_role"], seed=kwargs["seed"], result=result)
        )

    process = ProcessSpec((sys.executable, "main.py"), tmp_path)
    opponent = AntWarOpponent("rank01", 1, tmp_path / "rank01.zip", process)
    evaluator = AntWar2Evaluator(
        game=ProcessSpec(("game",), tmp_path),
        candidate_factory=lambda _version: process,
        learning_opponents=(opponent,),
        human_pool=(opponent,),
        fixed_gate_seeds=(7,),
        certification_seeds=(11,),
        artifact_root=tmp_path / "runs",
        match_runner=runner,
    )

    result = evaluator.evaluate(_version())

    assert result.status == "complete"
    assert result.score == 0.5
    assert calls == [("P0", 7), ("P1", 7)]
    assert {match["candidate_role"] for match in result.matches} == {"P0", "P1"}
    assert all(match["schema_version"] == "1.0" for match in result.matches)


def test_control_matrix_selects_exact_opponents_roles_and_seeds(tmp_path):
    from agentbench_frame.games.antwar2.evaluator import (
        AntWar2Evaluator,
        AntWarOpponent,
    )
    from agentbench_frame.games.antwar2.match import ProcessSpec

    calls = []

    def runner(**kwargs):
        calls.append(
            (kwargs["opponent"], kwargs["candidate_role"], kwargs["seed"])
        )
        return _Artifacts(
            _record(
                role=kwargs["candidate_role"],
                seed=kwargs["seed"],
                result="win",
            )
        )

    process = ProcessSpec((sys.executable, "main.py"), tmp_path)
    pool = tuple(
        AntWarOpponent(
            f"rank{rank:02d}", rank, tmp_path / f"rank{rank:02d}.zip", process
        )
        for rank in (1, 2, 3)
    )
    evaluator = AntWar2Evaluator(
        game=ProcessSpec(("game",), tmp_path),
        candidate_factory=lambda _version: process,
        learning_opponents=(pool[0],),
        human_pool=pool,
        fixed_gate_seeds=(7,),
        certification_seeds=(11,),
        artifact_root=tmp_path / "runs",
        match_runner=runner,
    )

    result = evaluator.evaluate_matrix(
        _version(),
        opponent_ids=("rank01", "rank03"),
        roles=("P1",),
        seeds=(1, 7),
        phase="rank01-rank03-audit",
    )

    assert result.status == "complete"
    assert calls == [
        ("rank01", "P1", 1),
        ("rank01", "P1", 7),
        ("rank03", "P1", 1),
        ("rank03", "P1", 7),
    ]


def test_transport_failure_is_fault_record_not_strategy_loss(tmp_path):
    from agentbench_frame.games.antwar2.evaluator import (
        AntWar2Evaluator,
        AntWarOpponent,
    )
    from agentbench_frame.games.antwar2.match import AntWarMatchError, ProcessSpec

    def runner(**_kwargs):
        raise AntWarMatchError("player 1 closed early")

    process = ProcessSpec((sys.executable, "main.py"), tmp_path)
    opponent = AntWarOpponent("rank01", 1, tmp_path / "rank01.zip", process)
    evaluator = AntWar2Evaluator(
        game=ProcessSpec(("game",), tmp_path),
        candidate_factory=lambda _version: process,
        learning_opponents=(opponent,),
        human_pool=(opponent,),
        fixed_gate_seeds=(7,),
        certification_seeds=(11,),
        artifact_root=tmp_path / "runs",
        match_runner=runner,
        roles=("P0",),
    )

    result = evaluator.evaluate(_version())

    assert result.status == "incomplete"
    assert result.score is None
    assert result.matches[0]["status"] == "failed"
    assert result.matches[0]["result"] is None
    assert result.matches[0]["points"] is None
    assert "closed early" in result.matches[0]["faults"][0]


def test_reporting_panel_uses_requested_prefix_of_frozen_seeds(tmp_path):
    from agentbench_frame.games.antwar2.evaluator import (
        AntWar2Evaluator,
        AntWarOpponent,
    )
    from agentbench_frame.games.antwar2.match import ProcessSpec

    calls = []

    def runner(**kwargs):
        calls.append((kwargs["candidate_role"], kwargs["seed"]))
        return _Artifacts(
            _record(
                role=kwargs["candidate_role"],
                seed=kwargs["seed"],
                result="win",
            )
        )

    process = ProcessSpec((sys.executable, "main.py"), tmp_path)
    opponent = AntWarOpponent("rank01", 1, tmp_path / "rank01.zip", process)
    evaluator = AntWar2Evaluator(
        game=ProcessSpec(("game",), tmp_path),
        candidate_factory=lambda _version: process,
        learning_opponents=(opponent,),
        human_pool=(opponent,),
        fixed_gate_seeds=(7,),
        certification_seeds=(11, 12, 13),
        artifact_root=tmp_path / "runs",
        match_runner=runner,
    )

    result = evaluator.evaluate_reporting_panel(_version(), seed_count=1)

    assert result.status == "complete"
    assert calls == [("P0", 11), ("P1", 11)]


def test_human_manifest_maps_only_packages_with_main_entry_as_runnable():
    from agentbench_frame.games.antwar2.evaluator import load_human_pool

    root = Path(
        "/Users/qingle/Code/SAST/AgentBench/top_algorithms/corpus/30_antwar2_ladder"
    )
    if not root.is_dir():
        return

    pool = load_human_pool(root / "MANIFEST.tsv", root / "extracted")

    assert len(pool) == 20
    assert pool[0].opponent_id == "rank01"
    assert pool[-1].opponent_id == "rank20"
    assert pool[0].process is not None
    assert next(item for item in pool if item.rank == 3).process is None

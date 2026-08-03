def _diagnostics(
    version_id,
    *,
    points=0.0,
    margin=-1000.0,
    worst=-1000.0,
    level=0,
    survival=0,
    captures=99,
    novelty=0.0,
    branch=0,
):
    from agentbench_frame.hl.selection import CandidateDiagnostics

    return CandidateDiagnostics(
        version_id=version_id,
        points=points,
        mean_score_margin=margin,
        worst_score_margin=worst,
        max_level=level,
        survival_decisions=survival,
        captures=captures,
        behavioral_novelty=novelty,
        branch_index=branch,
    )


def test_score_margin_improvement_advances_search_parent_without_champion_promotion():
    from agentbench_frame.hl.selection import select_linear_successor

    parent = _diagnostics("v0", margin=-500, worst=-700)
    candidate = _diagnostics("v1", margin=-120, worst=-250)

    decision = select_linear_successor(parent, (candidate,))

    assert decision.search_parent_version_id == "v1"
    assert decision.promote_official_champion is False
    assert decision.reason == "dense_progress"


def test_win_progress_has_priority_over_dense_margin():
    from agentbench_frame.hl.selection import select_linear_successor

    parent = _diagnostics("v0", points=0.0, margin=-20)
    winner = _diagnostics("v1", points=0.5, margin=-100, branch=1)
    margin_only = _diagnostics("v2", points=0.0, margin=100, branch=2)

    decision = select_linear_successor(parent, (margin_only, winner))

    assert decision.search_parent_version_id == "v1"
    assert decision.reason == "target_points_progress"


def test_exploration_debt_falls_back_to_best_specialist_not_origin():
    from agentbench_frame.hl.selection import SearchState, choose_debt_recovery

    state = SearchState(
        search_parent_version_id="v5",
        official_champion_version_id="v0",
        exploration_debt=3,
    )
    decision = choose_debt_recovery(
        state,
        specialists=(
            _diagnostics("v2", margin=-80),
            _diagnostics("v3", margin=-50),
        ),
    )

    assert decision.search_parent_version_id == "v3"
    assert decision.reason == "historical_target_specialist"


def test_diagnostics_ignore_source_size_and_use_only_valid_matches():
    from agentbench_frame.hl.selection import CandidateDiagnostics

    diagnostics = CandidateDiagnostics.from_matches(
        version_id="v7",
        branch_index=2,
        matches=(
            {
                "status": "complete",
                "result": "loss",
                "rollman_score": 40,
                "ghosts_score": 100,
                "game_agent_decisions": 200,
            },
            {
                "status": "incomplete",
                "result": "win",
                "rollman_score": 9999,
                "ghosts_score": -9999,
            },
        ),
    )

    assert diagnostics.points == 0.0
    assert diagnostics.mean_score_margin == -60.0
    assert diagnostics.survival_decisions == 200
    assert not hasattr(diagnostics, "source_lines")


def _dual_diagnostics(version_id, rank15_margin, rank16_margin):
    from agentbench_frame.hl.selection import CandidateDiagnostics

    return CandidateDiagnostics.from_matches(
        version_id=version_id,
        branch_index=0,
        matches=(
            {
                "status": "complete",
                "opponent": "rank15",
                "seed": 1,
                "result": "loss",
                "rollman_score": rank15_margin,
                "ghosts_score": 0,
            },
            {
                "status": "complete",
                "opponent": "rank16",
                "seed": 1,
                "result": "loss",
                "rollman_score": rank16_margin,
                "ghosts_score": 0,
            },
        ),
    )


def test_dual_opponent_successor_rejects_improvement_that_regresses_other_target():
    from agentbench_frame.hl.selection import select_linear_successor

    parent = _dual_diagnostics("v0", -100, -100)
    narrow = _dual_diagnostics("v1", -20, -110)

    decision = select_linear_successor(parent, (narrow,))

    assert decision.search_parent_version_id == "v0"
    assert decision.reason == "no_progress"


def test_dual_opponent_successor_accepts_pareto_dense_progress():
    from agentbench_frame.hl.selection import select_linear_successor

    parent = _dual_diagnostics("v0", -100, -100)
    robust = _dual_diagnostics("v1", -20, -90)

    decision = select_linear_successor(parent, (robust,))

    assert decision.search_parent_version_id == "v1"
    assert decision.reason == "robust_dense_progress"


def _generic_match(role, result, margin, *, live=True):
    points = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result]
    return {
        "schema_version": "1.0",
        "game": "fake",
        "candidate": "candidate",
        "opponent": "human",
        "candidate_role": role,
        "seed": 7,
        "status": "complete",
        "result": result,
        "points": points,
        "candidate_score": margin,
        "opponent_score": 0.0,
        "dense_margin": margin,
        "terminal_metrics": {},
        "rounds": 20,
        "replay": "replay.json",
        "trace": "trace.jsonl",
        "faults": [],
        "live_opponent": live,
    }


def test_diagnostics_accept_generic_match_records_and_group_roles():
    from agentbench_frame.hl.selection import CandidateDiagnostics

    diagnostics = CandidateDiagnostics.from_matches(
        version_id="v1",
        branch_index=0,
        matches=(
            _generic_match("north", "win", 3.0),
            _generic_match("south", "draw", -1.0),
        ),
    )

    assert diagnostics.points == 0.75
    assert diagnostics.mean_score_margin == 1.0
    assert diagnostics.role_points == (("north", 1.0), ("south", 0.5))


def test_successor_rejects_average_gain_that_regresses_one_role():
    from agentbench_frame.hl.selection import (
        CandidateDiagnostics,
        select_linear_successor,
    )

    parent = CandidateDiagnostics.from_matches(
        version_id="v0",
        branch_index=0,
        matches=(
            _generic_match("north", "draw", 0.0),
            _generic_match("south", "draw", 0.0),
        ),
    )
    candidate = CandidateDiagnostics.from_matches(
        version_id="v1",
        branch_index=0,
        matches=(
            _generic_match("north", "win", 10.0),
            _generic_match("south", "loss", 10.0),
        ),
    )

    decision = select_linear_successor(parent, (candidate,))

    assert decision.search_parent_version_id == "v0"
    assert decision.reason == "no_progress"


def test_replay_only_generic_match_cannot_enter_selection():
    import pytest

    from agentbench_frame.hl.selection import CandidateDiagnostics

    with pytest.raises(ValueError, match="valid completed matches"):
        CandidateDiagnostics.from_matches(
            version_id="v1",
            branch_index=0,
            matches=(_generic_match("north", "win", 10.0, live=False),),
        )

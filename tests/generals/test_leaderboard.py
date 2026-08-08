from dataclasses import replace

import pytest

from agentbench_frame.generals.leaderboard import (
    build_leaderboard,
    qualification_result_from_dict,
)
from agentbench_frame.generals.leaderboard_qualification import (
    QUALIFICATION_BUDGETS,
    QUALIFICATION_ID,
    LeaderboardQualificationResult,
    QualificationCheckpoint,
)


def _result(provider, wins):
    checkpoints = []
    for replicate_index in range(1, 4):
        for acts in QUALIFICATION_BUDGETS:
            checkpoint_wins = wins
            checkpoints.append(
                QualificationCheckpoint(
                    replicate_id=f"replicate-{replicate_index}",
                    coding_agent_acts=acts,
                    status="complete",
                    passed=True,
                    policy_hash="p" * 64,
                    valid_games=40,
                    wins=checkpoint_wins,
                    losses=40 - checkpoint_wins,
                    draws=0,
                    score=checkpoint_wins / 40,
                    win_rate=checkpoint_wins / 40,
                    win_rate_lower_bound=0.6,
                    per_seat_wins={0: checkpoint_wins // 2, 1: checkpoint_wins // 2},
                    reasons=(),
                    budget={
                        "coding_agent_acts": acts,
                        "total_tokens": acts * 100,
                        "wall_time_s": acts * 2.0,
                        "learning_episodes": acts * 6,
                        "cost_usd": acts * 0.25,
                    },
                )
            )
    return LeaderboardQualificationResult(
        qualification_id=QUALIFICATION_ID,
        status="qualified",
        qualified=True,
        provider=provider,
        model="frontier-model",
        model_revision="2026-08-09",
        harness_hash="h" * 64,
        qualifying_replicates=("replicate-1", "replicate-2", "replicate-3"),
        minimum_qualifying_replicates=2,
        qualification_budget_acts=8,
        checkpoints=tuple(checkpoints),
        reasons=(),
    )


def _entries(board, acts=1):
    return next(
        item for item in board["checkpoints"]
        if item["coding_agent_acts"] == acts
    )["entries"]


def test_aggregates_replicates_and_ranks_each_frozen_budget():
    board = build_leaderboard([_result("api-a", 30), _result("api-b", 26)])

    entries = _entries(board)
    assert board["provider_count"] == 2
    assert entries[0]["provider"] == "api-a"
    assert entries[0]["rank"] == 1
    assert entries[1]["rank"] == 2
    assert entries[0]["wins"] == 90
    assert entries[0]["budget"] == {
        "coding_agent_acts": 3,
        "total_tokens": 300,
        "wall_time_s": 6.0,
        "learning_episodes": 18,
        "cost_usd": 0.75,
    }
    assert entries[0]["budget_per_replicate"][0]["coding_agent_acts"] == 1


def test_cli_result_round_trip_preserves_checkpoint_budget():
    result = _result("api-a", 30)

    restored = qualification_result_from_dict(result.to_dict())

    assert restored == result


def test_checkpoint_and_budget_object_order_do_not_change_output():
    result = _result("api-a", 30)
    reordered = replace(
        result,
        checkpoints=tuple(
            replace(
                checkpoint,
                budget={
                    field: checkpoint.budget[field]
                    for field in reversed(tuple(checkpoint.budget))
                },
            )
            for checkpoint in reversed(result.checkpoints)
        ),
    )

    assert build_leaderboard([reordered]) == build_leaderboard([result])


def test_publication_rejects_unqualified_results():
    result = _result("api-a", 30)
    unqualified = LeaderboardQualificationResult(
        **{**result.__dict__, "status": "unqualified", "qualified": False}
    )

    with pytest.raises(ValueError, match="requires qualified results"):
        build_leaderboard([unqualified])


def test_publication_rejects_duplicate_identity_or_harness():
    result = _result("api-a", 30)
    duplicate = _result("api-a", 26)
    with pytest.raises(ValueError, match="duplicate provider identities"):
        build_leaderboard([result, duplicate])

    other_harness = _result("api-b", 26)
    other_harness = LeaderboardQualificationResult(
        **{**other_harness.__dict__, "harness_hash": "x" * 64}
    )
    with pytest.raises(ValueError, match="one harness"):
        build_leaderboard([result, other_harness])

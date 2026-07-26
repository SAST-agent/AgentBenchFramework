import json

import pytest

from agentbench_frame.generals.dense import (
    DenseEpisodeSummary,
    DenseMetricSummary,
)
from agentbench_frame.generals.prompt import build_round2_prompt
from agentbench_frame.generals.replay import (
    DecisionRecord,
    LearningReplay,
    build_compact_evidence,
)


def metric(terminal):
    return DenseMetricSummary(
        terminal=terminal,
        minimum=terminal,
        maximum=terminal,
        time_average=terminal,
        auc=terminal,
    )


def dense_summary(replay_id="learning-283101-seat-0"):
    del replay_id
    return DenseEpisodeSummary(
        case_id="learn-low-s283101-p0",
        evaluated_seat=0,
        outcome="loss",
        terminal_round=12,
        completed_rounds_survived=11,
        termination_type="normal",
        terminated=True,
        truncated=False,
        sample_count=12,
        target_territory=metric(3),
        opponent_territory=metric(5),
        territory_margin=metric(-2),
        territory_share=metric(0.375),
        target_army=metric(8),
        opponent_army=metric(15),
        army_margin=metric(-7),
        army_share=metric(8 / 23),
        target_coins=metric(4),
        opponent_coins=metric(10),
        coin_margin=metric(-6),
        coin_share=metric(2 / 7),
        pressure_for=metric(-8),
        pressure_against=metric(-3),
        net_main_pressure=metric(-5),
    )


def replay(seed=283101, seat=0, tier="low", decision_count=6):
    decisions = tuple(
        DecisionRecord(
            state_id=f"state-{index}",
            round_number=index + 1,
            seat=seat,
            state={
                "round": index + 1,
                "cells": {"0,0": {"army": 1000}},
                "opponent_source": "top_algorithms/private.py",
            },
            action=((8,),),
            outcome="loss",
        )
        for index in range(decision_count)
    )
    return LearningReplay(
        replay_id=f"learning-{seed}-seat-{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier=tier,
        termination_type="normal",
        decisions=decisions,
    )


def test_compact_evidence_keeps_first_and_last_decisions_without_board_state():
    compact = build_compact_evidence(
        replay(), dense_summary(), (), max_decisions=4
    )
    encoded = compact.to_json()

    assert compact.outcome == "loss"
    assert compact.dense["completed_rounds_survived"] == 11
    assert [item.state_id for item in compact.decisions] == [
        "state-0",
        "state-3",
        "state-4",
        "state-5",
    ]
    assert "cells" not in encoded
    assert "top_algorithms" not in encoded
    assert "1000" not in encoded


def test_short_episode_keeps_each_decision_exactly_once():
    compact = build_compact_evidence(
        replay(decision_count=3), dense_summary(), (), max_decisions=4
    )

    assert [item.state_id for item in compact.decisions] == [
        "state-0",
        "state-1",
        "state-2",
    ]


def test_round2_prompt_respects_byte_limit_at_whole_record_boundaries():
    records = tuple(
        build_compact_evidence(
            replay(seed=283101 + index, seat=index % 2),
            dense_summary(),
            (),
        )
        for index in range(20)
    )

    result = build_round2_prompt(
        benchmark_id="generals-hl-pilot-v1",
        strategy_doc="current strategy",
        rules_text="official rules",
        replay_guide="field guide",
        first_diff="v0 to v1 diff",
        evidence=records,
        max_bytes=2300,
    )

    assert len(result.prompt.encode("utf-8")) == result.prompt_bytes
    assert result.prompt_bytes <= 2300
    assert result.truncated is True
    assert result.included_episode_ids
    assert result.omitted_episode_ids
    assert set(result.included_episode_ids).isdisjoint(
        result.omitted_episode_ids
    )
    assert result.estimated_tokens == (result.prompt_bytes + 3) // 4
    evidence_text = result.prompt.split("COMPACT LEARNING EPISODES\n", 1)[1]
    parsed = [json.loads(line) for line in evidence_text.splitlines() if line]
    assert [item["replay_id"] for item in parsed] == list(
        result.included_episode_ids
    )


def test_round2_prompt_rejects_limit_smaller_than_mandatory_context():
    with pytest.raises(ValueError, match="mandatory prompt context"):
        build_round2_prompt(
            benchmark_id="generals-hl-pilot-v1",
            strategy_doc="current strategy",
            rules_text="official rules",
            replay_guide="field guide",
            first_diff="v0 to v1 diff",
            evidence=(),
            max_bytes=10,
        )


@pytest.mark.parametrize(
    "forbidden_seed",
    [280101, 280202, 280303, 282601, 282702, 282803, 282904, 283005],
)
def test_round2_prompt_rejects_formal_and_calibration_test_seeds(forbidden_seed):
    forbidden = build_compact_evidence(
        replay(seed=forbidden_seed), dense_summary(), ()
    )

    with pytest.raises(ValueError, match="forbidden evaluation seed"):
        build_round2_prompt(
            benchmark_id="generals-hl-pilot-v1",
            strategy_doc="current strategy",
            rules_text="official rules",
            replay_guide="field guide",
            first_diff="v0 to v1 diff",
            evidence=(forbidden,),
        )


def test_round2_prompt_rejects_high_tier_learning_feedback():
    high = build_compact_evidence(
        replay(tier="high"), dense_summary(), ()
    )

    with pytest.raises(ValueError, match="high-tier feedback"):
        build_round2_prompt(
            benchmark_id="generals-hl-pilot-v1",
            strategy_doc="current strategy",
            rules_text="official rules",
            replay_guide="field guide",
            first_diff="v0 to v1 diff",
            evidence=(high,),
        )

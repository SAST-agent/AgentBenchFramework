import json

import pytest

from agentbench_frame.generals.dense import (
    DenseEpisodeSummary,
    DenseMetricSummary,
)
from agentbench_frame.generals.measurement import DecisionClassSummary
from agentbench_frame.generals.prompt import build_round3_prompt
from agentbench_frame.generals.replay import (
    DecisionRecord,
    LearningReplay,
    build_compact_evidence,
)


def metric(value):
    return DenseMetricSummary(
        terminal=value,
        minimum=value,
        maximum=value,
        time_average=value,
        auc=value,
    )


def evidence(seed=284101, seat=0):
    replay = LearningReplay(
        replay_id=f"learn3-high-s{seed}-p{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        decisions=tuple(
            DecisionRecord(
                state_id=f"state-{index}",
                round_number=index + 1,
                seat=seat,
                state={
                    "cells": {"0,0": {"army": 9999}},
                    "generals": {},
                },
                action=((8,),),
                outcome="loss",
            )
            for index in range(8)
        ),
    )
    dense = DenseEpisodeSummary(
        case_id=replay.replay_id,
        evaluated_seat=seat,
        outcome="loss",
        terminal_round=12,
        completed_rounds_survived=11,
        termination_type="normal",
        terminated=True,
        truncated=False,
        sample_count=12,
        target_territory=metric(2),
        opponent_territory=metric(10),
        territory_margin=metric(-8),
        territory_share=metric(1 / 6),
        target_army=metric(5),
        opponent_army=metric(20),
        army_margin=metric(-15),
        army_share=metric(0.2),
        target_coins=metric(4),
        opponent_coins=metric(8),
        coin_margin=metric(-4),
        coin_share=metric(1 / 3),
        pressure_for=metric(-4),
        pressure_against=metric(-2),
        net_main_pressure=metric(-2),
    )
    return build_compact_evidence(replay, dense, (), max_decisions=4)


def classes():
    return DecisionClassSummary(
        total=100,
        counts={
            "main_army_move": 97,
            "non_main_army_move": 0,
            "end_only": 3,
        },
        rates={
            "main_army_move": 0.97,
            "non_main_army_move": 0.0,
            "end_only": 0.03,
        },
    )


def build(records, max_bytes=65_536):
    return build_round3_prompt(
        benchmark_id="generals-hl-pilot-v1",
        strategy_doc="v2 strategy",
        rules_text="official rules",
        replay_guide="replay fields",
        evidence=records,
        decision_classes=classes(),
        max_bytes=max_bytes,
    )


def test_round3_prompt_accepts_high_tier_learning_and_requires_experience():
    result = build((evidence(),))

    assert result.included_episode_ids == ("learn3-high-s284101-p0",)
    assert result.prompt_bytes <= 65_536
    assert "EXPERIENCE.md" in result.prompt
    assert "search" in result.prompt.lower()
    assert "compress" in result.prompt.lower()
    assert "make one coherent improvement" not in result.prompt.lower()
    assert '"main_army_move": 97' in result.prompt
    assert '"non_main_army_move": 0' in result.prompt
    assert "9999" not in result.prompt
    assert '"cells"' not in result.prompt


@pytest.mark.parametrize(
    "seed",
    [280101, 280202, 280303, 282601, 282702, 282803, 282904, 283005],
)
def test_round3_prompt_rejects_formal_and_heldout_seeds(seed):
    with pytest.raises(ValueError, match="forbidden evaluation seed"):
        build((evidence(seed=seed),))


def test_round3_prompt_truncates_at_whole_episode_boundaries():
    records = tuple(
        evidence(seed=284101 + index, seat=index % 2)
        for index in range(20)
    )

    result = build(records, max_bytes=5_000)

    assert result.truncated is True
    assert result.included_episode_ids
    assert result.omitted_episode_ids
    assert result.prompt_bytes <= 5_000
    evidence_text = result.prompt.split(
        "COMPACT STRONGEST-HUMAN LEARNING EPISODES\n",
        1,
    )[1]
    parsed = [
        json.loads(line)
        for line in evidence_text.splitlines()
        if line
    ]
    assert [item["replay_id"] for item in parsed] == list(
        result.included_episode_ids
    )


def test_round3_prompt_rejects_limit_smaller_than_mandatory_context():
    with pytest.raises(ValueError, match="mandatory prompt context"):
        build((), max_bytes=100)

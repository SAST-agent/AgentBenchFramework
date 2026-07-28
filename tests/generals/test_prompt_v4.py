from dataclasses import replace

import pytest

from agentbench_frame.generals.dense import DenseEpisodeSummary, DenseMetricSummary
from agentbench_frame.generals.measurement import DecisionClassSummary
from agentbench_frame.generals.prompt import build_round4_prompt
from agentbench_frame.generals.replay import (
    DecisionRecord,
    LearningReplay,
    build_critical_learning_evidence,
)


def _metric(value):
    return DenseMetricSummary(value, value, value, value, value)


def _evidence(seed=285101, seat=0):
    state = {
        "round": 1,
        "my_seat": seat,
        "coins": [4, 10],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": {
            "0,0": {
                "type": 0,
                "player": seat,
                "army": 10,
                "general_id": 0,
            },
            "0,2": {
                "type": 0,
                "player": 1 - seat,
                "army": 20,
                "general_id": 1,
            },
        },
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": seat,
                "position": [0, 0],
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": 1 - seat,
                "position": [0, 2],
            },
        },
    }
    replay = LearningReplay(
        replay_id=f"learn4-high-s{seed}-p{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        decisions=(
            DecisionRecord(
                state_id=f"state-{seed}-{seat}",
                round_number=1,
                seat=seat,
                state=state,
                action=((8,),),
                outcome="loss",
            ),
        ),
    )
    fields = {
        name: _metric(0)
        for name in (
            "target_territory",
            "opponent_territory",
            "territory_margin",
            "territory_share",
            "target_army",
            "opponent_army",
            "army_margin",
            "army_share",
            "target_coins",
            "opponent_coins",
            "coin_margin",
            "coin_share",
            "pressure_for",
            "pressure_against",
            "net_main_pressure",
        )
    }
    summary = DenseEpisodeSummary(
        case_id=replay.replay_id,
        evaluated_seat=seat,
        outcome="loss",
        terminal_round=1,
        completed_rounds_survived=0,
        termination_type="normal",
        terminated=True,
        truncated=False,
        sample_count=1,
        **fields,
    )
    return build_critical_learning_evidence(replay, summary, ())[0]


def _classes():
    return DecisionClassSummary(
        total=10,
        counts={
            "main_army_move": 2,
            "non_main_army_move": 7,
            "end_only": 1,
        },
        rates={
            "main_army_move": 0.2,
            "non_main_army_move": 0.7,
            "end_only": 0.1,
        },
    )


def _build(records, max_bytes=65_536):
    return build_round4_prompt(
        benchmark_id="generals-hl-pilot-v1",
        strategy_doc="v3 sends army - 1 from every stack",
        experience_text=(
            "Prior experience cites learn3-high-s284101-p0 and says reserve "
            "logic is missing."
        ),
        rules_text="official rules",
        replay_skill_text="# Human replay skill\nInspect reserves and pressure.",
        replay_skill_sha256="a" * 64,
        evidence=records,
        decision_classes=_classes(),
        max_bytes=max_bytes,
    )


def test_round4_prompt_includes_skill_experience_and_exact_read_receipt():
    records = (_evidence(285101, 0), _evidence(285101, 1))

    result = _build(records)

    assert result.included_episode_ids == (
        "learn4-high-s285101-p0",
        "learn4-high-s285101-p1",
    )
    assert result.feedback_episodes_read == 2
    assert result.feedback_decision_records_read == 2
    expected_bytes = sum(
        len((item.to_json() + "\n").encode("utf-8"))
        for item in records
    )
    assert result.feedback_serialized_bytes_read == expected_bytes
    assert "Human replay skill" in result.prompt
    assert "a" * 64 in result.prompt
    assert "Prior experience" in result.prompt
    assert "army - 1" in result.prompt
    assert "dynamic reserve" in result.prompt.lower()
    assert "formal score" not in result.prompt.lower()


@pytest.mark.parametrize("seed", [280101, 284101])
def test_round4_prompt_rejects_formal_or_old_raw_learning_evidence(seed):
    with pytest.raises(ValueError, match="forbidden round-4 evidence seed"):
        _build((_evidence(seed),))


def test_round4_prompt_rejects_non_high_feedback():
    item = replace(_evidence(), opponent_tier="medium")
    with pytest.raises(ValueError, match="high-tier"):
        _build((item,))


def test_round4_prompt_truncates_only_at_episode_boundaries():
    records = tuple(
        _evidence(285101 + index, index % 2)
        for index in range(20)
    )

    result = _build(records, max_bytes=5_000)

    assert result.truncated is True
    assert result.included_episode_ids
    assert result.omitted_episode_ids
    assert result.prompt_bytes <= 5_000
    included = {
        item.replay_id: item
        for item in records
        if item.replay_id in result.included_episode_ids
    }
    assert result.feedback_decision_records_read == sum(
        len(item.decisions) for item in included.values()
    )


def test_round4_prompt_rejects_limit_smaller_than_mandatory_context():
    with pytest.raises(ValueError, match="mandatory prompt context"):
        _build((), max_bytes=100)

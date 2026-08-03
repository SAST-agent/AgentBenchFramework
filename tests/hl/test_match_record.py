import math

import pytest


def _valid(**overrides):
    value = {
        "schema_version": "1.0",
        "game": "fake",
        "candidate": "v1",
        "opponent": "human-1",
        "candidate_role": "north",
        "seed": 7,
        "status": "complete",
        "result": "win",
        "points": 1.0,
        "candidate_score": 3.0,
        "opponent_score": 1.0,
        "dense_margin": 2.0,
        "terminal_metrics": {"camp_hp": 3.0},
        "rounds": 12,
        "replay": "r.json",
        "trace": "t.jsonl",
        "faults": [],
        "live_opponent": True,
    }
    value.update(overrides)
    return value


def test_complete_match_exposes_generic_comparison_and_round_trip():
    from agentbench_frame.hl.match_record import MatchRecord

    match = MatchRecord.from_mapping(_valid())

    assert match.comparison_key == ("human-1", "north", 7)
    assert match.promotable is True
    assert match.to_dict() == _valid()


@pytest.mark.parametrize(
    "overrides",
    (
        {"status": "failed", "result": None, "points": None,
         "candidate_score": None, "opponent_score": None,
         "dense_margin": None, "rounds": None, "faults": ["RE"]},
        {"live_opponent": False},
    ),
)
def test_faulted_or_replay_only_match_is_not_promotable(overrides):
    from agentbench_frame.hl.match_record import MatchRecord

    assert MatchRecord.from_mapping(_valid(**overrides)).promotable is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"unexpected": 1}, "unknown match record fields"),
        ({"points": 0.5}, "win result requires points 1.0"),
        ({"dense_margin": math.inf}, "dense_margin must be finite"),
        ({"terminal_metrics": {"camp_hp": math.nan}},
         "terminal metric camp_hp must be finite"),
        ({"status": "complete", "result": None},
         "complete match requires result"),
        ({"status": "failed", "faults": [], "result": None, "points": None,
          "candidate_score": None, "opponent_score": None,
          "dense_margin": None, "rounds": None},
         "non-complete match requires a fault"),
    ),
)
def test_match_record_rejects_inconsistent_or_non_finite_values(
    overrides,
    message,
):
    from agentbench_frame.hl.match_record import MatchRecord

    with pytest.raises(ValueError, match=message):
        MatchRecord.from_mapping(_valid(**overrides))


def test_completed_match_records_filters_faults_and_replay_screens():
    from agentbench_frame.hl.match_record import completed_match_records

    records = completed_match_records(
        (
            _valid(candidate="v1"),
            _valid(candidate="v2", live_opponent=False),
            _valid(
                candidate="v3",
                status="timeout",
                result=None,
                points=None,
                candidate_score=None,
                opponent_score=None,
                dense_margin=None,
                rounds=None,
                faults=["timeout"],
            ),
        )
    )

    assert [record.candidate for record in records] == ["v1"]

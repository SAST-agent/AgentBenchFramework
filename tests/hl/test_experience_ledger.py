import dataclasses

import pytest


def _match(
    opponent,
    role,
    seed,
    candidate_score,
    opponent_score,
    *,
    result=None,
    live_opponent=True,
):
    if result is None:
        result = (
            "win"
            if candidate_score > opponent_score
            else "draw"
            if candidate_score == opponent_score
            else "loss"
        )
    return {
        "schema_version": "1.0",
        "game": "antwar2",
        "candidate": "v000001" if candidate_score == 0 else "v000002",
        "status": "complete",
        "opponent": opponent,
        "candidate_role": role,
        "seed": seed,
        "result": result,
        "points": {"win": 1.0, "draw": 0.5, "loss": 0.0}[result],
        "candidate_score": candidate_score,
        "opponent_score": opponent_score,
        "dense_margin": candidate_score - opponent_score,
        "terminal_metrics": {},
        "rounds": 100,
        "replay": f"matches/{opponent}/{seed}/replay.jsonl",
        "trace": f"matches/{opponent}/{seed}/trace.jsonl",
        "faults": [],
        "live_opponent": live_opponent,
    }


def _derive(*, candidate_matches, activation=None, mechanism="bounded breakout"):
    from agentbench_frame.hl.experience_ledger import derive_experience_record

    return derive_experience_record(
        iteration_id="iter-000001",
        act_id="act-000002-b02",
        branch_index=2,
        parent_version_id="v000001",
        candidate_version_id="v000002",
        selected=False,
        brief={
            "activation_condition": "level == 3 and recent observable respawn",
            "mechanism": mechanism,
            "preservation_contract": "outside the respawn window preserve the parent",
        },
        parent_matches=(
            _match("rank15", "P0", 11, 0, 100),
            _match("rank15", "P1", 12, 0, 100),
        ),
        candidate_matches=tuple(candidate_matches),
        activation=activation
        or {
            "status": "complete",
            "decision_count": 20,
            "changed_action_count": 3,
        },
    )


def test_mixed_record_preserves_good_and_bad_match_conditions():
    record = _derive(
        candidate_matches=(
            _match("rank15", "P0", 11, 60, 100),
            _match("rank15", "P1", 12, -50, 100),
        )
    )

    assert record.verdict == "mixed"
    assert [row.dense_margin_delta for row in record.comparisons] == [60.0, -50.0]
    assert [row.candidate_role for row in record.comparisons] == ["P0", "P1"]
    assert record.activation_condition == "level == 3 and recent observable respawn"


def test_record_is_verified_good_only_without_a_measured_regression():
    record = _derive(
        candidate_matches=(
            _match("rank15", "P0", 11, 60, 100),
            _match("rank15", "P1", 12, 10, 100),
        )
    )

    assert record.verdict == "verified_good"


def test_zero_activation_is_invalid_even_when_matches_are_complete():
    record = _derive(
        candidate_matches=(
            _match("rank15", "P0", 11, 60, 100),
            _match("rank15", "P1", 12, 10, 100),
        ),
        activation={
            "status": "complete",
            "decision_count": 20,
            "changed_action_count": 0,
        },
    )

    assert record.verdict == "invalid"


def test_ledger_round_trips_canonical_jsonl_and_is_idempotent(tmp_path):
    from agentbench_frame.hl.experience_ledger import ExperienceLedger

    record = _derive(
        candidate_matches=(
            _match("rank15", "P0", 11, 60, 100),
            _match("rank15", "P1", 12, 10, 100),
        )
    )
    ledger = ExperienceLedger(tmp_path / "ledger.jsonl")

    assert ledger.append(record) is True
    assert ledger.append(record) is False
    assert ledger.load() == (record,)
    assert (tmp_path / "ledger.jsonl").read_text().count("\n") == 1


def test_ledger_rejects_credential_material(tmp_path):
    from agentbench_frame.hl.experience_ledger import ExperienceLedger

    record = _derive(
        candidate_matches=(
            _match("rank15", "P0", 11, 60, 100),
            _match("rank15", "P1", 12, 10, 100),
        )
    )
    unsafe = dataclasses.replace(record, mechanism="sk-abcdefghijklmno")

    with pytest.raises(ValueError, match="credential"):
        ExperienceLedger(tmp_path / "ledger.jsonl").append(unsafe)


def test_comparisons_join_on_role_and_ignore_non_live_diagnostics():
    record = _derive(
        candidate_matches=(
            _match("rank15", "P1", 11, 900, 0),
            _match("rank15", "P0", 11, 60, 100, live_opponent=False),
        )
    )

    assert record.verdict == "inconclusive"
    assert record.comparisons == ()

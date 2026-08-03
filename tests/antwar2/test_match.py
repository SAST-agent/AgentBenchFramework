import json


def _replay(*, winner=0, camps=(6, 0)):
    return [
        {
            "seed": 7,
            "op0": [],
            "op1": [],
            "round_state": {
                "camps": [50, 50],
                "coins": [50, 50],
                "towers": [],
                "ants": [],
                "winner": -1,
            },
        },
        {
            "op0": [{"type": 21, "id": -1, "args": -1,
                     "pos": {"x": 16, "y": 9}}],
            "op1": [],
            "round_state": {
                "camps": list(camps),
                "coins": [12, 9],
                "towers": [],
                "ants": [],
                "winner": winner,
            },
        },
    ]


def test_completed_replay_becomes_role_correct_generic_match_record(tmp_path):
    from agentbench_frame.games.antwar2.match import record_from_replay

    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_replay()), encoding="utf-8")

    p0 = record_from_replay(
        replay,
        candidate="v239",
        opponent="rank01",
        candidate_role="P0",
        seed=7,
    )
    p1 = record_from_replay(
        replay,
        candidate="human",
        opponent="v239",
        candidate_role="P1",
        seed=7,
    )

    assert p0.result == "win"
    assert p0.points == 1.0
    assert p0.candidate_score == 6
    assert p0.opponent_score == 0
    assert p0.dense_margin == 6
    assert p0.terminal_metrics == {"p0_camp_hp": 6.0, "p1_camp_hp": 0.0}
    assert p0.rounds == 2
    assert p1.result == "loss"
    assert p1.dense_margin == -6


def test_replay_trace_contains_only_public_state_and_accepted_atoms(tmp_path):
    from agentbench_frame.games.antwar2.match import write_public_trace

    replay = tmp_path / "replay.json"
    trace = tmp_path / "trace.jsonl"
    replay.write_text(json.dumps(_replay()), encoding="utf-8")

    write_public_trace(replay, trace)

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert {record["kind"] for record in records} == {
        "ai_operations",
        "public_state",
    }
    assert len(records) == 6
    assert records[3]["operations"][0]["type"] == 21
    assert records[-1]["public_state"]["winner"] == 0


def test_replay_without_valid_terminal_winner_is_rejected(tmp_path):
    import pytest

    from agentbench_frame.games.antwar2.match import AntWarMatchError, record_from_replay

    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_replay(winner=-1, camps=(50, 50))), encoding="utf-8")

    with pytest.raises(AntWarMatchError, match="terminal winner"):
        record_from_replay(
            replay,
            candidate="v1",
            opponent="rank01",
            candidate_role="P0",
            seed=1,
        )

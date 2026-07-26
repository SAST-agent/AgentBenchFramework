from agentbench_frame.generals.models import MatchResult, TurnRecord
from agentbench_frame.generals.prompt import build_codex_prompt
from agentbench_frame.generals.replay import build_learning_replay


def match():
    turn = TurnRecord(
        step=0,
        round_number=1,
        player=0,
        state_id_before="state-1",
        state_before={"round": 1, "coins": [40, 40]},
        commands=((8,),),
        state_id_after="state-2",
        state_after={"round": 1, "coins": [40, 40]},
    )
    return MatchResult(
        case_id="learn-medium-rank08-s281101-p0",
        valid=True,
        winner=1,
        termination_type="normal",
        seed=281101,
        evaluated_seat=0,
        turns=(turn,),
        elapsed_time_s=1.0,
        engine_hash="hash",
    )


def test_learning_replay_keeps_decisions_and_redacts_opponent_identity():
    replay = build_learning_replay(match(), evaluated_agent_id="baseline", opponent_tier="medium")
    assert replay.seed == 281101
    assert replay.decisions
    encoded = replay.to_json()
    assert "rank08" not in encoded
    assert "top_algorithms" not in encoded


def test_prompt_contains_learning_evidence_but_no_evaluation_material():
    replay = build_learning_replay(match(), "baseline", "medium")
    prompt = build_codex_prompt(
        benchmark_id="generals-hl-pilot-v1",
        strategy_doc="Current rules",
        rules_text="Official rules",
        replay_guide="Replay fields",
        learning_replays=(replay,),
    )
    assert "281101" in prompt
    assert "280101" not in prompt
    assert "opponent source" not in prompt.lower()
    assert "codex exec" not in prompt.lower()
    assert "/home/cathy" not in prompt

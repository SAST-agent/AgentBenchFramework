""".mrc → 事件时间线，并与同局 trace 交叉核对。"""

import json

import pytest

from agentbench_frame.miracle import EndRoundAgent, SampleAgent, run_match
from agentbench_frame.miracle.replay import (
    EVENT_NAMES,
    parse_replay,
    save_replay_json,
    load_replay_json,
    summarize,
)


@pytest.fixture(scope="module")
def full_match_replay(tmp_path_factory):
    out = tmp_path_factory.mktemp("miracle-replay")
    result = run_match(SampleAgent(), EndRoundAgent(), replay_dir=out, seed=11)
    assert result.terminated_by == "normal"
    assert result.errors == []
    return result.replay_path


def _ops_from_trace(trace_path):
    rows = [json.loads(line) for line in open(trace_path, encoding="utf-8")]
    return [json.loads(r["payload"]["content"]) for r in rows if r["kind"] == "to_logic"]


def test_event_name_tables_match_official():
    # 与官方 gamecode_logic/main.py 的事件表逐字一致
    assert EVENT_NAMES[3] == "Spawn" and EVENT_NAMES[10] == "GameEnd" and EVENT_NAMES[18] == "Summon"
    assert len(EVENT_NAMES) == 19


def test_parse_full_match_matches_trace_operations(full_match_replay):
    events = parse_replay(full_match_replay)
    s = summarize(events)

    assert s["ended"] is True
    assert s["winner"] == 0
    assert s["end_round"] >= 1

    # 事件与 trace 操作分布对齐（官方 1 个操作 → 1 个主事件）
    ops = _ops_from_trace(full_match_replay + ".trace.jsonl")
    from collections import Counter
    counts = Counter(o["operation_type"] for o in ops)
    assert s["event_counts"]["Summon"] == counts["summon"]
    assert s["event_counts"]["Move"] == counts["move"]
    assert s["event_counts"]["Attack"] == counts["attack"]
    assert s["event_counts"]["TurnEnd"] == counts["endround"]
    assert s["event_counts"]["GameStart"] == counts["init"]


def test_game_end_and_round_span(full_match_replay):
    events = parse_replay(full_match_replay)
    game_ends = [e for e in events if e.type == "GameEnd"]
    assert len(game_ends) == 1
    assert game_ends[0].args[0] in (0, 1)  # winner
    rounds = [e.round for e in events if e.round >= 0]
    assert max(rounds) == game_ends[0].round


def test_no_unsync_on_full_match(full_match_replay):
    events = parse_replay(full_match_replay)
    assert not [e for e in events if e.type in ("UNSYNC", "UNKNOWN")], "存在未同步事件"


def test_events_round_monotonic(full_match_replay):
    events = [e for e in parse_replay(full_match_replay) if e.round >= 0]
    prev = -1
    for e in events:
        assert e.round >= prev
        prev = e.round


def test_save_and_load_json_roundtrip(tmp_path, full_match_replay):
    events = parse_replay(full_match_replay)
    out = tmp_path / "events.jsonl"
    save_replay_json(events, out)
    reloaded = load_replay_json(out)
    assert len(reloaded) == len(events)
    assert reloaded[0].type == events[0].type
    assert reloaded[-1].type == "END"

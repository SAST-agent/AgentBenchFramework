"""Phase 1 验收：官方逻辑零改动子进程运行 + 规则策略对战闭环。

覆盖：
- 线协议编解码往返
- 双 endround：100 回合平局（scores=(0,1)、winner=1、normal 终局）
- sample vs endround：正常终局、replay/trace 落盘、行动类型齐全、无卡死/超时
"""

import json

import pytest

from agentbench_frame.miracle import EndRoundAgent, SampleAgent, run_match
from agentbench_frame.miracle.protocol import (
    decode_content,
    encode_to_logic,
    read_logic_frame,
)


def _ops_of(trace_path):
    ops = []
    for line in open(trace_path, encoding="utf-8"):
        row = json.loads(line)
        if row["kind"] == "to_logic" and row["payload"].get("player") in (0, 1):
            inner = json.loads(row["payload"]["content"])
            ops.append(inner.get("operation_type"))
    return ops


class TestProtocol:
    def test_roundtrip(self):
        payload = {"player_list": [1, 1], "replay": "/tmp/x.mrc"}
        raw = encode_to_logic(payload)
        assert raw[:4] == b"\x00\x00\x00" + bytes([len(raw) - 4])
        data = raw[4:]
        assert json.loads(data) == payload

    def test_decode_content_parse(self):
        payload = {"state": 5, "listen": [0], "content": ["000277" + json.dumps({"a": 1})]}
        assert decode_content(payload["content"]) == {"a": 1}


class TestFullMatch:
    def test_endround_pair_hits_round_cap(self, tmp_path):
        r = run_match(
            EndRoundAgent(), EndRoundAgent(),
            replay_dir=tmp_path, seed=7, tag="t_endround",
        )
        assert r.terminated_by == "normal"
        assert r.errors == []
        assert r.rounds >= 99  # 100 回合封顶
        # 官方终局语义：平局时后手 +1 分、winner=1
        assert r.scores == (0, 1)
        assert r.winner == 1
        import os
        assert os.path.exists(r.replay_path)
        assert os.path.getsize(r.replay_path) > 0
        assert os.path.exists(r.trace_path)

    def test_sample_vs_endround_finishes_and_acts(self, tmp_path):
        r = run_match(
            SampleAgent(), EndRoundAgent(),
            replay_dir=tmp_path, seed=11, tag="t_sample",
        )
        assert r.terminated_by == "normal", r.errors
        assert r.errors == []
        ops = _ops_of(r.trace_path)
        kinds = set(ops)
        # 规则策略应产生召唤/移动/攻击/endround（攻破神迹或打满 100 回合）
        assert "summon" in kinds
        assert "move" in kinds
        assert "attack" in kinds
        assert "endround" in kinds
        # 整局无卡死：若中途触发防卡死 timeout，terminated_by 会是 timeout
        import os
        assert os.path.getsize(r.replay_path) > 0

    def test_sample_vs_sample_reproducible(self, tmp_path):
        r1 = run_match(SampleAgent(), SampleAgent(), replay_dir=tmp_path, seed=3, tag="t_ss")
        r2 = run_match(SampleAgent(), SampleAgent(), replay_dir=tmp_path, seed=3, tag="t_ss2")
        assert r1.scores == r2.scores
        assert r1.winner == r2.winner
        assert r1.terminated_by == r2.terminated_by == "normal"

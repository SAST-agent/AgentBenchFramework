"""回放阅读 Skill 的实际解析验证（要求 2）：
用真实 trace（agentbench_data/replays/24_miracle/*.trace.jsonl）按
.miracle-replay-reader Skill 的字段定义解析，验证格式自洽。

无本地 trace 时跳过（产物在 gitignore 中，可由 test_smoke 重新生成）。
"""

import glob
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentbench_frame.miracle.decision_space import (
    PLAYER,
    UNIT,
    action_mask,
    termination_reason,
)

TRACES = sorted(glob.glob("agentbench_data/replays/24_miracle/*.trace.jsonl"))


def _parse_content(content: str) -> dict:
    """按 Skill：content 前 6 位是长度前缀，其后才是 JSON。"""
    assert len(content) >= 6
    body = content[6:]
    return json.loads(body)


@pytest.fixture(scope="module")
def trace():
    if not TRACES:
        pytest.skip("无本地 trace 样本，先跑 tests/miracle/test_smoke.py 生成")
    rows = [json.loads(line) for line in open(TRACES[-1], encoding="utf-8")]
    assert rows, "trace 为空"
    return rows


def test_trace_kind_and_state_sequence(trace):
    kinds = [r["kind"] for r in trace]
    assert kinds[0] == "init"
    assert "to_logic" in kinds
    assert "from_logic" in kinds
    states = [r["state"] for r in trace if r["kind"] == "from_logic"]
    assert states[0] == 0          # 选卡请求
    assert 2 in states             # 回合消息
    assert 3 in states             # 终局帧存在
    assert states[-1] in (3, -1)   # -1 = 官方错误帧（异常对局如实保留）


def test_payload_and_summary_consistent(trace):
    for r in trace:
        if r["kind"] not in ("from_logic", "to_logic"):
            continue  # init 帧的 payload 含 replay 元数据，格式不同
        # summary 是 payload 的 JSON 截断（host._send 截到 200 字符），应为其前缀
        full = json.dumps(r["payload"], ensure_ascii=False)
        assert full.startswith(r["summary"]), "summary 不是 payload JSON 的前缀"


def test_round_content_length_prefix(trace):
    """Skill 关键点：content 前 6 位是长度前缀。"""
    round_rows = [r for r in trace if r["kind"] == "from_logic" and r["state"] == 2]
    assert round_rows
    for r in round_rows:
        content = r["payload"]["content"][0]
        prefix_len = int(content[:6])
        body = json.loads(content[6:])
        assert len(content) == prefix_len + 6, "长度前缀与实际体长不一致"


def test_obs_fields_indices(trace):
    """按 Skill 的 18/5 字段索引在真实 obs 上成立。"""
    obs = None
    for r in trace:
        if r["kind"] == "from_logic" and r["state"] == 3:
            content = r["payload"]["content"][0]
            obs = _parse_content(content)
            break
    assert obs is not None, "无终局帧"
    for u in obs["map"]["units"]:
        assert len(u) == 18
        assert isinstance(u[UNIT["ID"]], int)
        assert u[UNIT["CAMP"]] in (0, 1)
        assert len(u[UNIT["POS"]]) == 3
        assert u[UNIT["CAN_ATK"]] in (0, 1)
        assert u[UNIT["CAN_MOVE"]] in (0, 1)
    assert len(obs["players"]) == 2
    for p in obs["players"]:
        assert len(p) == 5
        assert isinstance(p[PLAYER["MANA"]], int)
        assert isinstance(p[PLAYER["CAPACITIES"]], list)


def test_actions_parse_and_align(trace):
    """to_logic 操作可按官方 dict 解析，且 type 都在决策空间宏动作集合内。"""
    known = {"init", "summon", "move", "attack", "endround"}
    for r in trace:
        if r["kind"] != "to_logic":
            continue
        op = json.loads(r["payload"]["content"])
        assert op["operation_type"] in known
        assert "operation_parameters" in op
        assert op["player"] in (0, 1)


def test_terminal_classification(trace):
    """终局帧可用可观测特征映射到 decision_space 的终止条件分类（枚举合法、函数可用）。

    注：权威终止分类由 match.MatchResult 提供（见 test_smoke）；此处验证
    trace 里的终局帧确实能按同一口径分类。"""
    finals = [r for r in trace if r["kind"] == "from_logic" and r["state"] == 3]
    assert finals, "无终局帧"
    # 取 content 最长的一帧作为有效终局（官方可能重发空终局帧）
    best = max(finals, key=lambda r: len(r["payload"]["content"][0]))
    obs = _parse_content(best["payload"]["content"][0])
    miracles = obs["map"].get("miracles", [30, 30])
    rounds = obs.get("round", 0)
    if any(hp <= 0 for hp in miracles):
        reason = termination_reason(terminated_by="normal", rounds=rounds, scores=(30000, 0), errors=[])
        assert reason == "miracle_destroyed"
    elif rounds >= 99:
        reason = termination_reason(terminated_by="normal", rounds=rounds, scores=(0, 1), errors=[])
        assert reason == "max_round"
    else:
        # 观测信息不足时如实记 unknown（不冒充）
        reason = termination_reason(terminated_by="normal", rounds=rounds, scores=(0, 0), errors=[])
        assert reason == "unknown"


def test_mask_sanity_on_final_obs(trace):
    """终局 obs 上 action_mask 仍可运行（skill 工作流第 3 步的可用性）。"""
    for r in trace:
        if r["kind"] == "from_logic" and r["state"] == 3:
            obs = _parse_content(r["payload"]["content"][0])
            obs["camp"] = 0
            mask = action_mask(obs, camp=0)
            assert all(a.type in ("summon", "move", "attack", "endround") for a in mask)
            return
    pytest.skip("无终局帧")

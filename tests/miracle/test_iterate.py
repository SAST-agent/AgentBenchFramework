"""迭代闭环测试（要求 1 后半 + 要求 5）：决策点收集、ε-soft 分布、导出契约、曲线。

用已有真实对局产物（trace），不重跑对局；无产物时跳过。
"""

import glob
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentbench_frame.miracle.agent_bridge import SampleAgent, SampleV2Agent
from agentbench_frame.miracle.ig import compute_kl
from agentbench_frame.miracle.iterate import (
    AGENTS,
    collect_decision_points,
    curves_ascii,
    epsilon_soft,
    export_run,
    _obs_from_trace,
)
from agentbench_frame.miracle.match import MatchResult

TRACES = sorted(glob.glob("agentbench_data/replays/24_miracle/*sample*.trace.jsonl"))


def _long_trace():
    if not TRACES:
        pytest.skip("无本地 trace，先跑 tests/miracle/test_smoke.py 生成")
    # 取行数最多（最完整）的对局
    return max(TRACES, key=lambda p: len(open(p, encoding="utf-8").readlines()))


# ---------------------------------------------------------------------------
# 决策点收集（要求 1 后半：从回放提取可对齐的决策点）
# ---------------------------------------------------------------------------

def test_obs_from_trace_extracts_full_obs():
    trace = _long_trace()
    obs_list = _obs_from_trace(trace)
    assert len(obs_list) >= 10, "完整对局应提取 ≥10 个决策点 obs"
    for obs in obs_list:
        assert "map" in obs and "players" in obs
        assert "camp" in obs


def test_collect_decision_points_aligned():
    trace = _long_trace()
    v1 = SampleAgent(seed=11)
    v2 = SampleV2Agent(seed=11)
    points = collect_decision_points(trace, v1, v2)
    assert len(points) >= 10
    # 每个决策点：新旧分布都在支撑集内且归一化
    for p in points:
        assert abs(sum(p.p_new.values()) - 1.0) < 1e-9
        assert abs(sum(p.p_old.values()) - 1.0) < 1e-9
        # ε-soft 保证支撑集相同 ⇒ KL 有限（不应有缺失）
        kl, reason = compute_kl(p.p_new, p.p_old)
        assert reason is None


def test_epsilon_soft_properties():
    from agentbench_frame.miracle.decision_space import Action
    mask = [
        Action("summon", {"type": "Swordsman", "level": 1, "position": [-8, 6, 2]}),
        Action("move", {"mover": 1, "position": [-7, 6, 1]}),
        Action("endround"),
    ]
    chosen = {"operation_type": "move", "operation_parameters": {"mover": 1, "position": [-7, 6, 1]}}
    dist = epsilon_soft(chosen, mask, eps=0.1)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert max(dist.values()) == pytest.approx(0.9)  # 最优动作 1-ε
    assert len(dist) == 3
    # 选择不在 mask（被拒）→ 退化为均匀
    bad = {"operation_type": "attack", "operation_parameters": {"attacker": 99, "target": 99}}
    dist2 = epsilon_soft(bad, mask)
    assert abs(sum(dist2.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-9 for v in dist2.values())
    # 单动作 mask → 唯一动作概率 1
    dist3 = epsilon_soft({"operation_type": "endround", "operation_parameters": {}},
                         [Action("endround")])
    assert dist3 == {Action("endround").signature(): 1.0}


def test_v2_is_real_strategy_update():
    """v2 与 v1 必须真实不同（有效策略更新）：召唤优先级反转。"""
    assert SampleV2Agent.summon_order != SampleAgent.summon_order
    assert SampleV2Agent.summon_order[0] == "BlackBat"
    assert SampleAgent.summon_order[0] == "Archer"


# ---------------------------------------------------------------------------
# 导出契约（AgentBenchResults 对接）
# ---------------------------------------------------------------------------

def _fake_eval():
    match = MatchResult(
        winner=1, scores=(0, 1), rounds=99,
        replay_path="agentbench_data/replays/24_miracle/x.mrc",
        trace_path="agentbench_data/replays/24_miracle/x.mrc.trace.jsonl",
        terminated_by="normal", errors=[], duration=1.0,
    )
    trace = _long_trace()
    points = collect_decision_points(trace, SampleAgent(seed=11), SampleV2Agent(seed=11))
    from agentbench_frame.miracle.ig import episode_ig
    ig = episode_ig(points, episode_id="ep_test", iteration=1)
    return SimpleNamespace(iteration=1, version="sample_v2", agent_name="sample",
                           seed=11, match=match, decisions=points, ig=ig,
                           score=0)


from types import SimpleNamespace


def test_export_run_contract(tmp_path):
    ev = _fake_eval()
    run_dir = export_run(ev, runs_root=tmp_path, agent_dir_name="sample")
    assert (run_dir / "run.toml").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "ig.json").exists()

    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # AgentBenchResults 需要的标准字段 + 我们的扩展字段
    for key in ("run_type", "created", "git_commit", "game", "agent",
                "agent_version", "iteration", "winner", "score", "win_rate",
                "episode_ig", "ig_coverable_ratio", "ig_missing", "replay", "trace"):
        assert key in s, f"summary.json 缺 {key}"
    assert s["agent_version"] == "sample_v2"
    assert s["iteration"] == 1
    assert s["episode_ig"] is not None

    toml = (run_dir / "run.toml").read_text(encoding="utf-8")
    assert "agent_version = \"sample_v2\"" in toml
    assert "iteration = 1" in toml


def test_build_curves_version_aligned(tmp_path):
    ev1 = _fake_eval()
    ev1 = SimpleNamespace(iteration=0, version="sample", agent_name="sample",
                          seed=11, match=ev1.match, decisions=[],
                          ig={}, score=30000)
    export_run(ev1, runs_root=tmp_path, agent_dir_name="sample")
    ev2 = _fake_eval()
    export_run(ev2, runs_root=tmp_path, agent_dir_name="sample")

    from agentbench_frame.miracle.iterate import build_curves
    curves = build_curves(runs_root=tmp_path, agent="sample")
    assert len(curves["points"]) == 2
    by_iter = {p["iteration"]: p for p in curves["points"]}
    assert by_iter[0]["version"] == "sample"
    assert by_iter[0]["score"] == 30000
    assert by_iter[1]["version"] == "sample_v2"
    assert by_iter[1]["episode_ig"] is not None
    text = curves_ascii(curves)
    assert "sample" in text and "sample_v2" in text and "30000" in text


def test_agents_registry():
    assert set(AGENTS) == {"sample", "sample_v2"}

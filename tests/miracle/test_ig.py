"""信息增益测试（要求 4）：严格 KL 计算、缺失原因记录、episode/iteration 聚合落盘。"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentbench_frame.miracle.ig import (
    DecisionPoint,
    compute_kl,
    distribution_from_policy,
    episode_ig,
    save_episode_ig,
    save_iteration_index,
)


# ---------------------------------------------------------------------------
# 严格 KL
# ---------------------------------------------------------------------------

def test_compute_kl_exact():
    # p_new == p_old → KL = 0
    dist = {"a": 0.5, "b": 0.5}
    kl, reason = compute_kl(dist, dist)
    assert reason is None
    assert kl == 0.0
    # 手算：p_new=(0.75, 0.25) vs p_old=(0.5, 0.5)
    kl, reason = compute_kl({"a": 0.75, "b": 0.25}, {"a": 0.5, "b": 0.5})
    assert reason is None
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert math.isclose(kl, expected, rel_tol=1e-9)


def test_compute_kl_normalizes_inputs():
    # 未归一化输入按概率归一化处理（只保留正值）
    kl, reason = compute_kl({"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    assert reason is None
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert math.isclose(kl, expected, rel_tol=1e-9)


def test_compute_kl_support_expansion_is_missing_not_fake():
    """旧策略在动作 a 上概率为 0、新策略为正 → 严格 KL 发散，必须记缺失而非伪造有限值。"""
    kl, reason = compute_kl({"a": 0.5, "b": 0.5}, {"b": 1.0})
    assert kl is None
    assert reason == "support_expansion"


def test_compute_kl_degenerate():
    assert compute_kl({}, {"a": 1.0}) == (None, "degenerate")
    assert compute_kl({"a": 1.0}, {}) == (None, "degenerate")


# ---------------------------------------------------------------------------
# episode 聚合
# ---------------------------------------------------------------------------

def _dp(sig_new, sig_old, obs_hash="h1", round_=0):
    return DecisionPoint(
        obs_hash=obs_hash,
        round=round_,
        support=("summon", "move", "endround"),
        p_new=sig_new,
        p_old=sig_old,
    )


def test_episode_ig_aggregation_and_missing():
    decisions = [
        _dp({"a": 1.0}, {"a": 1.0}, obs_hash="h1"),          # KL=0
        _dp({"a": 0.75, "b": 0.25}, {"a": 0.5, "b": 0.5}, obs_hash="h2"),  # 有限 KL
        _dp({"a": 0.5, "b": 0.5}, {"b": 1.0}, obs_hash="h3"),  # 支撑集外扩
        _dp({"a": 1.0}, {"a": 1.0}, obs_hash="h4"),          # KL=0
    ]
    out = episode_ig(decisions, episode_id="ep1", iteration=2, mismatched=1)
    assert out["episode_id"] == "ep1"
    assert out["iteration"] == 2
    assert out["n_decisions"] == 4  # 4 个决策点；mismatched 计入缺失统计
    expected_kl = round(
        (
            0.0
            + (0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5))
            + 0.0
        ) / 3,
        6,  # episode_kl 落盘前 round 到 6 位
    )
    assert math.isclose(out["episode_kl"], expected_kl, rel_tol=1e-9)
    assert out["missing"] == {"support_expansion": 1, "state_mismatch": 1, "degenerate": 0}
    assert math.isclose(out["coverable_ratio"], 3 / 4)
    # 逐决策点明细落盘字段完整
    row3 = [r for r in out["decisions"] if r["obs_hash"] == "h3"][0]
    assert row3["kl_new_vs_old"] is None
    assert row3["missing_reason"] == "support_expansion"


def test_episode_ig_all_missing():
    out = episode_ig(
        [_dp({"a": 1.0}, {"b": 1.0})],  # 支撑集完全不相交
        episode_id="ep0", iteration=0,
    )
    assert out["episode_kl"] is None
    assert out["coverable_ratio"] == 0.0
    assert out["missing"]["support_expansion"] == 1


# ---------------------------------------------------------------------------
# 策略分布辅助 + 落盘
# ---------------------------------------------------------------------------

def test_distribution_from_policy():
    mask = [{"operation_type": "summon", "operation_parameters": {}},
            {"operation_type": "endround", "operation_parameters": {}}]

    def policy(obs, actions):
        return {"summon:1": 0.6, "endround": 0.6}  # 故意不归一化
    dist = distribution_from_policy(policy, {}, mask)
    assert math.isclose(sum(dist.values()), 1.0)
    assert math.isclose(dist["summon:1"], 0.5)


def test_save_and_reload(tmp_path):
    out = episode_ig(
        [_dp({"a": 1.0}, {"a": 1.0})],
        episode_id="ep_x", iteration=0,
    )
    path = tmp_path / "ig" / "24_miracle" / "0" / "ep_x.json"
    save_episode_ig(out, path)
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["episode_id"] == "ep_x"
    assert reloaded["episode_kl"] == 0.0

    index = {
        "game": "24_miracle",
        "iterations": [{"iteration": 0, "episodes": ["ep_x"],
                        "mean_ig": 0.0, "missing": {"support_expansion": 0}}],
    }
    ipath = tmp_path / "ig" / "24_miracle" / "ig_index.json"
    save_iteration_index(index, ipath)
    assert json.loads(ipath.read_text(encoding="utf-8"))["iterations"][0]["iteration"] == 0

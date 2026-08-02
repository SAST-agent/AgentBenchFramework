"""决策空间测试（要求 3）：obs schema、macro-action、action mask 与官方仲裁一致性。

测试进程内直接 import 官方 ``StateSystem``/``Parser`` 作为仲裁器：
``action_mask`` 产出的每个候选动作必须被官方 ``check_legality`` 接受；
构造的非法动作必须被官方拒绝且不在 mask 中。
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "src" / "agentbench_frame" / "miracle" / "official_logic"
sys.path.insert(0, str(_OFFICIAL))
sys.path.insert(0, str(_ROOT / "src"))

from StateSystem.StateSystem import StateSystem
from PlayerLegality.player_legality import Parser

from agentbench_frame.miracle.decision_space import (
    ABYSS,
    ACTION_TYPES,
    MAPBORDER,
    MAX_ROUND,
    MIRACLE_POS,
    MIRACLE_SUMMON_POS,
    PLAYER,
    TERMINATION,
    UNIT,
    Action,
    action_mask,
    cube_distance,
    encode_action,
    support_set,
    termination_reason,
)


def _fresh(deck_creatures=("Archer", "Swordsman", "BlackBat")):
    """官方 StateSystem + Parser，双方 Init 后 camp0 高 mana。"""
    st = StateSystem()
    p = Parser(st)
    p.parse(json.dumps({
        "player": 0, "round": 0, "operation_type": "init",
        "operation_parameters": {"artifacts": ["HolyLight"], "creatures": list(deck_creatures)},
    }))
    p.parse(json.dumps({
        "player": 1, "round": 0, "operation_type": "init",
        "operation_parameters": {"artifacts": ["HolyLight"], "creatures": list(deck_creatures)},
    }))
    p.set_round(0)
    st.player_list[0].mana = 100
    st.player_list[1].mana = 100
    return st, p


def _summon(p, player, type_, pos):
    p.parse(json.dumps({
        "player": player, "round": 0, "operation_type": "summon",
        "operation_parameters": {"type": type_, "level": 1, "position": pos},
    }))


def _official_legality(p, player, action):
    op = p.to_object({
        "player": str(player),
        "operation_type": action.type,
        "operation_parameters": action.params,
    })
    return op.check_legality()


# ---------------------------------------------------------------------------
# 1. Obs schema
# ---------------------------------------------------------------------------

def test_obs_schema_indices_match_official_parse():
    st, p = _fresh()
    _summon(p, 0, "Swordsman", [-8, 6, 2])
    _summon(p, 0, "BlackBat", [-7, 6, 1])
    _summon(p, 1, "Swordsman", [6, -8, 2])
    obs = st.parse()

    # units 行：18 字段，索引解读自洽
    for u in obs["map"]["units"]:
        assert len(u) == 18
        assert isinstance(u[UNIT["ID"]], int)
        assert u[UNIT["CAMP"]] in (0, 1)
        assert isinstance(u[UNIT["POS"]], (list, tuple)) and len(u[UNIT["POS"]]) == 3
        assert isinstance(u[UNIT["FLYING"]], int)
        assert u[UNIT["CAN_ATK"]] in (0, 1)
        assert u[UNIT["CAN_MOVE"]] in (0, 1)
    # 飞行单位 BlackBat 的 flying 字段=1（全局编号 UNIT_NAME_PARSED: BlackBat=2）
    bat = [u for u in obs["map"]["units"] if u[UNIT["TYPE"]] == 2][0]
    assert bat[UNIT["FLYING"]] == 1
    # players 行：5 字段，capacities 为 [[type_idx, limit, [ids]], ...]
    for camp in (0, 1):
        assert len(obs["players"][camp]) == 5
        caps = obs["players"][camp][PLAYER["CAPACITIES"]]
        assert len(caps) == 3
        for cap in caps:
            assert len(cap) == 3 and isinstance(cap[0], int)
    # barracks 4 个
    assert len(obs["map"]["barracks"]) == 4


def test_obs_from_real_trace_matches_decision_space():
    """用真实对局 trace 的 obs 验证 schema（样本冒烟产物存在时）。"""
    import glob
    traces = sorted(glob.glob("agentbench_data/replays/24_miracle/*.trace.jsonl"))
    if not traces:
        pytest.skip("无本地 trace 样本")
    for line in open(traces[-1]):
        row = json.loads(line)
        if row["kind"] != "from_logic" or row["state"] in (0, -1):
            continue
        content = row["payload"]["content"][0]
        msg = json.loads(content[6:])
        if "round" not in msg or "map" not in msg:
            continue
        obs = msg
        for u in obs["map"]["units"]:
            assert len(u) == 18
        assert len(obs["players"]) == 2
        assert all(len(p) == 5 for p in obs["players"])
        break


# ---------------------------------------------------------------------------
# 2. action mask ↔ 官方仲裁
# ---------------------------------------------------------------------------

def test_mask_actions_all_accepted_by_official():
    st, p = _fresh()
    for t, pos in [("Swordsman", [-8, 6, 2]), ("Archer", [-7, 6, 1]), ("BlackBat", [-6, 6, 0])]:
        _summon(p, 0, t, pos)
    _summon(p, 1, "Swordsman", [6, -8, 2])
    # 敌方单位挪到 camp0 Archer 射程 [3,4] 内的非障碍格，产生 attack 候选
    foe = [u for u in st.map.unit_list if u.camp == 1][0]
    foe.pos = (-4, 6, -2)
    p.set_round(1)
    obs = st.parse()
    obs["camp"] = 0

    mask = action_mask(obs, camp=0)
    assert {"summon", "move", "attack", "endround"} <= {a.type for a in mask}
    rejected = [a for a in mask if _official_legality(p, 0, a) is not True]
    assert rejected == [], f"官方拒绝的 mask 候选: {[a.signature() for a in rejected]}"


def test_mask_excludes_official_illegal():
    st, p = _fresh()
    _summon(p, 0, "Swordsman", [-8, 6, 2])
    p.set_round(1)
    obs = st.parse()
    obs["camp"] = 0
    mask = action_mask(obs, camp=0)
    signatures = {a.signature() for a in mask}
    # 单位 id 由官方模块级 UNIT_ID 计数器分配（测试间会漂移），动态取
    sw_id = [u for u in obs["map"]["units"] if u[UNIT["CAMP"]] == 0][0][UNIT["ID"]]

    illegal = [
        Action("summon", {"type": "Swordsman", "level": 1, "position": [99, 0, -99]}),
        Action("summon", {"type": "Swordsman", "level": 1, "position": [-8, 6, 2]}),  # 已占
        Action("move", {"mover": sw_id, "position": [5, 5, -10]}),  # 超距且跨障碍
        Action("attack", {"attacker": sw_id, "target": 1}),          # 神迹超射程
        Action("summon", {"type": "Witch", "level": 1, "position": [-8, 6, 2]}),  # 类型不存在
    ]
    for a in illegal:
        legality = _official_legality(p, 0, a)
        assert legality is not True, f"非法动作被官方接受: {a.signature()}"
        assert a.signature() not in signatures, f"非法动作出现在 mask: {a.signature()}"


def test_flying_ground_may_stack():
    """官方允许飞行/地面单位同格（get_unit_at 按 flying 过滤），mask 应放行。"""
    st, p = _fresh()
    _summon(p, 0, "Swordsman", [-8, 6, 2])  # 地面
    p.set_round(1)
    obs = st.parse()
    obs["camp"] = 0
    stack = Action("summon", {"type": "BlackBat", "level": 1, "position": [-8, 6, 2]})
    assert _official_legality(p, 0, stack) is True
    assert stack.signature() in {a.signature() for a in action_mask(obs, camp=0)}


def test_walkable_neighbors_respect_obstacles_and_border():
    """move 候选不落在 MAPBORDER / Abyss / Miracle 上（与官方 search_path 一致）。"""
    st, p = _fresh()
    _summon(p, 0, "Swordsman", [-8, 6, 2])
    p.set_round(1)
    obs = st.parse()
    obs["camp"] = 0
    for a in action_mask(obs, camp=0):
        if a.type != "move":
            continue
        pos = tuple(a.params["position"])
        assert pos not in MAPBORDER
        assert pos not in [tuple(m) for m in MIRACLE_POS.values()]
        assert pos not in ABYSS  # Swordsman 是地面单位


# ---------------------------------------------------------------------------
# 3. 常量 / 终止条件 / 编码
# ---------------------------------------------------------------------------

def test_constants_match_official():
    assert MAX_ROUND == 100
    assert len(ABYSS) == 39
    assert len(MAPBORDER) == 66
    assert len(MIRACLE_SUMMON_POS[0]) == 5 and len(MIRACLE_SUMMON_POS[1]) == 5
    assert cube_distance((0, 0, 0), (3, -3, 0)) == 3
    # 终止条件齐备
    for key in ("max_round", "miracle_destroyed", "ai_timeout", "winner_rule"):
        assert key in TERMINATION


def test_termination_reason_mapping():
    assert termination_reason(terminated_by="normal", rounds=99, scores=(0, 1), errors=[]) == "max_round"
    assert termination_reason(terminated_by="normal", rounds=44, scores=(30000, 0), errors=[]) == "miracle_destroyed"
    assert termination_reason(terminated_by="timeout", rounds=10, scores=(0, 1), errors=["stuck x3"]) == "ai_timeout"
    assert termination_reason(terminated_by="logic_exit", rounds=3, scores=(0, 0), errors=["x"]) == "abnormal(logic_exit)"


def test_action_encoding_and_signature():
    a = Action("summon", {"type": "Swordsman", "level": 1, "position": [-8, 6, 2]})
    d = a.to_dict()
    assert d["operation_type"] == "summon"
    assert d["operation_parameters"] == {"type": "Swordsman", "level": 1, "position": [-8, 6, 2]}
    assert a.signature() == a.signature()
    assert Action("move", {"mover": 1, "position": [0, 0, 0]}).signature() != a.signature()
    enc = encode_action("endround")
    assert enc == {"operation_type": "endround", "operation_parameters": {}}
    # 动作类型都是官方 Parser 支持的
    official_types = {"init", "summon", "move", "attack", "endround",
                      "startround", "forbid", "select", "use", "surrender"}
    assert set(ACTION_TYPES) <= official_types


def test_support_set_and_mask_stable():
    st, p = _fresh()
    p.set_round(1)
    obs = st.parse()
    obs["camp"] = 0
    s1 = support_set(obs, camp=0)
    m1 = {a.signature() for a in action_mask(obs, camp=0)}
    m2 = {a.signature() for a in action_mask(obs, camp=0)}
    assert m1 == m2  # 确定性
    assert s1 == {a.type for a in action_mask(obs, camp=0)}
    assert "endround" in s1

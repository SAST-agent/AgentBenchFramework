"""决策空间定义（要求 3）：observation schema、macro-action 枚举与编码、
action mask、终止条件、信息增益用的动作支持集。

口径说明
--------
- 官方逻辑是**最终仲裁**：本模块的 ``action_mask`` 基于 obs 可见信息做规则过滤
  （近似超集），个别场景（如障碍阻挡弹道/寻路）无法从 obs 完全判定，以官方
  ``Parser.check_legality`` 的接受/拒绝为准。
- 动作统一编码为官方操作 dict：``{"operation_type", "operation_parameters"}``，
  与 ``host.py`` 的 ``_send_action`` 输入一致；结构化编码见 ``encode_action``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "MAX_ROUND",
    "MIRACLE_POS",
    "MIRACLE_SUMMON_POS",
    "BARRACKS",
    "UNIT",
    "PLAYER",
    "TERMINATION",
    "ACTION_TYPES",
    "SUMMON_TYPES",
    "cube_distance",
    "encode_action",
    "action_mask",
    "support_set",
    "termination_reason",
    "Action",
]

# ---------------------------------------------------------------------------
# 常量（与官方 Data.json / StateSystem.py 一致）
# ---------------------------------------------------------------------------

#: 回合数上限（官方 MAX_ROUND）
MAX_ROUND = 100

#: 双方神迹位置
MIRACLE_POS = {0: (-7, 7, 0), 1: (7, -7, 0)}

#: 双方神迹召唤点（官方 StateSystem.miracle_list）
MIRACLE_SUMMON_POS: dict[int, list] = {
    0: [(-8, 6, 2), (-7, 6, 1), (-6, 6, 0), (-6, 7, -1), (-6, 8, -2)],
    1: [(8, -6, -2), (7, -6, -1), (6, -6, 0), (6, -7, 1), (6, -8, 2)],
}

#: 4 个固定驻扎点：[(pos, 初始 camp=-1, [3 个召唤点])]
BARRACKS = [
    ((-6, -6, 12), -1, [(-7, -5, 12), (-5, -7, 12), (-5, -6, 11)]),
    ((6, 6, -12), -1, [(7, 5, -12), (5, 7, -12), (5, 6, -11)]),
    ((0, -5, 5), -1, [(0, -4, 4), (-1, -4, 5), (-1, -5, 6)]),
    ((0, 5, -5), -1, [(0, 4, -4), (1, 4, -5), (1, 5, -6)]),
]

#: 官方 ABYSS_INIT_LIST（地面单位不可通过，飞行单位可）——逐项与官方 Obstacle.py 核对
ABYSS = [
    (0, 0, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0),
    (-2, -1, 3), (-1, -2, 3), (-2, -2, 4), (-3, -2, 5),
    (-4, -4, 8), (-5, -4, 9), (-4, -5, 9), (-5, -5, 10), (-6, -5, 11),
    (1, 2, -3), (2, 1, -3), (2, 2, -4), (3, 2, -5),
    (4, 4, -8), (5, 4, -9), (4, 5, -9), (5, 5, -10), (6, 5, -11),
    (5, 8, -13), (6, 7, -13), (7, 6, -13), (8, 5, -13), (6, 8, -14), (7, 7, -14), (8, 6, -14),
    (-5, -8, 13), (-6, -7, 13), (-7, -6, 13), (-8, -5, 13), (-6, -8, 14), (-7, -7, 14), (-8, -6, 14),
]

#: 官方 MAPBORDER（地图边界，任何单位不可出界）
MAPBORDER = (
    [(-6 + i, -9, 15 - i) for i in range(14)]
    + [(9, 6 - i, -15 + i) for i in range(14)]
    + [(-9, -6 + i, 15 - i) for i in range(14)]
    + [(6 - i, 9, -15 + i) for i in range(14)]
    + [(-7, -8, 15), (-8, -7, 15), (8, 7, -15), (7, 8, -15)]
    + [(7, -8, 1), (8, -8, 0), (8, -7, -1)]
    + [(-8, 7, 1), (-8, 8, 0), (-7, 8, -1)]
)

#: 神迹本体也是障碍（官方 main.py：Obstacle("Miracle", ...)）
MIRACLE_OBSTACLES = list(MIRACLE_POS.values())

#: 单位类型（官方 UNIT_DATA keys，卡组可选范围）
UNIT_TYPES = [
    "Archer", "Swordsman", "BlackBat", "Priest",
    "VolcanoDragon", "Inferno", "FrostDragon",
]

#: 默认卡组生物顺序（样例 AI 卡组，obs 的 capacities 按此顺序编号）
SUMMON_TYPES = ["Archer", "Swordsman", "BlackBat"]

#: 官方 Data.json 单位属性（决策空间自包含，进程内不再依赖官方目录）
UNIT_STATS = {
    "Archer": {"flying": False, "cost": [2, 4, 6], "max_move": [3, 3, 3]},
    "Swordsman": {"flying": False, "cost": [2, 4, 6], "max_move": [3, 3, 3]},
    "BlackBat": {"flying": True, "cost": [2, 3, 6], "max_move": [4, 4, 5]},
    "Priest": {"flying": False, "cost": [2, 4, 7], "max_move": [5, 5, 5]},
    "VolcanoDragon": {"flying": False, "cost": [5, 7, 9], "max_move": [2, 2, 2]},
    "Inferno": {"flying": False, "cost": [0], "max_move": [3]},
    "FrostDragon": {"flying": False, "cost": [5, 7, 9], "max_move": [2, 2, 2]},
}

#: 终止条件定义（官方规则）
TERMINATION = {
    "max_round": MAX_ROUND,          # 打满 100 回合
    "miracle_destroyed": "神迹 HP ≤ 0 或一方无单位（官方 score 记 30000）",
    "ai_timeout": "任一玩家超时/异常（官方发 error 终局帧）",
    "winner_rule": "平局时后手 +1 分，winner 为分数更高者",
}

# ---------------------------------------------------------------------------
# Obs schema（官方 statesystem.parse() 输出，字段索引与含义）
# ---------------------------------------------------------------------------

#: map.units 每行 18 个字段
UNIT = {
    "ID": 0, "CAMP": 1, "TYPE": 2, "COST": 3, "ATK": 4,
    "MAX_HP": 5, "HP": 6, "ATK_RANGE": 7, "MAX_MOVE": 8, "COOL_DOWN": 9,
    "POS": 10, "LEVEL": 11, "FLYING": 12, "ATK_FLYING": 13,
    "AGILITY": 14, "HOLY_SHIELD": 15, "CAN_ATK": 16, "CAN_MOVE": 17,
}

#: players[camp] 每项 5 个字段
PLAYER = {
    "ARTIFACTS": 0, "MANA": 1, "MAX_MANA": 2, "CAPACITIES": 3,
    "NEWLY_SUMMONED": 4,
}

#: macro-action 类型（官方 Parser 接受的 operation_type）
ACTION_TYPES = {
    "init": {"round": 0, "artifacts": "list[str]", "creatures": "list[str]"},
    "summon": {"type": "str", "level": "int 1..3", "position": "[x,y,z]"},
    "move": {"mover": "unit_id", "position": "[x,y,z]"},
    "attack": {"attacker": "unit_id", "target": "unit_id（敌方单位或神迹 id=camp）"},
    "endround": {},
}


@dataclass(frozen=True)
class Action:
    """一个结构化的 macro-action（可 hash，供信息增益统计）。"""

    type: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"operation_type": self.type, "operation_parameters": dict(self.params)}

    def signature(self) -> str:
        """规范化签名：同一动作的不同表示映射到同一字符串。"""
        return json.dumps(
            {"operation_type": self.type, "operation_parameters": self.params},
            sort_keys=True,
            ensure_ascii=False,
        )


def cube_distance(a, b) -> int:
    """六边形立方坐标距离（官方 Geometry.calculator.cube_distance 同式）。"""
    return (
        abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
    ) // 2


def encode_action(action_type: str, **params) -> dict:
    """把结构化动作编码为官方操作 dict。"""
    return {"operation_type": action_type, "operation_parameters": params}


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

def _own_units(obs: dict, camp: int) -> list:
    return [u for u in obs.get("map", {}).get("units", []) if u[UNIT["CAMP"]] == camp]


def _foe_units(obs: dict, camp: int) -> list:
    return [u for u in obs.get("map", {}).get("units", []) if u[UNIT["CAMP"]] != camp]


def _occupied_by(obs: dict, flying: bool) -> set:
    """按 flying 维度返回被占用的位置（官方 get_unit_at 同口径：飞行/地面互不阻挡）。"""
    return {
        tuple(u[UNIT["POS"]])
        for u in obs.get("map", {}).get("units", [])
        if bool(u[UNIT["FLYING"]]) == flying
    }


def _summon_positions(obs: dict, camp: int) -> list:
    """神迹召唤点 + 已占领 barrack 召唤点。"""
    result = list(MIRACLE_SUMMON_POS[camp])
    barracks = obs.get("map", {}).get("barracks", [])
    for barrack, bcamp in zip([b[0] for b in BARRACKS], barracks):
        if bcamp == camp:
            info = next(b for b in BARRACKS if b[0] == barrack)
            result += info[2]
    return result


def action_mask(obs: dict, camp: Optional[int] = None,
                deck: Optional[list] = None) -> list[Action]:
    """返回 ``camp`` 在当前 obs 下的合法 macro-action 集合（近似超集）。

    - ``deck``：卡组生物顺序（obs 的 capacities 按卡组顺序编号），默认
      ``SUMMON_TYPES``（样例 AI 卡组）。
    - 返回 ``list[Action]``，每个 Action 可直接 ``to_dict()`` 发给官方逻辑。
    """
    camp = int(obs.get("camp", 0)) if camp is None else camp
    deck = list(deck) if deck else list(SUMMON_TYPES)
    occupied_ground = _occupied_by(obs, flying=False)
    occupied_sky = _occupied_by(obs, flying=True)
    actions: list[Action] = []

    # endround：恒合法
    actions.append(Action("endround"))

    # summon：容量未满 + mana 够 + 召唤点空位（按 flying 维度）
    players = obs.get("players", [])
    mana = players[camp][PLAYER["MANA"]] if len(players) > camp else 0
    capacities = players[camp][PLAYER["CAPACITIES"]] if len(players) > camp else []
    # capacities: [[type_idx, limit, [已召唤 unit ids]], ...]
    for type_idx, type_name in enumerate(deck):
        cap = capacities[type_idx] if type_idx < len(capacities) else None
        if cap is None or len(cap[2]) >= cap[1]:
            continue
        stats = UNIT_STATS.get(type_name)
        if stats is None:
            continue
        cost = stats["cost"][0]
        if mana < cost:
            continue
        occupied = occupied_sky if stats["flying"] else occupied_ground
        for pos in _summon_positions(obs, camp):
            if tuple(pos) not in occupied:
                actions.append(
                    Action("summon", {"type": type_name, "level": 1, "position": list(pos)})
                )

    # move：每个己方 can_move 单位 → 一步可达且未被占用（按 flying 维度）的位置（近似）
    for u in _own_units(obs, camp):
        if not u[UNIT["CAN_MOVE"]]:
            continue
        pos = tuple(u[UNIT["POS"]])
        flying = bool(u[UNIT["FLYING"]])
        occupied = occupied_sky if flying else occupied_ground
        for nxt in _walkable_neighbors(pos, flying=flying):
            if nxt in occupied:
                continue
            actions.append(Action("move", {"mover": u[UNIT["ID"]], "position": list(nxt)}))

    # attack：己方 can_atk 单位 × 射程内目标（敌方单位 hp>0 / 敌方神迹）
    miracles = obs.get("map", {}).get("miracles", [])
    foe_miracle_id = 1 - camp
    foe_miracle_hp = miracles[foe_miracle_id] if len(miracles) > foe_miracle_id else 0
    for u in _own_units(obs, camp):
        if not u[UNIT["CAN_ATK"]]:
            continue
        lo, hi = u[UNIT["ATK_RANGE"]]
        pos = tuple(u[UNIT["POS"]])
        for f in _foe_units(obs, camp):
            if f[UNIT["HP"]] > 0 and lo <= cube_distance(pos, tuple(f[UNIT["POS"]])) <= hi:
                actions.append(Action("attack", {"attacker": u[UNIT["ID"]], "target": f[UNIT["ID"]]}))
        if foe_miracle_hp > 0 and lo <= cube_distance(pos, MIRACLE_POS[foe_miracle_id]) <= hi:
            actions.append(Action("attack", {"attacker": u[UNIT["ID"]], "target": foe_miracle_id}))

    return actions


def support_set(obs: dict, camp: Optional[int] = None,
                deck: Optional[list] = None) -> set[str]:
    """动作支持集（信息增益用）：当前 obs 下可选 macro-action 类型。"""
    return {a.type for a in action_mask(obs, camp=camp, deck=deck)}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _walkable_neighbors(pos, flying: bool = False) -> list:
    """一步可达且不是障碍/边界的位置（与官方 search_path 的 obstacles 一致）。"""
    result = []
    for d in [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]:
        nxt = tuple(pos[i] + d[i] for i in range(3))
        if nxt in MAPBORDER:
            continue
        if nxt in MIRACLE_OBSTACLES:
            continue
        if (not flying) and nxt in ABYSS:
            continue
        result.append(nxt)
    return result


def _cost_of(type_name: str, level: int) -> int:
    """单位召唤费用（官方 Data.json）。"""
    return UNIT_STATS[type_name]["cost"][level - 1]


def termination_reason(*, terminated_by: str, rounds: int, scores: tuple,
                       errors: list) -> str:
    """把对局结果映射为终止原因（要求 3 的终止条件判定）。"""
    if terminated_by == "timeout" or any("stuck" in e for e in errors):
        return "ai_timeout"
    if terminated_by in ("idle_timeout", "logic_exit", "host_error"):
        return f"abnormal({terminated_by})"
    if max(scores or (0, 0)) >= 30000:
        return "miracle_destroyed"
    if rounds >= MAX_ROUND - 1:
        return "max_round"
    return "unknown"

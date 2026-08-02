"""MiracleAgent 接口与内置策略（自己实现）。

Agent 与评测机（``host.py``）的接口约定：
- ``choose_cards(camp) -> {"artifacts": [...], "creatures": [...]}``：开局的卡组选择，
  返回的神器/生物名必须是官方 ``Data.json`` 中的合法名称。
- ``act(obs) -> {"operation_type": ..., "operation_parameters": {...}}``：回合内决策。
  ``obs`` 是官方内层消息（``statesystem.parse()`` + ``round`` + ``camp``）：
  ``{"map": {...}, "players": [...], "camp": int, "round": int}``。
  ``player``/``round`` 由评测机补全，Agent 不必携带。

obs 关键字段（官方 ``parse`` 输出，字段含义见 ``decision_space.py``）：
- ``map.units`` 每行 18 个元素：
  [id, camp, type, cost, atk, max_hp, hp, atk_range, max_move, cool_down,
   pos[x,y,z], level, flying, atk_flying, agility, holy_shield, can_atk, can_move]
- ``map.miracles``: [hp0, hp1]
- ``map.barracks``: [camp]（4 个固定驻扎点）
- ``players[camp]``: [artifacts_parsed, mana, max_mana, capacities_parsed, newly_summoned]
- ``camp``: 本玩家阵营 0/1；``round``: 当前回合号
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from typing import Optional

__all__ = [
    "MiracleAgent",
    "EndRoundAgent",
    "SampleAgent",
    "DEFAULT_ARTIFACTS",
    "DEFAULT_CREATURES",
    "MIRACLE_POS",
    "SUMMON_POS",
]

#: 初始卡组（1 神器 + 3 生物，均为官方合法名称）
DEFAULT_ARTIFACTS = ["HolyLight"]
DEFAULT_CREATURES = ["Archer", "Swordsman", "BlackBat"]

#: 双方神迹位置（官方 StateSystem 固定）：camp0 左上，camp1 右下
MIRACLE_POS = {0: (-7, 7, 0), 1: (7, -7, 0)}

#: 双方神迹召唤点（官方 StateSystem.miracle_list 的第 1 个召唤点）
SUMMON_POS = {0: (-8, 6, 2), 1: (8, -6, -2)}

#: 单位 level1 的召唤费用（官方 Data.json UnitData.cost[0]）
COST_LEVEL1 = {
    "Archer": 2, "Swordsman": 2, "BlackBat": 2, "Priest": 3,
    "VolcanoDragon": 6, "Inferno": 4, "FrostDragon": 7,
}


def cube_distance(a, b) -> int:
    """六边形立方坐标距离（官方 Geometry.calculator.cube_distance 同式）。"""
    return (
        abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
    ) // 2


class MiracleAgent(ABC):
    """对局策略接口。子类只需实现 ``choose_cards`` 与 ``act``。"""

    name: str = "MiracleAgent"

    @abstractmethod
    def choose_cards(self, camp: int) -> dict:
        """开局选卡，返回 ``{"artifacts": [1 个], "creatures": [3 个]}``。"""

    @abstractmethod
    def act(self, obs: dict) -> dict:
        """给定局面 obs 返回一个回合内操作。"""


class EndRoundAgent(MiracleAgent):
    """每回合直接 endround，用于协议/链路冒烟测试。"""

    name = "endround"

    def choose_cards(self, camp: int) -> dict:
        return {
            "artifacts": list(DEFAULT_ARTIFACTS),
            "creatures": list(DEFAULT_CREATURES),
        }

    def act(self, obs: dict) -> dict:
        return {"operation_type": "endround", "operation_parameters": {}}


class SampleAgent(MiracleAgent):
    """自写的规则策略（链路冒烟/评测基线，非强策略）：

    1. 能打则打：射程内优先敌方单位（hp>0），其次敌方神迹；
    2. 法力够就补单位（Swordsman lv1，放在神迹召唤点）；
    3. 还有能动的单位就向敌方神迹方向走一步；
    4. 否则 endround。
    """

    name = "sample"

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        #: 上次收到的 obs 指纹；相同 ⇒ 上一操作被官方拒绝（obs 未推进）
        self._last_obs_key: Optional[str] = None

    def choose_cards(self, camp: int) -> dict:
        return {
            "artifacts": list(DEFAULT_ARTIFACTS),
            "creatures": list(DEFAULT_CREATURES),
        }

    def act(self, obs: dict) -> dict:
        camp = int(obs.get("camp", 0))
        units = obs.get("map", {}).get("units", [])

        # 被拒感知：obs 与上次完全相同 ⇒ 官方拒绝了上一操作，本回合直接 endround，
        # 避免对同一局面无限重试同一操作（如弹道被障碍挡住、目标被挡等）。
        key = hashlib.md5(
            json.dumps(obs, sort_keys=True).encode()
        ).hexdigest()
        rejected = key == self._last_obs_key
        self._last_obs_key = key
        if rejected:
            return self._end()

        mine = [u for u in units if u[1] == camp]
        foes = [u for u in units if u[1] != camp]

        # 1) 攻击
        for u in mine:
            if not u[16]:  # can_atk
                continue
            lo, hi = u[7]
            pos = u[10]
            for f in foes:
                if f[6] > 0 and lo <= cube_distance(pos, f[10]) <= hi:
                    return self._attack(u[0], f[0])
            mpos = MIRACLE_POS[1 - camp]
            if lo <= cube_distance(pos, mpos) <= hi:
                return self._attack(u[0], 1 - camp)  # 神迹 id == camp

        # 2) 召唤：按卡组顺序选第一个容量未满且法力够的生物（卡组=Archer,Swordsman,BlackBat）
        players = obs.get("players", [])
        mana = players[camp][1] if len(players) > camp else 0
        capacities = players[camp][3] if len(players) > camp else []
        # capacities: [[type_index, capacity, [已召唤unit id...]], ...]
        for ti, (type_name, _cost) in enumerate(
            [("Archer", 2), ("Swordsman", 2), ("BlackBat", 2)]
        ):
            cap = capacities[ti] if ti < len(capacities) else None
            used = len(cap[2]) if cap else 0
            limit = cap[1] if cap else 0
            if cap is not None and used >= limit:
                continue
            if mana < COST_LEVEL1[type_name]:
                continue
            spos = SUMMON_POS[camp]
            if not any(cube_distance(u[10], spos) == 0 for u in units):
                return {
                    "operation_type": "summon",
                    "operation_parameters": {
                        "type": type_name,
                        "level": 1,
                        "position": list(spos),
                    },
                }

        # 3) 移动（向敌方神迹走一步，跳过被占位置）
        mpos = MIRACLE_POS[1 - camp]
        occupied = {tuple(u[10]) for u in units}
        for u in mine:
            if u[17]:  # can_move
                nxt = self._step_toward(u[10], mpos, occupied)
                if nxt is not None:
                    return {
                        "operation_type": "move",
                        "operation_parameters": {"mover": u[0], "position": nxt},
                    }

        return self._end()

    def _step_toward(self, pos, target, occupied=None) -> Optional[list]:
        """向目标沿立方坐标最短路走一格（六边形 6 邻居中使距离严格减小的第一个空位）。"""
        # 立方坐标 6 个邻居方向（dx+dy+dz==0）
        neighbors = [
            (1, -1, 0), (1, 0, -1), (0, 1, -1),
            (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
        ]
        occupied = occupied or set()
        d0 = cube_distance(pos, target)
        for n in neighbors:
            nxt = [pos[0] + n[0], pos[1] + n[1], pos[2] + n[2]]
            if cube_distance(nxt, target) >= d0:
                continue
            # 界内（官方地图为半径约 9 的六边形，±10 保守外框）
            if any(abs(c) > 10 for c in nxt):
                continue
            if tuple(nxt) in occupied:
                continue
            return nxt
        return None

    @staticmethod
    def _attack(attacker: int, target: int) -> dict:
        return {
            "operation_type": "attack",
            "operation_parameters": {"attacker": attacker, "target": target},
        }

    @staticmethod
    def _end() -> dict:
        return {"operation_type": "endround", "operation_parameters": {}}

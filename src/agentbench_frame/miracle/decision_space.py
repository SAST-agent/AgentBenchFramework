"""Miracle observation 上的有限 Benchmark 动作支持集。

官方 Logic 保持最终仲裁。WindBlessing 的官方实现允许无限坐标；为使 IG 支持集有限，
本模块只纳入官方地图内坐标，支持集外动作由 IG 记录为缺失。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Action:
    type: str
    params: dict

    def signature(self) -> str:
        return json.dumps(
            {"operation_type": self.type, "operation_parameters": self.params},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )

    def to_dict(self) -> dict:
        return {"operation_type": self.type, "operation_parameters": dict(self.params)}


_DATA = json.loads((Path(__file__).parent / "official_logic" / "Data.json").read_text())
UNIT_NAMES = {value: key for key, value in _DATA["UnitNameParsed"].items()}
ARTIFACT_NAMES = {value: key for key, value in _DATA["ArtifactNameParsed"].items()}
UNIT_DATA = _DATA["UnitData"]

MIRACLE_POS = {0: (-7, 7, 0), 1: (7, -7, 0)}
MIRACLE_SUMMON = {
    0: [(-8, 6, 2), (-7, 6, 1), (-6, 6, 0), (-6, 7, -1), (-6, 8, -2)],
    1: [(8, -6, -2), (7, -6, -1), (6, -6, 0), (6, -7, 1), (6, -8, 2)],
}
BARRACKS = [
    ((-6, -6, 12), [(-7, -5, 12), (-5, -7, 12), (-5, -6, 11)]),
    ((6, 6, -12), [(7, 5, -12), (5, 7, -12), (5, 6, -11)]),
    ((0, -5, 5), [(0, -4, 4), (-1, -4, 5), (-1, -5, 6)]),
    ((0, 5, -5), [(0, 4, -4), (1, 4, -5), (1, 5, -6)]),
]
ABYSS = {
    (0, 0, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1),
    (0, 1, -1), (-1, 1, 0), (-2, -1, 3), (-1, -2, 3), (-2, -2, 4),
    (-3, -2, 5), (-4, -4, 8), (-5, -4, 9), (-4, -5, 9), (-5, -5, 10),
    (-6, -5, 11), (1, 2, -3), (2, 1, -3), (2, 2, -4), (3, 2, -5),
    (4, 4, -8), (5, 4, -9), (4, 5, -9), (5, 5, -10), (6, 5, -11),
    (5, 8, -13), (6, 7, -13), (7, 6, -13), (8, 5, -13), (6, 8, -14),
    (7, 7, -14), (8, 6, -14), (-5, -8, 13), (-6, -7, 13), (-7, -6, 13),
    (-8, -5, 13), (-6, -8, 14), (-7, -7, 14), (-8, -6, 14),
}
DIRECTIONS = ((1, 0, -1), (1, -1, 0), (0, -1, 1),
              (-1, 0, 1), (-1, 1, 0), (0, 1, -1))


def in_map(pos) -> bool:
    x, y, z = pos
    return x + y + z == 0 and -8 <= x <= 8 and -8 <= y <= 8 and -14 <= z <= 14 \
        and tuple(pos) not in MIRACLE_POS.values()


ALL_MAP_POSITIONS = tuple(
    (x, y, -x - y) for x in range(-8, 9) for y in range(-8, 9)
    if in_map((x, y, -x - y))
)


def cube_distance(a, b) -> int:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) // 2


def _reachable(unit, units) -> set[tuple]:
    start = tuple(unit[10])
    flying = bool(unit[12])
    occupied = {tuple(u[10]) for u in units if bool(u[12]) == flying and u[0] != unit[0]}
    blocked = set(MIRACLE_POS.values()) | occupied
    if not flying:
        blocked |= ABYSS
    seen = {start}
    queue = deque([(start, 0)])
    out = set()
    while queue:
        pos, distance = queue.popleft()
        if distance >= int(unit[8]):
            continue
        for delta in DIRECTIONS:
            nxt = tuple(pos[i] + delta[i] for i in range(3))
            if nxt in seen or nxt in blocked or not in_map(nxt):
                continue
            seen.add(nxt)
            out.add(nxt)
            queue.append((nxt, distance + 1))
    return out


def _summon_positions(obs, camp: int) -> list[tuple]:
    positions = list(MIRACLE_SUMMON[camp])
    owners = obs.get("map", {}).get("barracks", [])
    for index, owner in enumerate(owners):
        if owner == camp and index < len(BARRACKS):
            positions.extend(BARRACKS[index][1])
    return positions


def action_mask(obs: dict, camp: int | None = None) -> list[Action]:
    """枚举当前 observation 的有限、带参数动作支持集。"""
    camp = int(obs.get("camp", 0) if camp is None else camp)
    map_data = obs.get("map", {})
    units = map_data.get("units", [])
    players = obs.get("players", [])
    player = players[camp] if len(players) > camp else [[], 0, 0, [], []]
    mana = int(player[1])
    actions: list[Action] = [Action("endround", {}), Action("surrender", {})]

    occupied_by_layer = {(tuple(u[10]), bool(u[12])) for u in units}
    for capacity in player[3]:
        type_name = UNIT_NAMES.get(int(capacity[0]))
        if not type_name or int(capacity[1]) <= 0 or type_name == "Inferno":
            continue
        flying = bool(UNIT_DATA[type_name]["flying"])
        for level, cost in enumerate(UNIT_DATA[type_name]["cost"], start=1):
            if mana < int(cost):
                continue
            for pos in _summon_positions(obs, camp):
                if (tuple(pos), flying) not in occupied_by_layer:
                    actions.append(Action("summon", {
                        "type": type_name, "level": level, "position": list(pos),
                    }))

    mine = [u for u in units if int(u[1]) == camp]
    enemies = [u for u in units if int(u[1]) != camp and int(u[6]) > 0]
    for unit in mine:
        if bool(unit[17]):
            for pos in sorted(_reachable(unit, units)):
                actions.append(Action("move", {"mover": int(unit[0]), "position": list(pos)}))
        if bool(unit[16]) and int(unit[4]) > 0:
            lo, hi = unit[7]
            for target in enemies:
                distance = cube_distance(unit[10], target[10])
                can_hit_air = bool(unit[12]) or bool(unit[13]) or not bool(target[12])
                if int(lo) <= distance <= int(hi) and can_hit_air:
                    actions.append(Action("attack", {
                        "attacker": int(unit[0]), "target": int(target[0]),
                    }))
            enemy_miracle = 1 - camp
            if int(lo) <= cube_distance(unit[10], MIRACLE_POS[enemy_miracle]) <= int(hi):
                actions.append(Action("attack", {
                    "attacker": int(unit[0]), "target": enemy_miracle,
                }))

    for artifact in player[0]:
        artifact_id, name_idx, cost, _, _, state_idx, target_idx, _ = artifact
        if int(state_idx) != 0 or mana < int(cost):
            continue
        name = ARTIFACT_NAMES[int(name_idx)]
        if int(target_idx) == 1:  # Unit；官方仅 SalamanderShield 使用该类型
            for unit in mine:
                actions.append(Action("use", {"card": int(artifact_id), "target": int(unit[0])}))
            continue
        for pos in ALL_MAP_POSITIONS:
            if name == "InfernoFlame":
                owned_barracks = [BARRACKS[i][0] for i, owner in enumerate(map_data.get("barracks", [])) if owner == camp]
                in_range = cube_distance(MIRACLE_POS[camp], pos) <= 7 or any(
                    cube_distance(barrack, pos) <= 5 for barrack in owned_barracks
                )
                ground_occupied = any(tuple(u[10]) == pos and not bool(u[12]) for u in units)
                if not in_range or pos in ABYSS or ground_occupied:
                    continue
            actions.append(Action("use", {"card": int(artifact_id), "target": list(pos)}))

    unique = {action.signature(): action for action in actions}
    return [unique[key] for key in sorted(unique)]


def support_set(obs: dict, camp: int | None = None) -> tuple[str, ...]:
    return tuple(action.signature() for action in action_mask(obs, camp))

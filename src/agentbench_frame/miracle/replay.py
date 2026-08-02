"""回放解析（要求 2 配套）：官方 `.mrc` 二进制 → 事件时间线 / 统计。

格式（与官方 ``gamecode_logic/main.py`` 的 ``get_media_info`` / ``send_media_info``
逐字核对）：
- 头部 7 个 int：``[0, 0, 0, map_type, day_time, 0, 0]``；
- 其后每轮事件流：每个事件编码为 ``(round, event_idx, args...)`` 的 int 序列，
  每轮补齐到 7 的倍数；
- 终局：``(round, 10, winner, 0, 0, 0, 0)``（10 = GameEnd），
  末尾 ``-1`` 结束标记；
- 所有 int 均为 4 字节大端有符号。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["EVENT_NAMES", "CREATURE_NAMES", "ARTIFACT_NAMES", "ReplayEvent",
           "parse_replay", "load_replay_json", "save_replay_json"]

#: 官方事件编号表（event_names，索引即编码）
EVENT_NAMES = ["", "TurnStart", "TurnEnd", "Spawn", "Move", "Attack",
               "Damage", "Death", "Heal", "ActivateArtifact",
               "GameEnd", "GameStart", "BuffAdd", "BuffRemove",
               "Attacking", "Attacked", "Leave", "Arrive", "Summon"]

#: 官方生物编号表（creature_names，索引即编码；camp 用 +10×camp 编码）
CREATURE_NAMES = ["", "Swordsman", "Archer", "BlackBat", "Priest",
                  "VolcanoDragon", "FrostDragon", "Inferno"]

#: 官方神器编号表（artifact_names）
ARTIFACT_NAMES = ["", "HolyLight", "SalamanderShield", "InfernoFlame",
                  "WindBlessing"]

DAMAGE_TYPES = ["", "Attack", "AttackBack", "VolcanoDragonSplash",
                "InfernoFlameActivate"]
BUFF_NAMES = ["BaseBuff", "PriestAtkBuff", "HolyShield", "HolyLightAtkBuff",
              "SalamanderShieldBuff"]


@dataclass
class ReplayEvent:
    """解码后的一条事件：``round``、``type``（事件名）、``args``（参数 int 列表）。"""

    round: int
    type: str
    args: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"round": self.round, "type": self.type, "args": list(self.args)}


def _decode_event(round_: int, idx: int, args: list) -> ReplayEvent:
    name = EVENT_NAMES[idx] if 0 <= idx < len(EVENT_NAMES) else f"UNKNOWN({idx})"
    return ReplayEvent(round=round_, type=name, args=list(args))


def _iter_blocks(ints: list):
    """把 int 流切成事件块。

    官方结构（已用真实 .mrc 验证）：整个 body 是连续的 7-int 块序列；一次
    ``send_media_info`` flush 写一批事件（每个事件 ``(round, idx, args...)``
    连续编码），该批末尾补齐到 7 的倍数。因此：
    - 事件头位置 idx ∈ 1..18 → 按参数数消费（同批事件可能连续，不 pad）；
    - 事件头位置 idx 非法（=0 的 pad 0，或 >18 的异常值）→ 跳过当前 7 边界；
    - 末尾单独 ``-1`` 为结束标记。
    """
    arg_counts = {
        1: 1, 2: 1, 3: 5, 4: 3, 5: 2, 6: 4, 7: 1, 8: 3,
        9: 0, 10: 0, 11: 5, 12: 2, 13: 2, 14: 2, 15: 2, 16: 3, 17: 3, 18: 4,
    }
    i = 0
    n = len(ints)
    cur_round = -1
    while i < n:
        r = ints[i]
        if r == -1 or (i + 1 >= n):
            yield ReplayEvent(r, "END", [])
            i += 1
            continue
        idx = ints[i + 1]
        if idx not in arg_counts or (idx in arg_counts and r < cur_round):
            # pad 区（flush 组尾对齐 7）或 round 回退（pad 0 被误读为事件头）：
            # 跳到下一 7 边界
            nxt = ((i // 7) + 1) * 7
            i = nxt if nxt > i else i + 1
            continue
        if idx == 10:  # GameEnd: [round, 10, winner, 0, 0, 0, 0]
            cur_round = max(cur_round, r)
            yield ReplayEvent(round=r, type="GameEnd", args=[ints[i + 2]])
            i += 7
            continue
        if idx == 9:  # ActivateArtifact: camp, name, 然后 2（HolyLight/InfernoFlame/WindBlessing: t0,t1）或 3（SalamanderShield: 0,0,id）个参数
            argc = 5 if (ints[i + 3] % 10) == 2 else 4
            cur_round = max(cur_round, r)
            yield ReplayEvent(round=r, type="ActivateArtifact", args=ints[i + 2:i + 2 + argc])
            i += 2 + argc
            continue
        count = arg_counts[idx]
        cur_round = max(cur_round, r)
        yield _decode_event(r, idx, ints[i + 2:i + 2 + count])
        i += 2 + count


def parse_replay(path) -> list:
    """解析 `.mrc` 二进制，返回事件时间线（``list[ReplayEvent]``）。"""
    data = Path(path).read_bytes()
    assert len(data) % 4 == 0, f"{path}: 文件长度不是 4 的倍数"
    ints = list(struct.unpack(f">{len(data) // 4}i", data))
    # 去掉头部 7 个 int（[0,0,0,map_type,day_time,0,0]）
    header, body = ints[:7], ints[7:]
    events = list(_iter_blocks(body))
    return events


def summarize(events: list) -> dict:
    """事件统计：各类型计数、终局信息、回合跨度。"""
    counts: dict = {}
    winner = None
    end_round = None
    rounds = set()
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
        if e.round >= 0:
            rounds.add(e.round)
        if e.type == "GameEnd":
            winner = e.args[0] if e.args else None
            end_round = e.round
    return {
        "n_events": len(events),
        "rounds": [min(rounds), max(rounds)] if rounds else [],
        "event_counts": counts,
        "winner": winner,
        "end_round": end_round,
        "ended": any(e.type == "GameEnd" for e in events),
    }


def save_replay_json(events: list, path) -> None:
    """事件时间线落盘 JSON（可与 trace 交叉核对）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def load_replay_json(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [ReplayEvent(**json.loads(line)) for line in f if line.strip()]

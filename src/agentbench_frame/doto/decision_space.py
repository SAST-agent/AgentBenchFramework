"""Typed DOTO observations, continuous joint actions, and explanatory masks."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any


class ActionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def is_noop(self) -> bool:
        return self.x == -1.0 and self.y == -1.0


@dataclass(frozen=True)
class HumanState:
    human_id: int
    x: float
    y: float
    hp: float
    meteor_num: int
    meteor_time: int
    flash_num: int
    flash_time: int
    fireball_time: int
    death_time: int
    inv_time: int


@dataclass(frozen=True)
class InitObservation:
    frame: int
    map_id: int
    faction: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class FrameObservation:
    frame: int
    faction: int
    humans: tuple[HumanState, ...]
    fireballs: tuple
    meteors: tuple
    balls: tuple
    scores: tuple[float, ...]
    bonus: tuple
    map_data: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class DotoAction:
    move: tuple[Point, ...]
    shoot: tuple[Point, ...]
    meteor: tuple[Point, ...]
    flash: tuple[bool, ...]

    def to_json(self) -> dict[str, Any]:
        def points(values: tuple[Point, ...]) -> list[list[float]]:
            return [[point.x, point.y] for point in values]
        return {
            "flag": 0,
            "move": points(self.move),
            "shoot": points(self.shoot),
            "meteor": points(self.meteor),
            "flash": list(self.flash),
        }


@dataclass(frozen=True)
class ActionMask:
    humans: tuple[dict[str, str | None], ...]

    @property
    def valid(self) -> bool:
        return all(reason is None for human in self.humans for reason in human.values())


@dataclass(frozen=True)
class Termination:
    terminated: bool
    reason: str | None


def _list(raw: Any, name: str) -> list:
    value = raw
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ActionValidationError(f"invalid {name}") from exc
    if not isinstance(value, list):
        raise ActionValidationError(f"invalid {name}")
    return value


def parse_observation(
    raw: dict[str, Any], faction: int, map_data: dict[str, Any]
) -> InitObservation | FrameObservation:
    frame = raw.get("frame")
    if not isinstance(frame, int):
        raise ActionValidationError("observation frame must be an integer")
    if faction not in (0, 1):
        raise ActionValidationError("faction must be 0 or 1")
    if frame == 0:
        return InitObservation(frame=0, map_id=int(raw["map"]), faction=faction, raw=raw)
    humans = []
    for row in _list(raw.get("humans", []), "humans"):
        if not isinstance(row, list) or len(row) < 11:
            raise ActionValidationError("each human row must contain 11 values")
        humans.append(HumanState(
            human_id=int(row[0]), x=float(row[1]), y=float(row[2]), hp=float(row[3]),
            meteor_num=int(row[4]), meteor_time=int(row[5]), flash_num=int(row[6]),
            flash_time=int(row[7]), fireball_time=int(row[8]), death_time=int(row[9]),
            inv_time=int(row[10]),
        ))
    return FrameObservation(
        frame=frame,
        faction=faction,
        humans=tuple(humans),
        fireballs=tuple(_list(raw.get("fireballs", []), "fireballs")),
        meteors=tuple(_list(raw.get("meteors", []), "meteors")),
        balls=tuple(tuple(item) for item in _list(raw.get("balls", []), "balls")),
        scores=tuple(float(item) for item in _list(raw.get("scores", []), "scores")),
        bonus=tuple(_list(raw.get("bonus", []), "bonus")),
        map_data=map_data,
        raw=raw,
    )


def _point(raw: Any, field: str) -> Point:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ActionValidationError(f"{field} point must contain x and y")
    if isinstance(raw[0], bool) or isinstance(raw[1], bool):
        raise ActionValidationError(f"{field} coordinates must be numeric")
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise ActionValidationError(f"{field} coordinates must be numeric") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ActionValidationError(f"{field} coordinates must be finite")
    if (x == -1.0) != (y == -1.0):
        raise ActionValidationError(f"{field} no-op must be [-1,-1]")
    return Point(x, y)


def canonicalize_action(raw: dict[str, Any]) -> DotoAction:
    if not isinstance(raw, dict) or raw.get("flag") != 0:
        raise ActionValidationError("action flag must be 0")
    fields: dict[str, tuple[Point, ...]] = {}
    for name in ("move", "shoot", "meteor"):
        values = raw.get(name)
        if not isinstance(values, list) or len(values) != 5:
            raise ActionValidationError(f"{name} must contain 5 points")
        fields[name] = tuple(_point(value, name) for value in values)
    flash = raw.get("flash")
    if not isinstance(flash, list) or len(flash) != 5 or any(type(item) is not bool for item in flash):
        raise ActionValidationError("flash must contain 5 booleans")
    return DotoAction(fields["move"], fields["shoot"], fields["meteor"], tuple(flash))


def _distance(human: HumanState, point: Point) -> float:
    return math.hypot(point.x - human.x, point.y - human.y)


def _destination_reason(obs: FrameObservation, point: Point) -> str | None:
    width, height = float(obs.map_data["width"]), float(obs.map_data["height"])
    if not (0 <= point.x < width and 0 <= point.y < height):
        return "destination_out_of_bounds"
    walls = obs.map_data["walls"]
    if bool(walls[int(point.x)][int(point.y)]):
        return "destination_is_wall"
    return None


def action_mask(obs: FrameObservation, action: DotoAction) -> ActionMask:
    by_id = {human.human_id: human for human in obs.humans}
    held_ids = {int(ball[2]) for ball in obs.balls if len(ball) >= 3 and int(ball[2]) >= 0}
    rows: list[dict[str, str | None]] = []
    for local_index in range(5):
        human_id = local_index * 2 + obs.faction
        human = by_id.get(human_id)
        reasons: dict[str, str | None] = {"move": None, "shoot": None, "meteor": None, "flash": None}
        if human is None:
            rows.append({name: "human_missing" for name in reasons})
            continue
        dead = human.death_time != -1
        move, shoot, meteor, flash = (
            action.move[local_index], action.shoot[local_index],
            action.meteor[local_index], action.flash[local_index],
        )
        if dead:
            if not move.is_noop:
                reasons["move"] = "human_dead"
            if not shoot.is_noop:
                reasons["shoot"] = "human_dead"
            if not meteor.is_noop:
                reasons["meteor"] = "human_dead"
            if flash:
                reasons["flash"] = "human_dead"
            rows.append(reasons)
            continue
        if not move.is_noop:
            reasons["move"] = _destination_reason(obs, move)
            limit = 20.0 if flash else 0.6
            if reasons["move"] is None and _distance(human, move) > limit + 1e-9:
                reasons["move"] = "flash_out_of_range" if flash else "move_out_of_range"
        if not shoot.is_noop:
            reasons["shoot"] = _destination_reason(obs, shoot)
            if human.fireball_time > 0:
                reasons["shoot"] = "fireball_on_cooldown"
            elif shoot.x == human.x and shoot.y == human.y:
                reasons["shoot"] = "shoot_target_is_self"
        if not meteor.is_noop:
            reasons["meteor"] = _destination_reason(obs, meteor)
            if human.meteor_num <= 0:
                reasons["meteor"] = "no_meteor_uses"
            elif human.meteor_time > 0:
                reasons["meteor"] = "meteor_on_cooldown"
            elif _distance(human, meteor) > 30.0 + 1e-9:
                reasons["meteor"] = "meteor_out_of_range"
        if flash:
            if move.is_noop:
                reasons["flash"] = "flash_requires_move_target"
            elif human.flash_num <= 0:
                reasons["flash"] = "no_flash_uses"
            elif human.flash_time > 0:
                reasons["flash"] = "flash_on_cooldown"
            elif human_id in held_ids:
                reasons["flash"] = "carrier_cannot_flash"
            elif reasons["move"] is not None:
                reasons["flash"] = reasons["move"]
        rows.append(reasons)
    return ActionMask(tuple(rows))


def support_pair(old: DotoAction, new: DotoAction) -> tuple[DotoAction, ...]:
    return (old,) if old == new else (old, new)


def termination_from_raw(raw: dict[str, Any]) -> Termination:
    if raw.get("frame") == -1:
        return Termination(True, "official_final_frame")
    return Termination(False, None)

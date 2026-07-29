"""Frozen research protocol primitives for ``24_miracle``.

This module is deliberately side-effect free.  It defines the approved
population split, the 72-case final test matrix, deterministic seed identities,
and the strict command-distribution boundary needed by trajectory KL.  It does
not start the Judge, an AI process, or a benchmark session.
"""

from __future__ import annotations

import hashlib
import json
import random
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from agentbench_frame.eval import ActionCandidate, ActionSupport, PolicyDecision


PROTOCOL_VERSION = "24-miracle-research-v1"
BENCHMARK_VERSION = "24m-frozen-v1"
MANIFEST_SCHEMA_VERSION = "24-miracle-research-manifest-v1"
ACTION_SCHEMA_VERSION = "24-miracle-command-v1"
TRAJECTORY_KL_EPSILON = 0.01
TEST_REPEATS = 3
SEED_MIN = 0
SEED_MAX = 0x7FFF_FFFF

TRAIN_OPPONENTS = (
    "rank01", "rank03", "rank04", "rank06", "rank07",
    "rank08", "rank11", "rank12", "rank14", "rank15",
)
VALIDATION_OPPONENTS = ("rank05", "rank09", "rank13")
TEST_OPPONENTS = ("rank02", "rank10", "rank16")

FROZEN_OPPONENT_DETERMINISM = {
    "rank02": {
        "qualified": False,
        "reason": "compiled ai.cpp seeds rand() from clock()+time(0)",
    },
    "rank10": {
        "qualified": False,
        "reason": "compiled debeat.cpp stops search using process CPU clock",
    },
    "rank16": {
        "qualified": False,
        "reason": (
            "no active RNG call found, but deterministic qualification and "
            "compiler-warning risk have not been closed"
        ),
    },
}

_OPERATION_TYPES = {
    "init", "move", "attack", "summon", "use", "endround", "surrender",
}

_UNIT_NAMES = (
    "Archer", "Swordsman", "BlackBat", "Priest", "VolcanoDragon",
    "Inferno", "FrostDragon",
)
_UNIT_COSTS = (
    (2, 4, 6), (2, 4, 6), (2, 3, 6), (2, 4, 7), (5, 7, 9),
    (0,), (5, 7, 9),
)
_UNIT_FLYING = (False, False, True, False, False, False, False)
_ARTIFACT_NAMES = (
    "HolyLight", "SalamanderShield", "InfernoFlame", "WindBlessing",
)
_MIRACLE_POSITIONS = ((-7, 7, 0), (7, -7, 0))
_MIRACLE_SUMMON_POSITIONS = (
    ((-8, 6, 2), (-7, 6, 1), (-6, 6, 0), (-6, 7, -1), (-6, 8, -2)),
    ((8, -6, -2), (7, -6, -1), (6, -6, 0), (6, -7, 1), (6, -8, 2)),
)
_BARRACK_POSITIONS = (
    (-6, -6, 12), (6, 6, -12), (0, -5, 5), (0, 5, -5),
)
_BARRACK_SUMMON_POSITIONS = (
    ((-7, -5, 12), (-5, -7, 12), (-5, -6, 11)),
    ((7, 5, -12), (5, 7, -12), (5, 6, -11)),
    ((0, -4, 4), (-1, -4, 5), (-1, -5, 6)),
    ((0, 4, -4), (1, 4, -5), (1, 5, -6)),
)
_ABYSS_POSITIONS = (
    (0, 0, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1),
    (0, 1, -1), (-1, 1, 0), (-2, -1, 3), (-1, -2, 3), (-2, -2, 4),
    (-3, -2, 5), (-4, -4, 8), (-5, -4, 9), (-4, -5, 9),
    (-5, -5, 10), (-6, -5, 11), (1, 2, -3), (2, 1, -3), (2, 2, -4),
    (3, 2, -5), (4, 4, -8), (5, 4, -9), (4, 5, -9), (5, 5, -10),
    (6, 5, -11), (5, 8, -13), (6, 7, -13), (7, 6, -13), (8, 5, -13),
    (6, 8, -14), (7, 7, -14), (8, 6, -14), (-5, -8, 13),
    (-6, -7, 13), (-7, -6, 13), (-8, -5, 13), (-6, -8, 14),
    (-7, -7, 14), (-8, -6, 14),
)
_GROUND_OBSTACLES = frozenset((*_ABYSS_POSITIONS, *_MIRACLE_POSITIONS))
_SPECIAL_MOVE_BORDER = frozenset((
    (7, -8, 1), (8, -8, 0), (8, -7, -1),
    (-8, 7, 1), (-8, 8, 0), (-7, 8, -1),
))
_DIRECTIONS = ((1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1),
               (-1, 1, 0), (0, 1, -1))


def _in_map(position: tuple[int, int, int]) -> bool:
    x, y, z = position
    return (
        -8 <= x <= 8 and -8 <= y <= 8 and -14 <= z <= 14
        and position not in _MIRACLE_POSITIONS
    )


_ALL_MAP_POSITIONS = tuple(
    (x, y, -(x + y))
    for x in range(-8, 9)
    for y in range(-8, 9)
    if _in_map((x, y, -(x + y)))
)


class IncompleteActionSupportError(ValueError):
    """The adapter cannot prove that its legal command support is complete."""


def _strict_int(
    value: Any,
    label: str,
    error_type: type[ValueError] = IncompleteActionSupportError,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise error_type(f"{label} must be an integer")
    return value


def _strict_enum_int(
    value: Any,
    allowed: Sequence[int],
    label: str,
    error_type: type[ValueError] = IncompleteActionSupportError,
) -> int:
    value = _strict_int(value, label, error_type)
    if value not in allowed:
        raise error_type(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class SeedBundle:
    """All random identities that must be controllable for a frozen case."""

    logic_seed: int
    evaluated_agent_seed: int
    opponent_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("logic_seed", self.logic_seed),
            ("evaluated_agent_seed", self.evaluated_agent_seed),
            ("opponent_seed", self.opponent_seed),
        ):
            value = _strict_int(value, name, ValueError)
            if not SEED_MIN <= value <= SEED_MAX:
                raise ValueError(
                    f"{name} must be in [{SEED_MIN}, {SEED_MAX}]"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "logic_seed": self.logic_seed,
            "evaluated_agent_seed": self.evaluated_agent_seed,
            "opponent_seed": self.opponent_seed,
        }


@dataclass(frozen=True)
class FrozenTestCase:
    opponent: str
    evaluated_agent_camp: int
    map_type: int
    day_time: int
    repeat: int
    seeds: SeedBundle

    def __post_init__(self) -> None:
        if self.opponent not in TEST_OPPONENTS:
            raise ValueError("final test cases must use a frozen test opponent")
        for name, value in (
            ("evaluated_agent_camp", self.evaluated_agent_camp),
            ("map_type", self.map_type),
            ("day_time", self.day_time),
        ):
            _strict_enum_int(value, (0, 1), name, ValueError)
        repeat = _strict_int(self.repeat, "repeat", ValueError)
        if not 1 <= repeat <= TEST_REPEATS:
            raise ValueError(f"repeat must be in [1, {TEST_REPEATS}]")

    @property
    def case_id(self) -> str:
        return (
            f"24m-test-{self.opponent}-c{self.evaluated_agent_camp}"
            f"-m{self.map_type}-d{self.day_time}-r{self.repeat:02d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "opponent": self.opponent,
            "evaluated_agent_camp": self.evaluated_agent_camp,
            "map_type": self.map_type,
            "day_time": self.day_time,
            "repeat": self.repeat,
            "seeds": self.seeds.to_dict(),
        }


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _logic_seed(label: str, map_type: int, day_time: int) -> int:
    """Find a stable seed whose first two Judge draws match the case."""

    candidate = _stable_seed(f"{label}:logic")
    while True:
        rng = random.Random(candidate)
        if rng.randint(0, 1) == map_type and rng.randint(0, 1) == day_time:
            return candidate
        candidate = (candidate + 1) & 0x7FFF_FFFF


def build_frozen_test_cases() -> tuple[FrozenTestCase, ...]:
    """Return the approved 3×2×2×2×3 = 72 final test cases."""

    cases: list[FrozenTestCase] = []
    for opponent in TEST_OPPONENTS:
        for camp in (0, 1):
            for map_type in (0, 1):
                for day_time in (0, 1):
                    for repeat in range(1, TEST_REPEATS + 1):
                        label = (
                            f"{PROTOCOL_VERSION}:{opponent}:{camp}:"
                            f"{map_type}:{day_time}:{repeat}"
                        )
                        cases.append(
                            FrozenTestCase(
                                opponent=opponent,
                                evaluated_agent_camp=camp,
                                map_type=map_type,
                                day_time=day_time,
                                repeat=repeat,
                                seeds=SeedBundle(
                                    logic_seed=_logic_seed(
                                        label, map_type, day_time
                                    ),
                                    evaluated_agent_seed=_stable_seed(
                                        f"{label}:evaluated-agent"
                                    ),
                                    opponent_seed=_stable_seed(
                                        f"{label}:opponent"
                                    ),
                                ),
                            )
                        )
    result = tuple(cases)
    if len(result) != 72 or len({case.case_id for case in result}) != 72:
        raise AssertionError("frozen test matrix must contain 72 unique cases")
    return result


def research_protocol_manifest() -> dict[str, Any]:
    """Produce a JSON-ready protocol manifest without executing the benchmark."""

    cases = build_frozen_test_cases()
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "BLOCKED_NOT_AUTHORITATIVE",
        "seed_contract": {
            "type": "integer",
            "minimum": SEED_MIN,
            "maximum": SEED_MAX,
            "bool_allowed": False,
        },
        "optimization_class": {
            "name": "heuristic_learning",
            "criterion": "no_backpropagation_or_gradient_updates",
        },
        "trajectory_kl": {
            "required": True,
            "epsilon": TRAJECTORY_KL_EPSILON,
            "direction": "new||old",
            "rollout_source": "new_policy",
            "decision_change_rate": "not_collected",
            "failure_policy": "mark_measurement_incomplete",
        },
        "population": {
            "train": list(TRAIN_OPPONENTS),
            "validation": list(VALIDATION_OPPONENTS),
            "test": list(TEST_OPPONENTS),
        },
        "frozen_test": {
            "case_definition": [
                "opponent_version",
                "evaluated_agent_camp",
                "map_type",
                "day_time",
                "repeat",
            ],
            "case_count": len(cases),
            "repeats": TEST_REPEATS,
            "requires_controllable_randomness": True,
            "authoritative_ready": False,
            "determinism_audit": FROZEN_OPPONENT_DETERMINISM,
            "blockers": [
                "real match runner does not invoke the Python seed launcher",
                "rank02 and rank10 consume uncontrolled time/clock state",
                "rank16 deterministic qualification remains incomplete",
            ],
            "cases": [case.to_dict() for case in cases],
        },
    }


def canonical_research_manifest_bytes(
    manifest: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a research manifest as deterministic UTF-8 canonical JSON."""

    value = research_protocol_manifest() if manifest is None else manifest
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def research_manifest_sha256(
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Hash the exact canonical bytes that accompany a Results research run."""

    return hashlib.sha256(canonical_research_manifest_bytes(manifest)).hexdigest()


def decode_ai_observation(payload: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Decode the Judge-to-AI six-digit-length observation frame."""

    if isinstance(payload, Mapping):
        return copy.deepcopy(dict(payload))
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise IncompleteActionSupportError("observation must be a frame or mapping")
    if len(raw) < 6 or not raw[:6].isdigit():
        raise IncompleteActionSupportError("observation frame has no six-digit length")
    declared = int(raw[:6].decode("ascii"))
    body = raw[6:]
    if declared != len(body):
        raise IncompleteActionSupportError(
            f"observation frame length mismatch: declared={declared}, actual={len(body)}"
        )
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompleteActionSupportError(f"invalid observation JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise IncompleteActionSupportError("observation JSON must be an object")
    return value


def _position(value: Any, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple)) or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise IncompleteActionSupportError(f"{label} must be an integer cube position")
    result = tuple(value)
    if sum(result) != 0:
        raise IncompleteActionSupportError(f"{label} must sum to zero")
    return result


def _require_sequence(value: Any, length: int | None, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise IncompleteActionSupportError(f"{label} must be a sequence")
    if length is not None and len(value) != length:
        raise IncompleteActionSupportError(f"{label} must contain {length} items")
    return value


def _cube_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum(abs(a - b) for a, b in zip(left, right)) // 2


def _cube_circle(center: tuple[int, int, int], radius: int) -> set[tuple[int, int, int]]:
    return {
        (x, y, -(x + y))
        for x in range(center[0] - radius, center[0] + radius + 1)
        for y in range(center[1] - radius, center[1] + radius + 1)
        if _cube_distance(center, (x, y, -(x + y))) <= radius
    }


def _command(observation: Mapping[str, Any], operation: str, **parameters: Any) -> dict[str, Any]:
    return {
        "player": observation["camp"],
        "round": observation.get("round", 0),
        "operation_type": operation,
        "operation_parameters": parameters,
    }


def _parse_runtime_observation(observation: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], Sequence[Any], Sequence[Any]
]:
    for key in ("round", "camp", "map", "players"):
        if key not in observation:
            raise IncompleteActionSupportError(f"observation missing {key}")
    _strict_int(observation["round"], "round")
    _strict_enum_int(observation["camp"], (0, 1), "camp")
    map_state = observation["map"]
    if not isinstance(map_state, Mapping):
        raise IncompleteActionSupportError("map must be an object")
    for key in ("units", "barracks", "miracles"):
        if key not in map_state:
            raise IncompleteActionSupportError(f"map missing {key}")
    barracks = _require_sequence(map_state["barracks"], 4, "barracks")
    miracles = _require_sequence(map_state["miracles"], 2, "miracles")
    for value in barracks:
        _strict_enum_int(value, (-1, 0, 1), "barracks owner")
    for value in miracles:
        _strict_int(value, "miracle hp")
    units = []
    for index, raw in enumerate(_require_sequence(map_state["units"], None, "units")):
        values = _require_sequence(raw, 18, f"units[{index}]")
        unit_id = _strict_int(values[0], f"units[{index}] id")
        unit_camp = _strict_enum_int(
            values[1], (0, 1), f"units[{index}] camp"
        )
        unit_type = _strict_enum_int(
            values[2], range(len(_UNIT_NAMES)), f"units[{index}] type"
        )
        attack_range = tuple(
            _strict_int(value, f"units[{index}].atk_range")
            for value in _require_sequence(
                values[7], 2, f"units[{index}].atk_range"
            )
        )
        units.append({
            "id": unit_id, "camp": unit_camp, "type": unit_type,
            "atk": _strict_int(values[4], f"units[{index}].atk"),
            "hp": _strict_int(values[6], f"units[{index}].hp"),
            "atk_range": attack_range,
            "max_move": _strict_int(values[8], f"units[{index}].max_move"),
            "pos": _position(values[10], f"units[{index}].pos"),
            "flying": bool(_strict_enum_int(
                values[12], (0, 1), f"units[{index}].flying"
            )),
            "atk_flying": bool(_strict_enum_int(
                values[13], (0, 1), f"units[{index}].atk_flying"
            )),
            "can_atk": bool(_strict_enum_int(
                values[16], (0, 1), f"units[{index}].can_atk"
            )),
            "can_move": bool(_strict_enum_int(
                values[17], (0, 1), f"units[{index}].can_move"
            )),
        })
    players_raw = _require_sequence(observation["players"], 2, "players")
    players = []
    for index, raw in enumerate(players_raw):
        values = _require_sequence(raw, 5, f"players[{index}]")
        artifacts = _require_sequence(values[0], None, f"players[{index}].artifacts")
        capacities = _require_sequence(values[3], None, f"players[{index}].capacities")
        _require_sequence(values[4], None, f"players[{index}].newly_summoned")
        mana = _strict_int(values[1], f"players[{index}].mana")
        for artifact_index, raw_artifact in enumerate(artifacts):
            artifact = _require_sequence(
                raw_artifact, 8, f"players[{index}].artifacts[{artifact_index}]"
            )
            _strict_int(artifact[0], "artifact id")
            _strict_enum_int(
                artifact[1], range(len(_ARTIFACT_NAMES)), "artifact type"
            )
            for field_index, label in (
                (2, "artifact cost"),
                (3, "artifact max cooldown"),
                (4, "artifact cooldown"),
            ):
                _strict_int(artifact[field_index], label)
            _strict_enum_int(artifact[5], (0, 1, 2), "artifact state")
            _strict_enum_int(artifact[6], (0, 1), "artifact target type")
        for capacity_index, raw_capacity in enumerate(capacities):
            capacity = _require_sequence(
                raw_capacity, 3, f"players[{index}].capacities[{capacity_index}]"
            )
            _strict_enum_int(
                capacity[0], range(len(_UNIT_NAMES)), "creature capacity type"
            )
            available = _strict_int(
                capacity[1], "creature capacity available count"
            )
            if available < 0:
                raise IncompleteActionSupportError(
                    "creature capacity available count is invalid"
                )
            _require_sequence(
                capacity[2], None,
                f"players[{index}].capacities[{capacity_index}].cooldowns",
            )
        players.append({"artifacts": artifacts, "mana": mana, "capacities": capacities})
    return units, players, barracks, miracles


def _reachable_positions(unit: Mapping[str, Any], units: Sequence[Mapping[str, Any]]) -> set[tuple[int, int, int]]:
    fixed = set((*_MIRACLE_POSITIONS, *_SPECIAL_MOVE_BORDER))
    if not unit["flying"]:
        fixed.update(_ABYSS_POSITIONS)
    fixed.update(other["pos"] for other in units if other["flying"] == unit["flying"])
    obstructs = set()
    for enemy in units:
        if enemy["camp"] == unit["camp"]:
            continue
        if enemy["flying"] == unit["flying"]:
            obstructs.update(
                (enemy["pos"][0] + dx, enemy["pos"][1] + dy, enemy["pos"][2] + dz)
                for dx, dy, dz in _DIRECTIONS
            )
        else:
            obstructs.add(enemy["pos"])
    obstructs.discard(unit["pos"])
    visited = {unit["pos"]}
    frontier = {unit["pos"]}
    for _ in range(unit["max_move"]):
        following = set()
        for current in frontier:
            if current in obstructs:
                continue
            for dx, dy, dz in _DIRECTIONS:
                candidate = (current[0] + dx, current[1] + dy, current[2] + dz)
                if candidate not in visited and candidate not in fixed and _in_map(candidate):
                    visited.add(candidate)
                    following.add(candidate)
        frontier = following
    visited.discard(unit["pos"])
    return visited


def enumerate_legal_commands(
    payload: str | bytes | Mapping[str, Any],
) -> LegalCommandSet:
    """Enumerate the complete finite Judge command support for one observation."""

    observation = decode_ai_observation(payload)
    if set(observation) == {"camp"}:
        _strict_enum_int(observation["camp"], (0, 1), "camp")
        commands = [
            _command(
                observation,
                "init",
                artifacts=[artifact],
                creatures=list(creatures),
            )
            for artifact in _ARTIFACT_NAMES
            for creatures in permutations(_UNIT_NAMES, 3)
        ]
        return LegalCommandSet(commands, complete=True)

    units, players, barracks, miracles = _parse_runtime_observation(observation)
    camp = observation["camp"]
    player = players[camp]
    commands: list[dict[str, Any]] = []
    for unit in units:
        if unit["camp"] != camp:
            continue
        if unit["can_move"]:
            commands.extend(
                _command(observation, "move", mover=unit["id"], position=list(position))
                for position in sorted(_reachable_positions(unit, units))
            )
        if unit["can_atk"] and unit["atk"] > 0:
            minimum, maximum = unit["atk_range"]
            for target in units:
                distance = _cube_distance(unit["pos"], target["pos"])
                if (
                    target["camp"] != camp and target["hp"] > 0
                    and minimum <= distance <= maximum
                    and (not target["flying"] or unit["flying"] or unit["atk_flying"])
                ):
                    commands.append(
                        _command(observation, "attack", attacker=unit["id"], target=target["id"])
                    )
            enemy = camp ^ 1
            if miracles[enemy] > 0 and minimum <= _cube_distance(unit["pos"], _MIRACLE_POSITIONS[enemy]) <= maximum:
                commands.append(_command(observation, "attack", attacker=unit["id"], target=enemy))

    summon_positions = list(_MIRACLE_SUMMON_POSITIONS[camp])
    for index, owner in enumerate(barracks):
        if owner == camp:
            summon_positions.extend(_BARRACK_SUMMON_POSITIONS[index])
    occupied = {(unit["pos"], unit["flying"]) for unit in units}
    for raw_capacity in player["capacities"]:
        capacity = _require_sequence(raw_capacity, 3, "creature capacity")
        type_index, available = capacity[0], capacity[1]
        type_index = _strict_enum_int(
            type_index, range(len(_UNIT_NAMES)), "creature capacity type"
        )
        available = _strict_int(available, "creature capacity available count")
        if available <= 0:
            continue
        for level, cost in enumerate(_UNIT_COSTS[type_index], start=1):
            if player["mana"] < cost:
                continue
            for position in summon_positions:
                if (position, _UNIT_FLYING[type_index]) not in occupied:
                    commands.append(_command(
                        observation, "summon", type=_UNIT_NAMES[type_index],
                        level=level, position=list(position),
                    ))

    for raw_artifact in player["artifacts"]:
        artifact = _require_sequence(raw_artifact, 8, "artifact")
        artifact_id, artifact_type, cost, _max_cd, _cd, state, target_type, _last = artifact
        artifact_id = _strict_int(artifact_id, "artifact id")
        artifact_type = _strict_enum_int(
            artifact_type, range(len(_ARTIFACT_NAMES)), "artifact type"
        )
        state = _strict_enum_int(state, (0, 1, 2), "artifact state")
        target_type = _strict_enum_int(
            target_type, (0, 1), "artifact target type"
        )
        if state != 0 or player["mana"] < cost:
            continue
        name = _ARTIFACT_NAMES[artifact_type]
        if name == "WindBlessing":
            raise IncompleteActionSupportError(
                "WindBlessing has unbounded legal position support in the authoritative Judge"
            )
        if name == "HolyLight":
            targets: Sequence[Any] = _ALL_MAP_POSITIONS
        elif name == "SalamanderShield":
            targets = [unit["id"] for unit in units]
        else:
            inferno_targets = _cube_circle(_MIRACLE_POSITIONS[camp], 7)
            for index, owner in enumerate(barracks):
                if owner == camp:
                    inferno_targets.update(_cube_circle(_BARRACK_POSITIONS[index], 5))
            ground_units = {unit["pos"] for unit in units if not unit["flying"]}
            targets = sorted(inferno_targets - _GROUND_OBSTACLES - ground_units)
        expected_target_type = 1 if name == "SalamanderShield" else 0
        if target_type != expected_target_type:
            raise IncompleteActionSupportError("artifact target type disagrees with protocol data")
        commands.extend(
            _command(
                observation, "use", card=artifact_id,
                target=(target if isinstance(target, int) else list(target)),
            )
            for target in targets
        )

    commands.append(_command(observation, "endround"))
    commands.append(_command(observation, "surrender"))
    return LegalCommandSet(commands, complete=True)


def canonical_command(command: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one atomic Judge command."""

    if not isinstance(command, Mapping):
        raise TypeError("Miracle command must be a mapping")
    operation_type = command.get("operation_type")
    if not isinstance(operation_type, str):
        raise ValueError("operation_type must be a string")
    operation_type = operation_type.lower()
    if operation_type not in _OPERATION_TYPES:
        raise ValueError(f"unsupported Miracle operation: {operation_type!r}")
    player = command.get("player")
    player = _strict_enum_int(player, (0, 1), "player", ValueError)
    round_number = command.get("round")
    if not isinstance(round_number, int) or isinstance(round_number, bool):
        raise ValueError("round must be an integer")
    parameters = command.get("operation_parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("operation_parameters must be a mapping")
    parameters = dict(parameters)

    expected_keys = {
        "init": {"artifacts", "creatures"},
        "move": {"mover", "position"},
        "attack": {"attacker", "target"},
        "summon": {"type", "level", "position"},
        "use": {"card", "target"},
        "endround": set(),
        "surrender": set(),
    }[operation_type]
    if set(parameters) != expected_keys:
        raise ValueError(
            f"{operation_type} parameters must be {sorted(expected_keys)}"
        )

    def command_id(value: Any, label: str) -> int:
        return _strict_int(value, label, ValueError)

    if operation_type == "init":
        if round_number != 0:
            raise ValueError("init round must be 0")
        artifacts = parameters["artifacts"]
        creatures = parameters["creatures"]
        if (
            not isinstance(artifacts, (list, tuple)) or len(artifacts) != 1
            or artifacts[0] not in _ARTIFACT_NAMES
        ):
            raise ValueError("init must select one known artifact")
        if (
            not isinstance(creatures, (list, tuple)) or len(creatures) != 3
            or len(set(creatures)) != 3
            or any(item not in _UNIT_NAMES for item in creatures)
        ):
            raise ValueError("init must select three distinct known creatures")
        parameters = {"artifacts": list(artifacts), "creatures": list(creatures)}
    elif operation_type == "move":
        parameters["mover"] = command_id(parameters["mover"], "mover")
        parameters["position"] = list(_position(parameters["position"], "position"))
    elif operation_type == "attack":
        parameters["attacker"] = command_id(parameters["attacker"], "attacker")
        parameters["target"] = command_id(parameters["target"], "target")
    elif operation_type == "summon":
        if parameters["type"] not in _UNIT_NAMES:
            raise ValueError("summon type is unknown")
        parameters["level"] = _strict_enum_int(
            parameters["level"], (1, 2, 3), "summon level", ValueError
        )
        parameters["position"] = list(_position(parameters["position"], "position"))
    elif operation_type == "use":
        parameters["card"] = command_id(parameters["card"], "card")
        target = parameters["target"]
        if isinstance(target, bool) or not isinstance(target, (int, list, tuple)):
            raise ValueError("artifact target must be a unit ID or cube position")
        parameters["target"] = (
            command_id(target, "target")
            if isinstance(target, int)
            else list(_position(target, "target"))
        )
    return {
        "player": player,
        "round": round_number,
        "operation_type": operation_type,
        "operation_parameters": parameters,
    }


def command_action_id(command: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_command(command),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{ACTION_SCHEMA_VERSION}:{hashlib.sha256(payload.encode()).hexdigest()}"


@dataclass(frozen=True)
class LegalCommandSet:
    """Adapter-owned claim about all legal atomic commands at one state."""

    commands: Sequence[Mapping[str, Any]]
    complete: bool
    errors: Sequence[str] = ()


def build_action_support(legal: LegalCommandSet) -> ActionSupport:
    """Build strict support, failing closed when completeness is not proven."""

    if not legal.complete:
        detail = "; ".join(str(error) for error in legal.errors)
        suffix = f": {detail}" if detail else ""
        raise IncompleteActionSupportError(
            f"complete legal Miracle command support is unavailable{suffix}"
        )
    candidates = [
        ActionCandidate(command_action_id(command), canonical_command(command))
        for command in legal.commands
    ]
    candidates.sort(key=lambda candidate: candidate.action_id)
    return ActionSupport(candidates, ACTION_SCHEMA_VERSION)


def one_hot(action_id: str, support: ActionSupport) -> dict[str, float]:
    if action_id not in support.action_ids:
        raise ValueError("deterministic policy selected an action outside support")
    return {
        candidate_id: float(candidate_id == action_id)
        for candidate_id in support.action_ids
    }


def _select_id(
    selector: Callable[[Any, ActionSupport], str | Mapping[str, Any]],
    observation: Any,
    support: ActionSupport,
) -> str:
    selected = selector(observation, support)
    return selected if isinstance(selected, str) else command_action_id(selected)


class IsolatedDeterministicPolicy:
    """Query a deterministic selector on a fresh deep-copied policy instance.

    The source selector, observation, and support are never handed to the query.
    This is the fake-testable isolation primitive required by an actual
    Miracle policy adapter; it deliberately does not pretend an unknown or
    randomized policy is one-hot.
    """

    def __init__(
        self,
        selector: Callable[[Any, ActionSupport], str | Mapping[str, Any]],
    ) -> None:
        self._selector = selector

    def _query(self, observation: Any, support: ActionSupport) -> str:
        try:
            selector = copy.deepcopy(self._selector)
            isolated_observation = copy.deepcopy(observation)
            isolated_support = copy.deepcopy(support)
        except Exception as exc:
            raise RuntimeError(f"policy query cannot be isolated: {exc}") from exc
        return _select_id(selector, isolated_observation, isolated_support)

    def decide_with_distribution(
        self, observation: Any, support: ActionSupport
    ) -> PolicyDecision:
        action_id = self._query(observation, support)
        return PolicyDecision(action_id, one_hot(action_id, support))

    def distribution_for_measurement(
        self, observation: Any, support: ActionSupport
    ) -> dict[str, float]:
        return one_hot(self._query(observation, support), support)


class DeterministicHLActivePolicy:
    """Strict no-gradient HL adapter that reports its true one-hot policy."""

    def __init__(
        self,
        selector: Callable[[Any, ActionSupport], str | Mapping[str, Any]],
        *,
        reset: Callable[[], None] | None = None,
        observe_transition: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        self._selector = selector
        self._reset = reset
        self._observe_transition = observe_transition

    def decide_with_distribution(
        self, observation: Any, support: ActionSupport
    ) -> PolicyDecision:
        action_id = _select_id(self._selector, observation, support)
        return PolicyDecision(action_id, one_hot(action_id, support))

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def observe_transition(self, transition: Mapping[str, Any]) -> None:
        if self._observe_transition is not None:
            self._observe_transition(transition)


class DeterministicHLReferencePolicy:
    """Read-only old-version query adapter on the active rollout state."""

    def __init__(
        self,
        read_only_selector: Callable[
            [Any, ActionSupport], str | Mapping[str, Any]
        ],
        *,
        reset: Callable[[], None] | None = None,
        observe_transition: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        self._selector = read_only_selector
        self._reset = reset
        self._observe_transition = observe_transition

    def distribution_for_measurement(
        self, observation: Any, support: ActionSupport
    ) -> dict[str, float]:
        action_id = _select_id(self._selector, observation, support)
        return one_hot(action_id, support)

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def observe_transition(self, transition: Mapping[str, Any]) -> None:
        if self._observe_transition is not None:
            self._observe_transition(transition)

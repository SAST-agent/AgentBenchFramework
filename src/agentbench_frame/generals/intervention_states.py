"""Deterministic official-state recipes for the expanded Generals KL domain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .assets import load_expanded_kl_config
from .engine import OfficialGeneralsEngine
from .measurement_state import measurement_state_id
from .models import ExpandedPolicyKLConfig, InterventionStateSpec


PACK_SCHEMA = "generals-policy-kl-intervention-states-v1"
MAIN_POSITIONS = {0: (7, 4), 1: (7, 10)}
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class InterventionState:
    state_key: str
    scenario: str
    actor: int
    variant: int
    snapshot: Mapping[str, Any]
    measurement_state_id: str
    construction_receipt: Mapping[str, Any]
    assertion_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class InterventionStatePack:
    measurement_id: str
    schema: str
    states: tuple[InterventionState, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "schema": self.schema,
            "state_count": len(self.states),
            "states": [asdict(item) for item in self.states],
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.payload(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _cells(snapshot: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        (int(item["position"][0]), int(item["position"][1])): item
        for item in snapshot["state"]["board"]
    }


def _neighbors(position: tuple[int, int]):
    return tuple((position[0] + dr, position[1] + dc) for dr, dc in _DELTAS)


def _owned_movable_stack_touches_hostile(state: InterventionState) -> bool:
    cells = _cells(state.snapshot)
    for position, cell in cells.items():
        if int(cell["player"]) != state.actor or int(cell["army"]) <= 1:
            continue
        for neighbor in _neighbors(position):
            other = cells.get(neighbor)
            if (
                other is not None
                and int(other["type"]) != 2
                and int(other["player"]) == 1 - state.actor
            ):
                return True
    return False


def _hostile_pressure_exceeds_main_reserve(state: InterventionState) -> bool:
    cells = _cells(state.snapshot)
    main = next(
        item for item in state.snapshot["state"]["generals"]
        if item["type"] == "MainGenerals" and int(item["player"]) == state.actor
    )
    position = tuple(int(item) for item in main["position"])
    reserve = int(cells[position]["army"])
    pressure = sum(
        max(int(cells[neighbor]["army"]) - 1, 0)
        for neighbor in _neighbors(position)
        if neighbor in cells
        and int(cells[neighbor]["player"]) == 1 - state.actor
    )
    return pressure >= reserve and pressure > 0


def _two_routes_include_stack_at_least_24(state: InterventionState) -> bool:
    main_positions = {
        tuple(int(value) for value in item["position"])
        for item in state.snapshot["state"]["generals"]
    }
    stacks = [
        int(cell["army"])
        for position, cell in _cells(state.snapshot).items()
        if int(cell["player"]) == state.actor
        and int(cell["army"]) > 1
        and position not in main_positions
    ]
    return len(stacks) >= 2 and max(stacks) >= 24


def _contact_and_affordable_upgrade(state: InterventionState) -> bool:
    return (
        _owned_movable_stack_touches_hostile(state)
        and int(state.snapshot["state"]["coin"][state.actor]) >= 40
    )


def _exact_adjacent_counter_capture_exists(state: InterventionState) -> bool:
    cells = _cells(state.snapshot)
    for position, cell in cells.items():
        if int(cell["player"]) != state.actor or int(cell["army"]) <= 1:
            continue
        for neighbor in _neighbors(position):
            hostile = cells.get(neighbor)
            if (
                hostile is not None
                and int(hostile["player"]) == 1 - state.actor
                and int(cell["army"]) - 1 > int(hostile["army"])
            ):
                return True
    return False


def _round_at_least_100_and_owned_merge_exists(state: InterventionState) -> bool:
    if int(state.snapshot["state"]["round"]) < 100:
        return False
    main_positions = {
        tuple(int(value) for value in item["position"])
        for item in state.snapshot["state"]["generals"]
    }
    cells = _cells(state.snapshot)
    stacks = [
        position
        for position, cell in cells.items()
        if int(cell["player"]) == state.actor
        and int(cell["army"]) > 1
        and position not in main_positions
    ]
    for left in stacks:
        for right in stacks:
            if left >= right:
                continue
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) != 2:
                continue
            middle = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
            cell = cells.get(middle)
            if cell is not None and int(cell["type"]) != 2:
                return True
    return False


SCENARIO_ASSERTIONS: Mapping[str, Callable[[InterventionState], bool]] = MappingProxyType({
    "contact": _owned_movable_stack_touches_hostile,
    "main_general_danger": _hostile_pressure_exceeds_main_reserve,
    "large_stack_routing": _two_routes_include_stack_at_least_24,
    "economy_combat_conflict": _contact_and_affordable_upgrade,
    "counter_capture": _exact_adjacent_counter_capture_exists,
    "mid_late_consolidation": _round_at_least_100_and_owned_merge_exists,
})


def assert_intervention_scenario(state: InterventionState) -> None:
    if state.scenario not in SCENARIO_ASSERTIONS:
        raise ValueError(f"unknown intervention scenario: {state.scenario}")
    if measurement_state_id(state.snapshot) != state.measurement_state_id:
        raise ValueError("intervention measurement-state hash changed")
    if state.snapshot.get("actor") != state.actor:
        raise ValueError("intervention actor changed")
    if not SCENARIO_ASSERTIONS[state.scenario](state):
        raise ValueError(f"intervention predicate failed: {state.scenario}")


def _official_module(engine: OfficialGeneralsEngine):
    return sys.modules[engine._main.GameState.__module__]


def _reset_engine(
    engine: OfficialGeneralsEngine,
    *,
    actor: int,
    variant: int,
) -> None:
    state = engine.state
    module = _official_module(engine)
    mains = sorted(
        (item for item in state.generals if type(item).__name__ == "MainGenerals"),
        key=lambda item: int(item.player),
    )
    if len(mains) != 2 or {int(item.player) for item in mains} != {0, 1}:
        raise ValueError("official initial state does not contain both mains")
    for row in state.board:
        for cell in row:
            cell.type = module.CellType.PLAIN
            cell.player = -1
            cell.army = 0
            cell.generals = None
            cell.weapon_activate = []
    for general in mains:
        player = int(general.player)
        position = MAIN_POSITIONS[player]
        general.position = [position[0], position[1]]
        general.rest_move = 1
        general.produce_level = 1
        general.defense_level = 1
        general.mobility_level = 1
        general.skills_cd = [0 for _ in general.skills_cd]
        general.skill_duration = [0 for _ in general.skill_duration]
        cell = state.board[position[0]][position[1]]
        cell.player = player
        cell.army = 20
        cell.generals = general
    state.generals = mains
    state.next_generals_id = 2
    state.active_super_weapon = []
    state.super_weapon_unlocked = [False, False]
    state.super_weapon_cd = [-1, -1]
    state.tech_level = [[2, 0, 0, 0], [2, 0, 0, 0]]
    state.rest_move_step = [2, 2]
    # Keep ordinary scenarios below the first economy upgrade threshold;
    # the economy/combat recipe raises only the acting side to exactly 40.
    state.coin = [20, 20]
    state.round = 60
    state.winner = -1
    state.changed_cells = []
    terrain_position = (0, 0) if variant == 0 else (14, 14)
    state.board[terrain_position[0]][terrain_position[1]].type = module.CellType.BOG
    if actor not in (0, 1):
        raise ValueError("intervention actor must be 0 or 1")


def _set_cell(
    engine: OfficialGeneralsEngine,
    position: tuple[int, int],
    *,
    player: int,
    army: int,
) -> None:
    module = _official_module(engine)
    cell = engine.state.board[position[0]][position[1]]
    if cell.generals is not None:
        raise ValueError(f"recipe attempted to overwrite a main at {position}")
    cell.type = module.CellType.PLAIN
    cell.player = player
    cell.army = army
    cell.generals = None
    cell.weapon_activate = []


def _toward_center(actor: int, row: int, distance: int = 0) -> tuple[int, int]:
    base = MAIN_POSITIONS[actor][1]
    direction = 1 if actor == 0 else -1
    return row, base + direction * distance


def _apply_recipe(
    engine: OfficialGeneralsEngine,
    spec: InterventionStateSpec,
) -> dict[str, Any]:
    actor = spec.actor
    hostile = 1 - actor
    receipt: dict[str, Any] = {
        "recipe_seed": 0,
        "actor_main_position": list(MAIN_POSITIONS[actor]),
        "opponent_main_position": list(MAIN_POSITIONS[hostile]),
        "variant_terrain_position": [0, 0] if spec.variant == 0 else [14, 14],
    }
    if spec.scenario in {"contact", "economy_combat_conflict"}:
        owned = _toward_center(actor, 5, 1)
        enemy = _toward_center(actor, 5, 2)
        _set_cell(engine, owned, player=actor, army=7)
        _set_cell(engine, enemy, player=hostile, army=2)
        receipt.update(owned_army=7, hostile_army=2)
        if spec.scenario == "economy_combat_conflict":
            engine.state.coin[actor] = 40
            receipt["coins"] = 40
    elif spec.scenario == "main_general_danger":
        main = MAIN_POSITIONS[actor]
        enemy = _toward_center(actor, 7, 1)
        engine.state.board[main[0]][main[1]].army = 13
        _set_cell(engine, enemy, player=hostile, army=20)
        receipt.update(main_army=13, hostile_army=20)
    elif spec.scenario == "large_stack_routing":
        small = _toward_center(actor, 5, 1)
        large = _toward_center(actor, 9, 1)
        _set_cell(engine, small, player=actor, army=4)
        _set_cell(engine, large, player=actor, army=30)
        receipt.update(small_stack=4, large_stack=30)
    elif spec.scenario == "counter_capture":
        owned = _toward_center(actor, 6, 0)
        enemy = _toward_center(actor, 6, 1)
        _set_cell(engine, owned, player=actor, army=9)
        _set_cell(engine, enemy, player=hostile, army=5)
        receipt.update(owned_army=9, hostile_army=5)
    elif spec.scenario == "mid_late_consolidation":
        first = _toward_center(actor, 5, 0)
        middle = _toward_center(actor, 5, 1)
        second = _toward_center(actor, 5, 2)
        _set_cell(engine, first, player=actor, army=18)
        _set_cell(engine, middle, player=actor, army=1)
        _set_cell(engine, second, player=actor, army=14)
        engine.state.round = 120
        receipt.update(round=120, stack_a=18, stack_b=14)
    else:
        raise ValueError(f"unsupported intervention scenario: {spec.scenario}")
    return receipt


def _build_state(
    engine_root: Path,
    spec: InterventionStateSpec,
    recipe_index: int,
) -> InterventionState:
    seed = 304000 + recipe_index
    engine = OfficialGeneralsEngine(engine_root, seed=seed)
    _reset_engine(engine, actor=spec.actor, variant=spec.variant)
    construction = _apply_recipe(engine, spec)
    construction["recipe_seed"] = seed
    snapshot = engine.measurement_state(spec.actor)
    state_id = measurement_state_id(snapshot)
    provisional = InterventionState(
        state_key=spec.state_key,
        scenario=spec.scenario,
        actor=spec.actor,
        variant=spec.variant,
        snapshot=snapshot,
        measurement_state_id=state_id,
        construction_receipt=construction,
        assertion_receipt={},
    )
    assert_intervention_scenario(provisional)
    with tempfile.TemporaryDirectory(prefix="generals-intervention-") as temporary:
        restored = OfficialGeneralsEngine.from_measurement_state(
            engine_root,
            snapshot,
            Path(temporary) / "replay.jsonl",
        )
        round_trip = restored.measurement_state(spec.actor) == snapshot
        legal_end = not restored.apply_turn(spec.actor, ((8,),)).done
    mains = [
        item for item in snapshot["state"]["generals"]
        if item["type"] == "MainGenerals"
    ]
    assertions = {
        "scenario_predicate": True,
        "measurement_state_round_trip": round_trip,
        "nonterminal": snapshot["state"]["winner"] == -1,
        "both_mains_present": len(mains) == 2 and {item["player"] for item in mains} == {0, 1},
        "legal_end_macro": legal_end,
    }
    if not all(assertions.values()):
        raise ValueError(f"intervention assertions failed for {spec.state_key}")
    return InterventionState(
        **{
            **asdict(provisional),
            "assertion_receipt": assertions,
        }
    )


def build_intervention_state_pack(
    engine_root: Path,
    manifest: ExpandedPolicyKLConfig | Path,
) -> InterventionStatePack:
    config = (
        load_expanded_kl_config(Path(manifest))
        if isinstance(manifest, (str, Path))
        else manifest
    )
    states = tuple(
        _build_state(Path(engine_root), spec, index)
        for index, spec in enumerate(config.intervention_states)
    )
    if len({item.measurement_state_id for item in states}) != len(states):
        raise ValueError("intervention measurement-state IDs must be unique")
    return InterventionStatePack(
        measurement_id=config.measurement_id,
        schema=PACK_SCHEMA,
        states=states,
    )


def load_intervention_state_pack(path: Path) -> InterventionStatePack:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    states = tuple(
        InterventionState(
            state_key=str(item["state_key"]),
            scenario=str(item["scenario"]),
            actor=int(item["actor"]),
            variant=int(item["variant"]),
            snapshot=item["snapshot"],
            measurement_state_id=str(item["measurement_state_id"]),
            construction_receipt=item["construction_receipt"],
            assertion_receipt=item["assertion_receipt"],
        )
        for item in raw["states"]
    )
    pack = InterventionStatePack(
        measurement_id=str(raw["measurement_id"]),
        schema=str(raw["schema"]),
        states=states,
    )
    if raw.get("state_count") != len(states) or pack.schema != PACK_SCHEMA:
        raise ValueError("invalid intervention state pack")
    for item in states:
        assert_intervention_scenario(item)
    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentbench-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    engine_root = (
        args.agentbench_root
        / "backend_sources/corpus/28_generals/logic/gamecode_logic"
    )
    pack = build_intervention_state_pack(engine_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(pack.canonical_bytes())
    temporary.replace(args.output)
    print(f"{pack.sha256}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

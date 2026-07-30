from copy import deepcopy
from itertools import islice
import os
from pathlib import Path

import pytest

from agentbench_frame.generals.engine import OfficialGeneralsEngine


@pytest.fixture
def engine_root():
    value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    return (
        Path(value)
        / "backend_sources/corpus/28_generals/logic/gamecode_logic"
    )


@pytest.fixture
def controlled_state(engine_root, tmp_path):
    engine = OfficialGeneralsEngine(
        engine_root,
        289101,
        tmp_path / "controlled.jsonl",
    )
    state = engine.state
    state.coin = [2_000, 2_000]
    state.rest_move_step = [5, 5]
    state.super_weapon_unlocked = [True, True]
    state.super_weapon_cd = [0, 0]
    main = next(
        general
        for general in state.generals
        if type(general).__name__ == "MainGenerals"
        and general.player == 0
    )
    main.rest_move = 4
    main.skills_cd = [0, 0, 0, 0, 0]
    source = state.board[main.position[0]][main.position[1]]
    source.army = 10
    source.player = 0
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        row = main.position[0] + dx
        column = main.position[1] + dy
        if 0 <= row < 15 and 0 <= column < 15:
            target = state.board[row][column]
            if target.generals is None:
                target.type = type(target.type)(0)
                target.player = 0
                target.army = 1
                break
    return engine.measurement_state(actor=0)


@pytest.fixture
def action_space(engine_root, tmp_path):
    from agentbench_frame.generals.macro_action_space import (
        GeneralsMacroActionSpaceV1,
    )

    return GeneralsMacroActionSpaceV1(
        engine_root,
        replay_path=tmp_path / "probes.jsonl",
    )


@pytest.mark.parametrize("opcode", range(1, 8))
def test_action_space_exposes_every_official_command_family(
    action_space,
    controlled_state,
    opcode,
):
    assert next(
        action_space.legal_transitions(
            controlled_state,
            command_prefix=(opcode,),
        )
    ).command[0] == opcode


def test_action_space_includes_immediate_end(action_space, controlled_state):
    assert action_space.contains(controlled_state, [[8]])
    assert action_space.canonicalize(controlled_state, [[8]]) == ((8,),)


@pytest.mark.parametrize("quality", (1, 2, 3))
def test_all_general_upgrade_families_are_reachable(
    action_space,
    controlled_state,
    quality,
):
    commands = {
        item.command[2]
        for item in action_space.legal_transitions(
            controlled_state,
            command_prefix=(3,),
        )
    }
    assert quality in commands


@pytest.mark.parametrize("skill", (1, 2, 3, 4, 5))
def test_all_general_skill_families_are_reachable(
    action_space,
    controlled_state,
    skill,
):
    transition = next(
        action_space.legal_transitions(
            controlled_state,
            command_prefix=(4,),
            subcode=skill,
        )
    )
    assert transition.command[:3:2] == (4, skill)


@pytest.mark.parametrize("technology", (1, 2, 3, 4))
def test_all_technology_families_are_reachable(
    action_space,
    controlled_state,
    technology,
):
    transition = next(
        action_space.legal_transitions(
            controlled_state,
            command_prefix=(5, technology),
        )
    )
    assert transition.command == (5, technology)


@pytest.mark.parametrize("weapon", (1, 2, 3, 4))
def test_all_super_weapon_families_are_reachable(
    action_space,
    controlled_state,
    weapon,
):
    transition = next(
        action_space.legal_transitions(
            controlled_state,
            command_prefix=(6, weapon),
        )
    )
    assert transition.command[:2] == (6, weapon)


def test_canonicalize_collapses_army_request_sentinel(
    action_space,
    controlled_state,
):
    general = next(
        item
        for item in controlled_state["state"]["generals"]
        if item["type"] == "MainGenerals" and item["player"] == 0
    )
    row, column = general["position"]
    direction = next(
        direction
        for direction, (dx, dy) in enumerate(
            ((-1, 0), (0, 1), (1, 0), (0, -1)),
            start=1,
        )
        if 0 <= row + dx < 15 and 0 <= column + dy < 15
        and controlled_state["state"]["board"][
            (row + dx) * 15 + column + dy
        ]["type"]
        != 2
    )

    canonical = action_space.canonicalize(
        controlled_state,
        [[1, row, column, direction, 1_000_000_000], [8]],
    )

    assert canonical == ((1, row, column, direction, 9), (8,))


def test_canonicalize_drops_ignored_skill_coordinates(
    action_space,
    controlled_state,
):
    general_id = next(
        item["id"]
        for item in controlled_state["state"]["generals"]
        if item["type"] == "MainGenerals" and item["player"] == 0
    )

    assert action_space.canonicalize(
        controlled_state,
        [[4, general_id, 3, 7, 8], [8]],
    ) == ((4, general_id, 3), (8,))


def test_canonicalize_rejects_missing_or_early_end(
    action_space,
    controlled_state,
):
    with pytest.raises(ValueError, match="end marker"):
        action_space.canonicalize(controlled_state, [[5, 1]])
    with pytest.raises(ValueError, match="after end marker"):
        action_space.canonicalize(
            controlled_state,
            [[8], [5, 1]],
        )


def test_canonicalize_drops_unreachable_terminal_suffix(
    action_space,
    controlled_state,
):
    terminal = deepcopy(controlled_state)
    opponent_main = next(
        item
        for item in terminal["state"]["generals"]
        if item["type"] == "MainGenerals" and item["player"] == 1
    )
    terminal["state"]["generals"].remove(opponent_main)
    for cell in terminal["state"]["board"]:
        if cell["general_id"] == opponent_main["id"]:
            cell["general_id"] = None
            break

    assert action_space.canonicalize(
        terminal,
        [[5, 1], [99, 1, 2], [8]],
    ) == ((5, 1), (8,))


def test_candidate_order_is_stable(action_space, controlled_state):
    first = list(islice(action_space.primitive_candidates(controlled_state), 50))
    second = list(
        islice(action_space.primitive_candidates(controlled_state), 50)
    )

    assert first == second
    assert first == sorted(first)


def test_early_state_candidate_domain_prunes_only_impossible_requests(
    action_space,
    engine_root,
    tmp_path,
):
    engine = OfficialGeneralsEngine(
        engine_root,
        289101,
        tmp_path / "early.jsonl",
    )
    state = engine.measurement_state(actor=0)
    candidates = list(action_space.primitive_candidates(state))
    raw = state["state"]
    owned_general_ids = {
        item["id"]
        for item in raw["generals"]
        if item["player"] == 0
    }
    board_by_position = {
        tuple(item["position"]): item for item in raw["board"]
    }

    assert all(
        command[1] in owned_general_ids
        for command in candidates
        if command[0] in (2, 3, 4)
    )
    assert all(
        board_by_position[(command[1], command[2])]["player"] == 0
        and board_by_position[(command[1], command[2])]["army"] > 1
        for command in candidates
        if command[0] == 1
    )
    assert all(command[0] != 6 for command in candidates)
    assert all(command[0] != 7 for command in candidates)
    assert len(candidates) < 1_000


@pytest.mark.parametrize(
    "action",
    (
        [],
        [[0], [8]],
        [[9], [8]],
        [[1, 0], [8]],
        [["5", 1], [8]],
    ),
)
def test_contains_returns_false_for_malformed_actions(
    action_space,
    controlled_state,
    action,
):
    assert action_space.contains(controlled_state, action) is False

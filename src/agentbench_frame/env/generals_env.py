"""
Generals Game Environment

Implements the 28th THUAC Generals game as a gym-compatible environment.
Generals is a 2-player turn-based strategy game on a 15x15 grid with:
- Terrain types: Plain, Bog, Mountain
- Generals (main + sub) with skills: Surprise Attack, Rout, Command, Defence, Weaken
- Super Weapons: Nuclear Boom, Attack Enhance, Transmission, Time Stop
- Tech upgrades: Army Mobility, Mountaineering, Swap Immunity, Super Weapon Unlock
- Resource management: Coins, Army production

This module provides a self-contained implementation suitable for both
rule-based iteration and RL training.
"""

import random
import copy
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from agentbench_frame.env.base import BaseEnv, EnvMode, ActionSpace, Observation


# ---- Game Constants ----

BOARD_ROWS = 15
BOARD_COLS = 15
MAX_ROUNDS = 500
INITIAL_COINS = 40
ARMY_PRODUCTION_BASE = 1

# Terrain generation probabilities
BOG_PERCENT = 0.15
MOUNTAIN_PERCENT = 0.40

# General counts
FARMER_NUM = 10
SUBGENERAL_NUM = 4

# Upgrade costs
UPGRADE_COSTS = {
    "farmer_production_1": 10, "farmer_production_2": 25, "farmer_production_3": 35,
    "farmer_defense_1": 10, "farmer_defense_2": 15, "farmer_defense_3": 30,
    "sub_recruit": 50,
    "sub_production_1": 40, "sub_production_2": 80,
    "sub_defense_1": 40, "sub_defense_2": 100,
    "general_movement_1": 20, "general_movement_2": 40,
    "tactical_strike": 20, "breakthrough": 15,
    "leadership": 30, "fortification": 30, "weakening": 30,
}

TECH_COSTS = {
    "army_mobility_1": 80, "army_mobility_2": 150,
    "mountaineering": 100,
    "swamp_immunity": 75,
    "unlock_super_weapon": 250,
}

SUPERWEAPON_COST = 50


# ---- Enums ----

class CellType(IntEnum):
    PLAIN = 0
    BOG = 1
    MOUNTAIN = 2


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SkillType(IntEnum):
    SURPRISE_ATTACK = 0
    ROUT = 1
    COMMAND = 2
    DEFENCE = 3
    WEAKEN = 4


class WeaponType(IntEnum):
    NUCLEAR_BOOM = 0
    ATTACK_ENHANCE = 1
    TRANSMISSION = 2
    TIME_STOP = 3


class CommandType(IntEnum):
    """Operation command types matching the original Generals protocol."""
    MOVE_ARMY = 1
    MOVE_GENERAL = 2
    UPGRADE = 3
    SKILL = 4
    TECH = 5
    SUPERWEAPON = 6
    CALL_SUBGENERAL = 7
    END_TURN = 8
    SURRENDER = 9


# Direction offsets
DIRECTION_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


# ---- Data Classes ----

@dataclass
class General:
    """Base class for generals (main, sub, farmer)."""
    id: int
    player: int
    position: Tuple[int, int]
    produce_level: int = 1
    defense_level: int = 1
    mobility_level: int = 1
    is_main: bool = False
    is_sub: bool = False
    skills_cd: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    skill_duration: List[int] = field(default_factory=lambda: [0, 0, 0])

    @property
    def is_farmer(self) -> bool:
        return not self.is_main and not self.is_sub


@dataclass
class Cell:
    """A single cell on the board."""
    position: Tuple[int, int] = (0, 0)
    type: CellType = CellType.PLAIN
    player: int = -1
    army: int = 0
    general: Optional[General] = None


# ---- Game State ----

@dataclass
class GeneralsState:
    """Complete game state for Generals."""
    board: List[List[Cell]] = field(default_factory=list)
    generals: List[General] = field(default_factory=list)
    coins: List[int] = field(default_factory=lambda: [INITIAL_COINS, INITIAL_COINS])
    round: int = 0
    current_player: int = 0
    winner: int = -1
    tech_levels: List[Dict[str, int]] = field(default_factory=lambda: [
        {"mobility": 0, "climb": 0, "immune": 0, "unlock": 0},
        {"mobility": 0, "climb": 0, "immune": 0, "unlock": 0},
    ])
    superweapon_cd: List[int] = field(default_factory=lambda: [0, 0])
    moves_made: List[int] = field(default_factory=lambda: [0, 0])

    def get_player_generals(self, player: int) -> List[General]:
        return [g for g in self.generals if g.player == player]

    def get_player_cells(self, player: int) -> List[Cell]:
        cells = []
        for row in self.board:
            for cell in row:
                if cell.player == player:
                    cells.append(cell)
        return cells

    def get_total_army(self, player: int) -> int:
        return sum(c.army for c in self.get_player_cells(player))

    def get_cell_count(self, player: int) -> int:
        return len(self.get_player_cells(player))

    def is_valid_pos(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS

    def get_cell(self, r: int, c: int) -> Optional[Cell]:
        if self.is_valid_pos(r, c):
            return self.board[r][c]
        return None


# ---- Core Game Logic ----

class GeneralsLogic:
    """Core game logic for Generals, separated from the environment wrapper."""

    @staticmethod
    def generate_board(seed: Optional[int] = None) -> List[List[Cell]]:
        """Generate a random 15x15 board."""
        if seed is not None:
            random.seed(seed)

        board = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                cell_type = CellType.PLAIN
                roll = random.random()
                if roll < BOG_PERCENT:
                    cell_type = CellType.BOG
                elif roll > 1 - MOUNTAIN_PERCENT:
                    cell_type = CellType.MOUNTAIN
                row.append(Cell(position=(r, c), type=cell_type))
            board.append(row)
        return board

    @staticmethod
    def init_generals(state: GeneralsState):
        """Initialize generals on the board."""
        gen_id = 0

        # Place main generals at opposite corners
        main_positions = [
            (BOARD_ROWS - 2, 1),  # Player 0: bottom-left area
            (1, BOARD_COLS - 2),  # Player 1: top-right area
        ]

        for player in range(2):
            # Main general
            pos = main_positions[player]
            main = General(
                id=gen_id, player=player, position=pos,
                produce_level=1, defense_level=1, mobility_level=1,
                is_main=True,
                skills_cd=[0, 0, 0, 0, 0],  # starts ready
            )
            gen_id += 1
            state.generals.append(main)
            state.board[pos[0]][pos[1]].general = main
            state.board[pos[0]][pos[1]].player = player
            state.board[pos[0]][pos[1]].army = 1

            # Place sub generals near main general
            offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            for off_idx in range(SUBGENERAL_NUM):
                placed = False
                for attempt in range(20):
                    dr, dc = offsets[off_idx % 4]
                    nr = pos[0] + dr * (1 + attempt // 4)
                    nc = pos[1] + dc * (1 + attempt // 4)
                    if state.is_valid_pos(nr, nc) and state.board[nr][nc].general is None:
                        sub = General(
                            id=gen_id, player=player, position=(nr, nc),
                            produce_level=1, defense_level=1, mobility_level=1,
                            is_sub=True,
                        )
                        gen_id += 1
                        state.generals.append(sub)
                        state.board[nr][nc].general = sub
                        state.board[nr][nc].player = player
                        state.board[nr][nc].army = 1
                        placed = True
                        break
                if not placed:
                    # Try random position
                    for _ in range(50):
                        nr = random.randint(0, BOARD_ROWS - 1)
                        nc = random.randint(0, BOARD_COLS - 1)
                        if state.board[nr][nc].general is None and state.board[nr][nc].type != CellType.MOUNTAIN:
                            sub = General(
                                id=gen_id, player=player, position=(nr, nc),
                                produce_level=1, defense_level=1, mobility_level=1,
                                is_sub=True,
                            )
                            gen_id += 1
                            state.generals.append(sub)
                            state.board[nr][nc].general = sub
                            state.board[nr][nc].player = player
                            state.board[nr][nc].army = 1
                            break

        # Place farmers randomly
        for player in range(2):
            for _ in range(FARMER_NUM):
                placed = False
                for _ in range(100):
                    r = random.randint(0, BOARD_ROWS - 1)
                    c = random.randint(0, BOARD_COLS - 1)
                    if state.board[r][c].general is None and state.board[r][c].type != CellType.MOUNTAIN:
                        farmer = General(
                            id=gen_id, player=player, position=(r, c),
                            produce_level=1, defense_level=1, mobility_level=0,
                        )
                        gen_id += 1
                        state.generals.append(farmer)
                        state.board[r][c].general = farmer
                        state.board[r][c].player = player
                        state.board[r][c].army = 1
                        placed = True
                        break

    @staticmethod
    def produce_armies(state: GeneralsState):
        """Produce armies for each player based on their generals and cells."""
        for player in range(2):
            # Base income from cells
            income = max(1, state.get_cell_count(player) // 2)

            # Additional income from generals' production
            for gen in state.get_player_generals(player):
                income += gen.produce_level

            state.coins[player] += income

    @staticmethod
    def execute_move_army(state: GeneralsState, player: int,
                          from_pos: Tuple[int, int],
                          direction: Direction,
                          amount: int) -> bool:
        """Move armies from one cell to an adjacent cell."""
        r, c = from_pos
        if not state.is_valid_pos(r, c):
            return False
        if state.board[r][c].player != player:
            return False

        dr, dc = DIRECTION_DELTA[direction]
        nr, nc = r + dr, c + dc
        if not state.is_valid_pos(nr, nc):
            return False

        from_cell = state.board[r][c]
        to_cell = state.board[nr][nc]

        # Check terrain
        tech = state.tech_levels[player]
        if to_cell.type == CellType.MOUNTAIN and tech["climb"] < 1:
            return False
        if to_cell.type == CellType.BOG and tech["immune"] < 1:
            amount = max(1, amount // 2)  # Halve in bog without immunity

        if amount > from_cell.army - 1:
            amount = from_cell.army - 1
        if amount <= 0:
            return False

        # Move armies
        from_cell.army -= amount

        if to_cell.player == player:
            to_cell.army += amount
        elif to_cell.player == -1:
            # Capture neutral
            to_cell.player = player
            to_cell.army = amount
        else:
            # Attack enemy
            if amount > to_cell.army:
                remaining = amount - to_cell.army
                # Kill enemy general
                if to_cell.general is not None:
                    state.generals.remove(to_cell.general)
                    to_cell.general = None
                to_cell.player = player
                to_cell.army = remaining
            elif amount < to_cell.army:
                to_cell.army -= amount
            else:
                to_cell.army = 1

        # Check if main general was killed
        main_alive = {0: False, 1: False}
        for gen in state.generals:
            if gen.is_main:
                main_alive[gen.player] = True

        for p in range(2):
            if not main_alive[p]:
                state.winner = 1 - p
                return True

        return True

    @staticmethod
    def execute_move_general(state: GeneralsState, player: int,
                             general_id: int,
                             new_pos: Tuple[int, int]) -> bool:
        """Move a general to a new position."""
        gen = None
        for g in state.generals:
            if g.id == general_id and g.player == player:
                gen = g
                break
        if gen is None:
            return False

        # Check if position is adjacent
        old_r, old_c = gen.position
        nr, nc = new_pos
        dist = abs(old_r - nr) + abs(old_c - nc)
        max_dist = gen.mobility_level
        if dist > max_dist:
            return False

        if not state.is_valid_pos(nr, nc):
            return False

        # Move general
        old_cell = state.board[old_r][old_c]
        new_cell = state.board[nr][nc]
        old_cell.general = None
        new_cell.general = gen
        gen.position = (nr, nc)

        return True

    @staticmethod
    def execute_upgrade(state: GeneralsState, player: int,
                        general_id: int, upgrade_type: int) -> bool:
        """Upgrade a general (1=production, 2=defense, 3=mobility)."""
        gen = None
        for g in state.generals:
            if g.id == general_id and g.player == player:
                gen = g
                break
        if gen is None:
            return False

        if upgrade_type == 1:  # Production
            max_level = 3 if gen.is_main or gen.is_sub else 3
            if gen.produce_level >= max_level:
                return False
            cost_key = f"{'farmer' if gen.is_farmer else 'sub'}_production_{gen.produce_level}"
            cost_map = {
                "farmer_production_1": 10, "farmer_production_2": 25, "farmer_production_3": 35,
                "sub_production_1": 40, "sub_production_2": 80,
            }
            cost = cost_map.get(cost_key, 50)
            if state.coins[player] < cost:
                return False
            state.coins[player] -= cost
            gen.produce_level += 1

        elif upgrade_type == 2:  # Defense
            max_def = 3 if gen.is_farmer else 2
            if gen.defense_level >= max_def:
                return False
            cost_key = f"{'farmer' if gen.is_farmer else 'sub'}_defense_{gen.defense_level}"
            cost_map = {
                "farmer_defense_1": 10, "farmer_defense_2": 15, "farmer_defense_3": 30,
                "sub_defense_1": 40, "sub_defense_2": 100,
            }
            cost = cost_map.get(cost_key, 50)
            if state.coins[player] < cost:
                return False
            state.coins[player] -= cost
            gen.defense_level += 1

        elif upgrade_type == 3:  # Mobility
            max_mob = 2 if gen.is_main or gen.is_sub else 1
            if gen.mobility_level >= max_mob:
                return False
            cost = UPGRADE_COSTS[f"general_movement_{gen.mobility_level}"]
            if state.coins[player] < cost:
                return False
            state.coins[player] -= cost
            gen.mobility_level += 1

        return True

    @staticmethod
    def execute_skill(state: GeneralsState, player: int,
                      general_id: int, skill_type: SkillType,
                      target: Tuple[int, int] = (-1, -1)) -> bool:
        """Use a general's skill."""
        gen = None
        for g in state.generals:
            if g.id == general_id and g.player == player:
                gen = g
                break
        if gen is None:
            return False

        skill_idx = int(skill_type)
        if skill_idx < 0 or skill_idx >= len(gen.skills_cd):
            return False
        if gen.skills_cd[skill_idx] > 0:
            return False

        # Cost
        skill_costs = [20, 15, 30, 30, 30]
        cost = skill_costs[skill_idx]
        if state.coins[player] < cost:
            return False

        state.coins[player] -= cost
        gen.skills_cd[skill_idx] = 5  # cooldown

        # Apply effects
        tr, tc = target
        if skill_type == SkillType.SURPRISE_ATTACK:
            if state.is_valid_pos(tr, tc):
                target_cell = state.board[tr][tc]
                target_cell.army = max(0, target_cell.army - 3)
        elif skill_type == SkillType.ROUT:
            if state.is_valid_pos(tr, tc):
                target_cell = state.board[tr][tc]
                target_cell.army = max(0, target_cell.army // 2)
        elif skill_type == SkillType.COMMAND:
            # Increase nearby army production temporarily
            pass
        elif skill_type == SkillType.DEFENCE:
            # Increase defense temporarily
            gen.skill_duration[1] = 3
        elif skill_type == SkillType.WEAKEN:
            if state.is_valid_pos(tr, tc):
                target_cell = state.board[tr][tc]
                target_cell.army = max(0, target_cell.army - 1)

        return True

    @staticmethod
    def execute_tech(state: GeneralsState, player: int, tech_id: int) -> bool:
        """Research a technology upgrade."""
        tech = state.tech_levels[player]

        if tech_id == 1:  # Army Mobility 1
            if tech["mobility"] >= 2:
                return False
            cost = TECH_COSTS[f"army_mobility_{tech['mobility'] + 1}"]
            if state.coins[player] < cost:
                return False
            state.coins[player] -= cost
            tech["mobility"] += 1
        elif tech_id == 2:  # Mountaineering
            if tech["climb"] >= 1:
                return False
            if state.coins[player] < TECH_COSTS["mountaineering"]:
                return False
            state.coins[player] -= TECH_COSTS["mountaineering"]
            tech["climb"] = 1
        elif tech_id == 3:  # Swamp Immunity
            if tech["immune"] >= 1:
                return False
            if state.coins[player] < TECH_COSTS["swamp_immunity"]:
                return False
            state.coins[player] -= TECH_COSTS["swamp_immunity"]
            tech["immune"] = 1
        elif tech_id == 4:  # Unlock super weapon
            if tech["unlock"] >= 1:
                return False
            if state.coins[player] < TECH_COSTS["unlock_super_weapon"]:
                return False
            state.coins[player] -= TECH_COSTS["unlock_super_weapon"]
            tech["unlock"] = 1
        else:
            return False

        return True

    @staticmethod
    def execute_superweapon(state: GeneralsState, player: int,
                            weapon: WeaponType,
                            pos: Tuple[int, int],
                            start_pos: Tuple[int, int] = (-1, -1)) -> bool:
        """Use a super weapon."""
        if state.tech_levels[player]["unlock"] < 1:
            return False
        if state.superweapon_cd[player] > 0:
            return False
        if state.coins[player] < SUPERWEAPON_COST:
            return False

        state.coins[player] -= SUPERWEAPON_COST
        state.superweapon_cd[player] = 10

        r, c = pos
        if weapon == WeaponType.NUCLEAR_BOOM:
            # Damage in 3x3 area
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if state.is_valid_pos(nr, nc):
                        cell = state.board[nr][nc]
                        if cell.player != player and cell.player != -1:
                            cell.army = max(0, cell.army - 5)
        elif weapon == WeaponType.ATTACK_ENHANCE:
            # Boost nearby armies
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if state.is_valid_pos(nr, nc):
                        cell = state.board[nr][nc]
                        if cell.player == player:
                            cell.army += 3
        elif weapon == WeaponType.TRANSMISSION:
            sr, sc = start_pos
            if state.is_valid_pos(r, c) and state.is_valid_pos(sr, sc):
                src = state.board[sr][sc]
                dst = state.board[r][c]
                if src.player == player:
                    transferred = src.army // 2
                    src.army -= transferred
                    if dst.player == player:
                        dst.army += transferred
        elif weapon == WeaponType.TIME_STOP:
            # Opponent skips next turn
            pass

        return True

    @staticmethod
    def update_cooldowns(state: GeneralsState):
        """Decrease all cooldowns."""
        for gen in state.generals:
            for i in range(len(gen.skills_cd)):
                if gen.skills_cd[i] > 0:
                    gen.skills_cd[i] -= 1
        for p in range(2):
            if state.superweapon_cd[p] > 0:
                state.superweapon_cd[p] -= 1

    @staticmethod
    def end_turn(state: GeneralsState):
        """Process end-of-turn effects."""
        GeneralsLogic.produce_armies(state)
        GeneralsLogic.update_cooldowns(state)

    @staticmethod
    def check_game_over(state: GeneralsState) -> int:
        """Check if the game is over. Returns winner ID or -1."""
        # Check if main generals are alive
        main_alive = {0: False, 1: False}
        for gen in state.generals:
            if gen.is_main:
                main_alive[gen.player] = True

        for p in range(2):
            if not main_alive[p]:
                return 1 - p

        # After max rounds, decide by army count
        if state.round >= MAX_ROUNDS:
            armies = [state.get_total_army(p) for p in range(2)]
            if armies[0] > armies[1]:
                return 0
            elif armies[1] > armies[0]:
                return 1
            else:
                cells = [state.get_cell_count(p) for p in range(2)]
                if cells[0] > cells[1]:
                    return 0
                elif cells[1] > cells[0]:
                    return 1
                else:
                    return 0 if state.coins[0] > state.coins[1] else 1

        return -1


# ---- Environment ----

class GeneralsEnv(BaseEnv):
    """
    Gym-compatible environment for the Generals game.

    Supports both DIRECT (in-process) and SUBPROCESS (stdio) modes.

    Action space: List of integer commands in the format:
        [command_type, args...]
        Command types: 1=move_army, 2=move_general, 3=upgrade,
                       4=skill, 5=tech, 6=superweapon, 7=call_sub,
                       8=end_turn, 9=surrender

    Observation: Dictionary containing full game state.
    """

    metadata = {"game": "generals", "players": 2, "mode": "turn-based"}
    game_name = "Generals"
    num_players = 2

    def __init__(self, mode: EnvMode = EnvMode.DIRECT, **kwargs):
        super().__init__(mode=mode, **kwargs)
        self._state: Optional[GeneralsState] = None
        self._logic = GeneralsLogic()
        self._action_history: List[List[int]] = []

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            type="list",
            description="List of [command_type, args...] integer commands. "
                       "Types: 1=move_army, 2=move_general, 3=upgrade, "
                       "4=skill, 5=tech, 6=superweapon, 7=call_sub, "
                       "8=end_turn, 9=surrender",
        )

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": ["board", "generals", "coins", "round", "current_player",
                     "tech_levels", "superweapon_cd", "winner"],
        }

    def _reset_direct(self, seed: Optional[int] = None) -> Observation:
        """Initialize a new Generals game."""
        if seed is not None:
            random.seed(seed)
            if HAS_NUMPY:
                np.random.seed(seed)

        self._state = GeneralsState()
        self._state.board = self._logic.generate_board(seed)
        self._logic.init_generals(self._state)
        self._current_player = 0
        self._round = 0
        self._done = False
        self._action_history = []

        return self._build_observation()

    def _step_direct(self, commands: List[List[int]]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute a list of commands for the current player.

        Args:
            commands: List of [cmd_type, args...] lists

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        player = self._current_player
        reward = 0.0
        info = {"errors": [], "actions_succeeded": 0}

        # Track state before for reward calculation
        prev_cells = self._state.get_cell_count(player)
        prev_army = self._state.get_total_army(player)

        for cmd in commands:
            if not cmd:
                continue

            cmd_type = cmd[0]
            success = False

            try:
                if cmd_type == CommandType.END_TURN:
                    success = True
                    break
                elif cmd_type == CommandType.SURRENDER:
                    self._state.winner = 1 - player
                    self._done = True
                    info["surrendered"] = True
                    reward = -10.0
                    break
                elif cmd_type == CommandType.MOVE_ARMY:
                    # [1, from_row, from_col, direction(1-4), amount]
                    r, c = cmd[1], cmd[2]
                    direction = Direction(cmd[3] - 1)
                    amount = cmd[4] if len(cmd) > 4 else 1
                    success = self._logic.execute_move_army(
                        self._state, player, (r, c), direction, amount
                    )
                elif cmd_type == CommandType.MOVE_GENERAL:
                    # [2, general_id, new_row, new_col]
                    gid, nr, nc = cmd[1], cmd[2], cmd[3]
                    success = self._logic.execute_move_general(
                        self._state, player, gid, (nr, nc)
                    )
                elif cmd_type == CommandType.UPGRADE:
                    # [3, general_id, upgrade_type]
                    gid, utype = cmd[1], cmd[2]
                    success = self._logic.execute_upgrade(
                        self._state, player, gid, utype
                    )
                elif cmd_type == CommandType.SKILL:
                    # [4, general_id, skill_type, target_r?, target_c?]
                    gid, stype = cmd[1], cmd[2]
                    target = (-1, -1)
                    if len(cmd) > 4:
                        target = (cmd[3], cmd[4])
                    success = self._logic.execute_skill(
                        self._state, player, gid, SkillType(stype - 1), target
                    )
                elif cmd_type == CommandType.TECH:
                    # [5, tech_id]
                    tid = cmd[1]
                    success = self._logic.execute_tech(self._state, player, tid)
                elif cmd_type == CommandType.SUPERWEAPON:
                    # [6, weapon_type, target_r, target_c, start_r?, start_c?]
                    wtype = WeaponType(cmd[1] - 1)
                    tr, tc = cmd[2], cmd[3]
                    start = (-1, -1)
                    if len(cmd) > 5:
                        start = (cmd[4], cmd[5])
                    success = self._logic.execute_superweapon(
                        self._state, player, wtype, (tr, tc), start
                    )
                elif cmd_type == CommandType.CALL_SUBGENERAL:
                    # [7, row, col]
                    pass  # Simplified - not implemented in demo
                else:
                    info["errors"].append(f"Unknown command type: {cmd_type}")
            except (IndexError, ValueError) as e:
                info["errors"].append(f"Invalid command {cmd}: {e}")
                success = False

            if success:
                info["actions_succeeded"] += 1
            else:
                info["errors"].append(f"Command failed: {cmd}")

        # End turn processing
        if not self._done:
            self._logic.end_turn(self._state)
            self._state.round += 1
            self._state.moves_made[player] += 1

            # Check game over
            winner = self._logic.check_game_over(self._state)
            if winner >= 0:
                self._state.winner = winner
                self._done = True
                if winner == player:
                    reward += 10.0
                else:
                    reward -= 10.0
                info["winner"] = winner

            # Switch players
            self._current_player = 1 - player

        # Calculate intermediate reward
        new_cells = self._state.get_cell_count(player)
        new_army = self._state.get_total_army(player)
        reward += (new_cells - prev_cells) * 0.5
        reward += (new_army - prev_army) * 0.1

        obs = self._build_observation()
        return obs, reward, self._done, info

    def _build_observation(self) -> Observation:
        """Build the observation from the current game state."""
        if self._state is None:
            return Observation()

        # Build a compact board representation
        board_repr = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                cell = self._state.board[r][c]
                row.append({
                    "type": int(cell.type),
                    "player": cell.player,
                    "army": cell.army,
                    "has_general": cell.general is not None,
                    "general_id": cell.general.id if cell.general else -1,
                    "general_is_main": cell.general.is_main if cell.general else False,
                })
            board_repr.append(row)

        generals_info = []
        for g in self._state.generals:
            generals_info.append({
                "id": g.id,
                "player": g.player,
                "position": list(g.position),
                "produce_level": g.produce_level,
                "defense_level": g.defense_level,
                "mobility_level": g.mobility_level,
                "is_main": g.is_main,
                "is_sub": g.is_sub,
                "skills_cd": g.skills_cd.copy(),
            })

        state_dict = {
            "board": board_repr,
            "generals": generals_info,
            "coins": self._state.coins.copy(),
            "round": self._state.round,
            "current_player": self._current_player,
            "tech_levels": [t.copy() for t in self._state.tech_levels],
            "superweapon_cd": self._state.superweapon_cd.copy(),
            "winner": self._state.winner,
        }

        return Observation(
            state=state_dict,
            player_id=self._current_player,
            round_num=self._state.round,
            done=self._done,
            info={"winner": self._state.winner},
        )

    def to_feature_vector(self, obs: Observation, player: int) -> Any:
        """
        Convert observation to a flat feature vector for RL.

        Returns a fixed-size array suitable for neural network input.
        Requires numpy for array operations.
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for to_feature_vector(). Install with: pip install numpy")

        state = obs.state
        player = obs.player_id

        features = []

        # Board features (15x15 x 7 channels)
        board = np.zeros((BOARD_ROWS, BOARD_COLS, 7), dtype=np.float32)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = state["board"][r][c]
                # Channel 0: terrain type (one-hot friendly via value/2)
                board[r, c, 0] = cell["type"] / 2.0
                # Channel 1: is my territory
                board[r, c, 1] = 1.0 if cell["player"] == player else 0.0
                # Channel 2: is enemy territory
                board[r, c, 2] = 1.0 if cell["player"] == 1 - player and cell["player"] >= 0 else 0.0
                # Channel 3: is neutral
                board[r, c, 3] = 1.0 if cell["player"] == -1 else 0.0
                # Channel 4: normalized army count (clamped)
                board[r, c, 4] = min(cell["army"], 20) / 20.0
                # Channel 5: has my general
                board[r, c, 5] = 1.0 if cell["has_general"] and cell["player"] == player else 0.0
                # Channel 6: has enemy general
                board[r, c, 6] = 1.0 if cell["has_general"] and cell["player"] == 1 - player else 0.0

        features.append(board.flatten())

        # Global features
        features.append(np.array([
            state["coins"][player] / 100.0,
            state["coins"][1 - player] / 100.0,
            state["round"] / MAX_ROUNDS,
            state["superweapon_cd"][player] / 10.0,
            state["superweapon_cd"][1 - player] / 10.0,
        ], dtype=np.float32))

        return np.concatenate(features)

    def get_legal_actions(self) -> List[List[int]]:
        """
        Get a list of legal actions for the current player.
        Returns simplified action templates.
        """
        if self._state is None:
            return []

        player = self._current_player
        actions = []

        # Move armies from each owned cell
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = self._state.board[r][c]
                if cell.player == player and cell.army > 1:
                    for d in range(4):
                        actions.append([1, r, c, d + 1, cell.army // 2])

        # Move generals
        for gen in self._state.get_player_generals(player):
            r, c = gen.position
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if self._state.is_valid_pos(nr, nc):
                    actions.append([2, gen.id, nr, nc])

        # Upgrade generals
        for gen in self._state.get_player_generals(player):
            for utype in [1, 2, 3]:
                actions.append([3, gen.id, utype])

        # Skills
        for gen in self._state.get_player_generals(player):
            if gen.is_main or gen.is_sub:
                for s in range(5):
                    if gen.skills_cd[s] == 0:
                        actions.append([4, gen.id, s + 1, gen.position[0], gen.position[1]])

        # Tech
        for tid in [1, 2, 3, 4]:
            actions.append([5, tid])

        # Superweapons
        if self._state.tech_levels[player]["unlock"] > 0:
            for wtype in [1, 2, 3, 4]:
                actions.append([6, wtype, player * 7 + 3, 7])

        # End turn (always available)
        actions.append([8])

        return actions

    def render(self, mode: str = "text") -> str:
        """Render the current game state as text."""
        if self._state is None:
            return "Game not started"

        lines = []
        lines.append(f"=== Generals - Round {self._state.round} ===")
        lines.append(f"Player 0 coins: {self._state.coins[0]} | Player 1 coins: {self._state.coins[1]}")
        lines.append(f"Current player: {self._current_player}")
        lines.append("")

        # Board with compact representation
        for r in range(BOARD_ROWS):
            row_str = ""
            for c in range(BOARD_COLS):
                cell = self._state.board[r][c]
                if cell.general and cell.general.is_main:
                    ch = "M" if cell.player == 0 else "m"
                elif cell.general:
                    ch = "G" if cell.player == 0 else "g"
                elif cell.player == 0:
                    ch = str(min(9, cell.army))
                elif cell.player == 1:
                    ch = chr(ord('a') + min(9, cell.army)) if cell.army <= 9 else '+'
                elif cell.type == CellType.MOUNTAIN:
                    ch = "#"
                elif cell.type == CellType.BOG:
                    ch = "~"
                else:
                    ch = "."
                row_str += ch
            lines.append(row_str)

        return "\n".join(lines)

    def close(self):
        """Clean up resources."""
        self._state = None

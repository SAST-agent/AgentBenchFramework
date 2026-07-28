"""
AntWAR2 game environment.

Wraps the vendored AntWAR2 engine (env._antwar2) behind the framework's
BaseEnv interface so it plays identically to GeneralsEnv in Match / Arena /
training loops. The game is a simultaneous-move 2-player tower-defense match:
both sides commit operations for a round, then the engine resolves the round
together. To stay compatible with the framework's alternating step()/Match
contract, DIRECT mode buffers the first player's operations and only advances
the round once the second player has moved (two steps == one game round).

An optional opponent callable switches the env into single-agent mode for
self-play RL: each step plays the controlled side plus the opponent and
resolves one full round, keeping player_id on the controlled side.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from agentbench_frame.env.base import BaseEnv, EnvMode, ActionSpace, Observation

from agentbench_frame.env._antwar2.backend.state import PythonBackendState, create_python_backend_state
from agentbench_frame.env._antwar2.backend.model import Operation
from agentbench_frame.env._antwar2.utils.constants import (
    MAP_SIZE,
    MAX_ROUND,
    MAX_ACTIONS,
    PLAYER_BASES,
    OperationType,
)
from agentbench_frame.env._antwar2.utils.features import FeatureExtractor
from agentbench_frame.env._antwar2.utils.actions import ActionBundle, ActionCatalog

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

GAME_ID = "30_antwar2"
_OP_NONE_MARKER = 8


def _to_operations(action: Any) -> List[Operation]:
    if action is None:
        return []
    out: List[Operation] = []
    if isinstance(action, Operation):
        return [action]
    if isinstance(action, dict):
        action = [action]
    if not isinstance(action, (list, tuple)):
        return out
    for item in action:
        if item is None:
            continue
        if isinstance(item, Operation):
            out.append(item)
            continue
        if isinstance(item, ActionBundle):
            out.extend(item.operations)
            continue
        if isinstance(item, dict):
            op_type = item.get("op_type", item.get("type"))
            if op_type is None:
                continue
            out.append(Operation(OperationType(int(op_type)),
                                 int(item.get("arg0", -1)),
                                 int(item.get("arg1", -1))))
            continue
        if isinstance(item, (list, tuple)):
            try:
                toks = [int(t) for t in item]
            except (TypeError, ValueError):
                continue
            if not toks:
                continue
            if toks[0] == _OP_NONE_MARKER:
                continue
            if len(toks) == 1:
                out.append(Operation(OperationType(toks[0])))
            elif len(toks) == 2:
                out.append(Operation(OperationType(toks[0]), toks[1]))
            else:
                out.append(Operation(OperationType(toks[0]), toks[1], toks[2]))
    return out


class AntWar2Env(BaseEnv):
    metadata = {"game": GAME_ID, "players": 2, "mode": "simultaneous-turn"}
    game_name = "AntWAR2"
    num_players = 2

    def __init__(self,
                 mode: EnvMode = EnvMode.DIRECT,
                 opponent: Optional[Callable[[PythonBackendState, int], List[Operation]]] = None,
                 controlled_player: int = 0,
                 cold_handle_rule_illegal: bool = False,
                 max_actions: int = MAX_ACTIONS,
                 **kwargs):
        super().__init__(mode=mode, **kwargs)
        self._opponent = opponent
        self._controlled_player = controlled_player
        self._cold_handle_rule_illegal = cold_handle_rule_illegal
        self._state: Optional[PythonBackendState] = None
        self._feature = FeatureExtractor(max_actions=max_actions)
        self._catalog = ActionCatalog(max_actions=max_actions, feature_extractor=self._feature)
        self._pending: Dict[int, List[Operation]] = {0: [], 1: []}
        self._last_value: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._last_round: int = 0
        self._seed: Optional[int] = None

    @property
    def backend_state(self) -> Optional[PythonBackendState]:
        return self._state

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            type="list",
            description=(
                "List of operations for one round. Each operation is "
                "[op_type, arg0, arg1]. Types: 11=build_tower(x,y), "
                "12=upgrade_tower(id,target_type), 13=downgrade_tower(id), "
                "21=lightning(x,y), 22=emp(x,y), 23=deflector(x,y), "
                "24=evasion(x,y), 31=upgrade_generation_speed, "
                "32=upgrade_generated_ant. An empty list means hold."
            ),
        )

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": ["round", "current_player", "winner", "terminal", "coins",
                     "bases", "towers", "ants", "weapon_cooldowns",
                     "active_effects", "die_count", "old_count"],
        }

    def _reset_direct(self, seed: Optional[int] = None) -> Observation:
        self._seed = seed
        self._state = create_python_backend_state(
            seed=seed if seed is not None else 0,
            cold_handle_rule_illegal=self._cold_handle_rule_illegal,
        )
        self._current_player = self._controlled_player if self._opponent else 0
        self._round = 0
        self._done = False
        self._pending = {0: [], 1: []}
        self._last_round = 0
        self._last_value = {0: 0.0, 1: 0.0}
        if self._state is not None:
            self._last_value[0] = self._feature.evaluate(self._state, 0)
            self._last_value[1] = self._feature.evaluate(self._state, 1)
        return self._build_observation()

    def _step_direct(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        player = self._current_player
        operations = _to_operations(action)

        if self._opponent is not None:
            return self._step_single_agent(player, operations)

        self._pending[player] = operations
        info: Dict[str, Any] = {"round_resolved": False, "player": player}

        if player == 0:
            self._current_player = 1
            return self._build_observation(), 0.0, False, info

        return self._resolve_round(perspective=0)

    def _step_single_agent(self, player: int,
                           operations: List[Operation]
                           ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        me = self._controlled_player
        opp = 1 - me
        my_ops = operations if player == me else self._opponent(self._state, me)  # type: ignore[misc]
        opp_ops = self._opponent(self._state, opp)  # type: ignore[misc]
        obs, reward, done, info = self._resolve_turn(my_ops, opp_ops, perspective=me)
        self._current_player = me
        return obs, reward, done, info

    def _resolve_round(self, perspective: int
                       ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        ops0 = self._pending[0]
        ops1 = self._pending[1]
        obs, reward, done, info = self._resolve_turn(ops0, ops1, perspective=perspective)
        self._pending = {0: [], 1: []}
        self._current_player = 0
        info["round_resolved"] = True
        return obs, reward, done, info

    def _resolve_turn(self, ops0: List[Operation], ops1: List[Operation],
                      perspective: int
                      ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        prev_value = self._last_value[perspective]
        self._state.resolve_turn(ops0, ops1)  # type: ignore[union-attr]
        new_round = self._state.round_index  # type: ignore[union-attr]
        rounds_advanced = max(1, new_round - self._last_round)
        self._last_round = new_round
        self._round = new_round

        done = bool(self._state.terminal)  # type: ignore[union-attr]
        winner = self._state.winner  # type: ignore[union-attr]
        self._done = done

        self._last_value[0] = self._feature.evaluate(self._state, 0)  # type: ignore[arg-type]
        self._last_value[1] = self._feature.evaluate(self._state, 1)  # type: ignore[arg-type]
        new_value = self._last_value[perspective]
        reward = float(new_value - prev_value) / rounds_advanced
        if done:
            if winner == perspective:
                reward += 1000.0
            elif winner == 1 - perspective:
                reward -= 1000.0

        info = {
            "round_resolved": True,
            "winner": -1 if winner is None else int(winner),
            "rounds_advanced": rounds_advanced,
        }
        return self._build_observation(), reward, done, info

    def _build_observation(self) -> Observation:
        if self._state is None:
            return Observation()
        public = self._state.to_public_round_state()  # type: ignore[attr-defined]
        winner = -1 if self._state.winner is None else int(self._state.winner)  # type: ignore[attr-defined]
        state_dict = {
            "game": GAME_ID,
            "round": self._state.round_index,  # type: ignore[attr-defined]
            "current_player": self._current_player,
            "winner": winner,
            "terminal": bool(self._state.terminal),  # type: ignore[attr-defined]
            "coins": list(self._state.coins),  # type: ignore[attr-defined]
            "bases": [
                {
                    "player": b.player, "x": b.x, "y": b.y, "hp": b.hp,
                    "generation_level": b.generation_level, "ant_level": b.ant_level,
                }
                for b in self._state.bases  # type: ignore[attr-defined]
            ],
            "towers": [list(t) for t in public.towers],
            "ants": [list(a) for a in public.ants],
            "weapon_cooldowns": [list(w) for w in (public.weapon_cooldowns or ())],
            "active_effects": [list(e) for e in (public.active_effects or [])],
            "die_count": list(self._state.die_count),  # type: ignore[attr-defined]
            "old_count": list(self._state.old_count),  # type: ignore[attr-defined]
            "map_size": MAP_SIZE,
            "player_bases": [list(b) for b in PLAYER_BASES],
            "max_round": MAX_ROUND,
        }
        info = {
            "winner": winner,
            "_backend_state": self._state,
        }
        return Observation(
            state=state_dict,
            player_id=self._current_player,
            round_num=self._state.round_index,  # type: ignore[attr-defined]
            done=self._done,
            info=info,
        )

    def get_backend_state(self) -> Optional[PythonBackendState]:
        return self._state

    def list_action_bundles(self) -> List[ActionBundle]:
        if self._state is None:
            return [ActionBundle(name="hold")]
        bundles = self._catalog.build(self._state, self._current_player)
        if not bundles:
            return [ActionBundle(name="hold")]
        return bundles

    def get_legal_actions(self) -> List[List[int]]:
        bundles = self.list_action_bundles()
        out: List[List[int]] = []
        for bundle in bundles:
            tokens = [op.to_protocol_tokens() for op in bundle.operations]
            flat: List[int] = []
            for tok in tokens:
                flat.extend(tok)
            out.append(flat if flat else [_OP_NONE_MARKER])
        if not out:
            out.append([_OP_NONE_MARKER])
        return out

    def to_feature_vector(self, obs: Observation, player: int) -> Any:
        if not HAS_NUMPY:
            raise ImportError("numpy is required for AntWar2Env.to_feature_vector()")
        state = obs.info.get("_backend_state") if obs.info else None
        if state is None:
            state = self._state
        if state is None:
            raise RuntimeError("No backend state available")
        bundles = self._catalog.build(state, player)
        mask = self._catalog.action_mask(bundles)
        board = self._feature.encode_board(state, player)
        stats = self._feature.encode_stats(state, player)
        return {
            "board": board,
            "stats": stats,
            "action_mask": mask,
            "bundles": bundles,
        }

    def render(self, mode: str = "text") -> str:
        if self._state is None:
            return "Game not started"
        st = self._state
        lines = [
            "=== AntWAR2 - Round {} ===".format(st.round_index),
            "P0 coins={} base_hp={} | P1 coins={} base_hp={}".format(
                st.coins[0], st.bases[0].hp, st.coins[1], st.bases[1].hp),
            "Current player: {}".format(self._current_player),
            "Towers: {} | Ants: {}".format(len(st.towers), len(st.ants)),
        ]
        if st.terminal:
            w = "draw" if st.winner is None else "player {}".format(st.winner)
            lines.append("GAME OVER: winner={}".format(w))
        return "\n".join(lines)

    def close(self):
        self._state = None
        self._pending = {0: [], 1: []}

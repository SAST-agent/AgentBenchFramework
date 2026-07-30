"""
SnakeGoEnv -- adapts the bit-exact SnakeGo engine into BaseEnv.

Turn structure: SnakeGo is phase-based (P0 operates ALL their snakes, then
P1 operates ALL their snakes, then round++). The framework Match/runner
assumes alternating single-action steps. This adapter flattens the phase
structure: each step() = ONE snake operation. The engine state machine
(do_operation -> find_next_snake -> round_preprocess) handles phase
transitions, and we expose eng.current_player as obs.player_id.

Subprocess backends: compiled human AIs communicate via the adk socket
protocol. The env manages init/ack/relay internally.
"""
import os
import sys

from agentbench_frame.env.base import BaseEnv, EnvMode, ActionSpace, Observation

_SNAKEGO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _SNAKEGO_DIR not in sys.path:
    sys.path.insert(0, _SNAKEGO_DIR)

from snakego.engine import Engine
from snakego.host import generate_items, build_init_bytes, LENGTH, WIDTH, MAX_ROUND


class SnakeGoEnv(BaseEnv):
    """Gym-style wrapper around the SnakeGo Engine."""

    metadata = {"render_modes": ["text"]}

    def __init__(self, mode=EnvMode.DIRECT, subprocess_players=None, **kwargs):
        super().__init__(mode=mode, **kwargs)
        self._subprocs = subprocess_players or {}
        self.eng = None
        self.items = []
        self.ops = []
        self._seed = 0

    @property
    def action_space(self):
        return ActionSpace(type="discrete", n=6, description="1=R 2=U 3=L 4=D 5=RAILGUN 6=SPLIT")

    @property
    def observation_space(self):
        return {"type": "dict", "fields": ["engine", "scores", "winner"]}

    @property
    def num_players(self):
        return 2

    @property
    def game_name(self):
        return "26_snakego"

    def _reset_direct(self, seed=None):
        self._seed = seed or 0
        self.items = generate_items(self._seed)
        self.eng = Engine(LENGTH, WIDTH, MAX_ROUND, self.items)
        self.ops = []
        for pid, sp in self._subprocs.items():
            sp.start(build_init_bytes(self.items, pid))
        return self._make_obs()

    def _step_direct(self, action):
        cur = self.eng.current_player
        moved_snake_id = self.eng.current_snake_id
        if not self.eng.alive_player():
            action = 0
        running = self.eng.do_operation(action)
        self.ops.append([self.eng.current_round, cur, moved_snake_id, action])
        self._relay(cur, action)
        while running and not self.eng.alive_player():
            dead = self.eng.current_player
            running = self.eng.do_operation(0)
            self.ops.append([self.eng.current_round, dead, -1, 0])
            self._relay(dead, 0)
        done = not running
        scores = self.eng.score()
        winner = 0 if scores[0] >= scores[1] else 1
        if done:
            for pid, sp in self._subprocs.items():
                try:
                    sp.send_gameover(0, winner, scores)
                except Exception:
                    pass
        return (self._make_obs(done, winner, scores), 0.0, done,
                {"winner": winner, "scores": scores})

    def _relay(self, player, op):
        if player in self._subprocs:
            try:
                self._subprocs[player].send_ack(op)
            except Exception:
                pass
        other = 1 - player
        if other in self._subprocs:
            try:
                self._subprocs[other].send_op(op)
            except Exception:
                pass

    def _make_obs(self, done=False, winner=None, scores=None):
        return Observation(
            state={
                "engine": self.eng,
                "scores": scores or (self.eng.score() if self.eng else [0, 0]),
                "winner": winner,
            },
            player_id=self.eng.current_player if self.eng else 0,
            round_num=self.eng.current_round if self.eng else 0,
            done=done,
            info={},
        )

    def replay_dict(self, names=None):
        scores = self.eng.score() if self.eng else [0, 0]
        return {
            "winner": 0 if scores[0] >= scores[1] else 1,
            "scores": scores,
            "moves": len(self.ops),
            "rounds": self.eng.current_round if self.eng else 0,
            "names": names or ["p0", "p1"],
            "items": [{"x": it.x, "y": it.y, "time": it.time, "type": it.type, "param": it.param} for it in self.items],
            "ops": self.ops,
        }

    def close(self):
        for sp in self._subprocs.values():
            try:
                sp.close()
            except Exception:
                pass

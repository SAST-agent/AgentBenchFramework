"""Official-engine adapter.

Wraps the UNMODIFIED gamecode_logic judge (operate.py Controller +
infrastructure.py Context + spawn.py + result.py) so every move, seal,
railgun, split, and score goes through the exact official Python code.
Our strategies speak the same engine API as engine.py, so swapping is
transparent -- zero port drift.

Only two read-only additions were made to the official classes:
  Snake.coord_list / Snake.length / Snake.railgun_item_id  (infrastructure.py)
  Item.eaten / Item.expired                                (spawn.py)
Neither changes any game logic.
"""
import os
import sys
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.join(_HERE, "official_logic", "gamecode_logic")
if _OFFICIAL not in sys.path:
    sys.path.insert(0, _OFFICIAL)

from logic.infrastructure import Context, Snake, Operation
from logic.operate import Controller
from logic.result import settle_round
from logic.spawn import GameConfig, Item
from logic.communication import PlayerError
from logic.constants import ITEM_EXPIRE_TIME

LENGTH = 16
WIDTH = 16
MAX_ROUND = 512


class OfficialEngine:
    """Drop-in replacement for engine.Engine backed by the official judge."""

    def __init__(self, length=LENGTH, width=WIDTH, max_round=MAX_ROUND,
                 items=None, seed=None):
        Snake.snake_num = 0
        Item.item_num = 0

        config = GameConfig()
        config.length = length
        config.width = width
        config.max_round = max_round
        config.random_seed = seed if seed is not None else int(random.random() * 1e6)

        self._ctx = Context(config)
        self._ctx.turn = 1
        self._ctrl = Controller(self._ctx)

        self._running = True
        self._scores = [0, 0]
        self._error = None

        self._ctrl.round_preprocess(ITEM_EXPIRE_TIME)
        self._ctrl.round_init()
        self._advance()

    def _advance(self):
        while self._running:
            if self._ctrl.next_snake != -1:
                return True
            self._ctrl.next_player()
            if self._ctrl.player == 0:
                self._running, self._scores = settle_round(self._ctx)
                if not self._running:
                    return False
                self._ctrl.round_preprocess(ITEM_EXPIRE_TIME)
            self._ctrl.round_init()
        return False

    def do_operation(self, op_type):
        if not self._running:
            return False
        if not self._advance():
            return False

        op = Operation(type=int(op_type), snake=-1, direction=-1, item_id=-1)
        try:
            self._ctrl.apply(op)
        except PlayerError as e:
            self._running = False
            self._error = e
            self._scores = [0, 0]
            self._scores[e.player] = -100
            return False

        self._advance()
        return self._running

    @property
    def current_player(self):
        return self._ctrl.player

    @property
    def current_round(self):
        return self._ctx.turn

    @property
    def current_snake_id(self):
        ns = self._ctrl.next_snake
        if ns == -1 or ns >= len(self._ctrl.current_snake_list):
            return -1
        return self._ctrl.current_snake_list[ns][0].id

    @property
    def length(self):
        return self._ctx.game_map.length

    @property
    def width(self):
        return self._ctx.game_map.width

    @property
    def max_round(self):
        return self._ctx.max_round

    @property
    def wall_map(self):
        return self._ctx.game_map.wall_map

    @property
    def snake_map(self):
        return self._ctx.game_map.snake_map

    @property
    def item_map(self):
        return self._ctx.game_map.item_map

    @property
    def item_list(self):
        return self._ctx.game_map.item_list

    def current_snake(self):
        ns = self._ctrl.next_snake
        if ns == -1 or not hasattr(self._ctrl, 'current_snake_list'):
            return None
        cl = self._ctrl.current_snake_list
        if ns >= len(cl):
            return None
        return cl[ns][0]

    def my_snakes(self):
        return [s for s in self._ctx.snake_list if s.camp == self.current_player]

    def opponents_snakes(self):
        return [s for s in self._ctx.snake_list if s.camp != self.current_player]

    @property
    def snake_list_0(self):
        return [s for s in self._ctx.snake_list if s.camp == 0]

    @property
    def snake_list_1(self):
        return [s for s in self._ctx.snake_list if s.camp == 1]

    def alive_player(self):
        return any(s.camp == self.current_player for s in self._ctx.snake_list)

    def score(self):
        if self._error:
            return list(self._scores)
        _, s = settle_round(self._ctx)
        return s

    def find_next_snake(self):
        return not self._advance()

    def round_preprocess(self):
        self._ctrl.round_preprocess(ITEM_EXPIRE_TIME)
        return True

    def snapshot(self):
        return {
            "round": self.current_round,
            "player": self.current_player,
            "snake_id": self.current_snake_id,
            "wall": [row[:] for row in self.wall_map],
            "snake": [row[:] for row in self.snake_map],
            "item": [row[:] for row in self.item_map],
            "snakes": {
                "p0": [[[c[0], c[1]] for c in s.coor_list]
                       + [[s.length_bank, s.railgun_item_id]]
                       for s in self.snake_list_0],
                "p1": [[[c[0], c[1]] for c in s.coor_list]
                       + [[s.length_bank, s.railgun_item_id]]
                       for s in self.snake_list_1],
            },
            "score": self.score(),
        }

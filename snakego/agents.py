"""Built-in SnakeGo agents.

* ``RandomAgent``   — picks a uniformly-random legal action.
* ``GreedyAgent``   — Python port of the official sample AI (main.cpp):
  first snake chases items / splits / railguns; other snakes prefer to
  solidify or follow toward their own tail.

Any agent is a callable object with a ``name`` and an ``act(obs, snake_id)``
method returning an action 1-6.  Subclass :class:`BaseAgent` to add your own.
"""
from __future__ import annotations

import random
from typing import Any, List, Optional

from .env import (
    ACT_MOVE_BASE, ACT_RAILGUN, ACT_SPLIT, DX, DY, IllegalAction,
    SnakeGoGame,
)


class BaseAgent:
    """Subclass and implement :meth:`act`."""

    name: str = "base"

    def act(self, obs: dict, snake_id: int, game: SnakeGoGame) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


# --------------------------------------------------------------------------- #


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, obs: dict, snake_id: int, game: SnakeGoGame) -> int:
        acts = game.valid_actions()
        return self.rng.choice(acts)


# --------------------------------------------------------------------------- #


class GreedyAgent(BaseAgent):
    """Port of the Saiblo sample AI.

    Strategy:
      * fire railgun immediately if held;
      * first snake: split when long enough & population < 4, else head
        toward the nearest reachable item;
      * other snakes: prefer to solidify (close a loop), else move toward
        own tail to stay compact;
      * fallback: any safe move, then any solidify move.
    """

    name = "greedy"

    def act(self, obs: dict, snake_id: int, game: SnakeGoGame) -> int:
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        coor = snake.coor_list
        head = coor[0]

        # 1. railgun
        if ACT_RAILGUN in acts and snake.has_railgun():
            return ACT_RAILGUN

        my_snakes = [s for s in obs["snakes"] if s["camp"] == snake.camp]
        is_first = (len(my_snakes) > 0 and my_snakes[0]["id"] == snake_id)

        # 2. split
        if is_first and ACT_SPLIT in acts and snake.length >= 10 \
                and len(my_snakes) < 4:
            return ACT_SPLIT

        # classify moves
        L, W = obs["length"], obs["width"]
        safe: List[int] = []      # normal move into empty space
        solidify: List[int] = []  # move onto own body (closes loop)
        for d in range(4):
            a = ACT_MOVE_BASE + d
            if a not in acts:
                continue
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if not (0 <= nx < obs["length"] and 0 <= ny < obs["width"]):
                continue
            occ = obs["snake_map"][nx][ny]
            if occ == snake_id:
                solidify.append(a)
            else:
                safe.append(a)

        # 3. first snake → chase items
        if is_first:
            best = self._toward_item(obs, snake_id, head, safe)
            if best is not None:
                return best
        # 4. other snakes → solidify or tail-follow
        else:
            if solidify:
                return solidify[0]
            tail = coor[-1]
            best = self._toward(head, tail, safe)
            if best is not None:
                return best

        # 5. fallback
        if safe:
            return safe[0]
        if solidify:
            return solidify[0]
        return acts[0]

    def _toward_item(self, obs, snake_id, head, safe):
        cur_round = obs["turn"]
        best_action = None
        best_dist = 10 ** 9
        for it in obs["items"]:
            if it["time"] > cur_round:
                continue
            if it["time"] + 16 <= cur_round:
                continue
            dist = abs(it["x"] - head[0]) + abs(it["y"] - head[1])
            if it["time"] <= cur_round + dist and it["time"] + 16 > cur_round + dist:
                for a in safe:
                    d = a - ACT_MOVE_BASE
                    nx, ny = head[0] + DX[d], head[1] + DY[d]
                    nd = abs(it["x"] - nx) + abs(it["y"] - ny)
                    if nd <= dist and nd < best_dist:
                        best_dist = nd
                        best_action = a
        return best_action

    def _toward(self, src, dst, safe):
        dist = abs(dst[0] - src[0]) + abs(dst[1] - src[1])
        best = None
        best_d = dist
        for a in safe:
            d = a - ACT_MOVE_BASE
            nx, ny = src[0] + DX[d], src[1] + DY[d]
            nd = abs(dst[0] - nx) + abs(dst[1] - ny)
            if nd <= best_d:
                best_d = nd
                best = a
        return best



# --------------------------------------------------------------------------- #
# Human-written strategy variants (population pool)
# --------------------------------------------------------------------------- #


class WallHuggerAgent(BaseAgent):
    """Always move toward the nearest map edge."""
    name = "wall_hugger"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        head = snake.coor_list[0]
        L, W = obs["length"], obs["width"]
        best_a = None
        best_d = 99
        for a in acts:
            if a > 4:
                continue
            d = a - ACT_MOVE_BASE
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if not (0 <= nx < L and 0 <= ny < W):
                continue
            if obs["snake_map"][nx][ny] == snake_id:
                continue
            wall_d = min(nx, L - 1 - nx, ny, W - 1 - ny)
            if wall_d < best_d:
                best_d = wall_d
                best_a = a
        if best_a is not None:
            return best_a
        return acts[0]


class CenterSeekerAgent(BaseAgent):
    """Always move toward the center of the map."""
    name = "center_seeker"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        head = snake.coor_list[0]
        L, W = obs["length"], obs["width"]
        cx, cy = L // 2, W // 2
        best_a = None
        best_d = abs(head[0] - cx) + abs(head[1] - cy)
        for a in acts:
            if a > 4:
                continue
            d = a - ACT_MOVE_BASE
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if not (0 <= nx < L and 0 <= ny < W):
                continue
            if obs["snake_map"][nx][ny] == snake_id:
                continue
            nd = abs(nx - cx) + abs(ny - cy)
            if nd < best_d:
                best_d = nd
                best_a = a
        if best_a is not None:
            return best_a
        return acts[0]


class AggressiveChaserAgent(BaseAgent):
    """Chase the nearest enemy snake head, fire railgun on sight."""
    name = "aggressive_chaser"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        coor = snake.coor_list
        head = coor[0]
        if ACT_RAILGUN in acts and snake.has_railgun() and len(coor) > 3:
            return ACT_RAILGUN
        my_count = sum(1 for s in obs["snakes"] if s["camp"] == snake.camp)
        if ACT_SPLIT in acts and len(coor) >= 12 and my_count < 3:
            return ACT_SPLIT
        enemies = [s for s in obs["snakes"] if s["camp"] != snake.camp]
        if enemies:
            target = min(enemies, key=lambda s: abs(s["coor_list"][0][0] - head[0])
                         + abs(s["coor_list"][0][1] - head[1]))
            tx, ty = target["coor_list"][0]
            best_a = None
            best_d = abs(tx - head[0]) + abs(ty - head[1])
            for a in acts:
                if a > 4:
                    continue
                d = a - ACT_MOVE_BASE
                nx, ny = head[0] + DX[d], head[1] + DY[d]
                if not (0 <= nx < obs["length"] and 0 <= ny < obs["width"]):
                    continue
                if obs["snake_map"][nx][ny] == snake_id:
                    continue
                nd = abs(tx - nx) + abs(ty - ny)
                if nd < best_d:
                    best_d = nd
                    best_a = a
            if best_a is not None:
                return best_a
        for a in acts:
            if a > 4:
                continue
            d = a - ACT_MOVE_BASE
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if 0 <= nx < obs["length"] and 0 <= ny < obs["width"]:
                if obs["snake_map"][nx][ny] != snake_id:
                    return a
        return acts[0]


class TerritoryMaximizerAgent(BaseAgent):
    """Maximize enclosed territory: solidify early and often."""
    name = "territory_max"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        coor = snake.coor_list
        head = coor[0]
        if ACT_RAILGUN in acts and snake.has_railgun() and len(coor) > 3:
            return ACT_RAILGUN
        my_count = sum(1 for s in obs["snakes"] if s["camp"] == snake.camp)
        if ACT_SPLIT in acts and len(coor) >= 10 and my_count < 4:
            return ACT_SPLIT
        safe = []
        solidify = []
        for d in range(4):
            a = ACT_MOVE_BASE + d
            if a not in acts:
                continue
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if 0 <= nx < obs["length"] and 0 <= ny < obs["width"]:
                if obs["snake_map"][nx][ny] == snake_id:
                    solidify.append(a)
                else:
                    safe.append(a)
        if len(coor) >= 6 and solidify:
            return solidify[0]
        best_a = None
        best_d = 10 ** 9
        for it in obs["items"]:
            dist = abs(it["x"] - head[0]) + abs(it["y"] - head[1])
            if dist < best_d:
                for a in safe:
                    d2 = a - ACT_MOVE_BASE
                    nx, ny = head[0] + DX[d2], head[1] + DY[d2]
                    nd = abs(it["x"] - nx) + abs(it["y"] - ny)
                    if nd < best_d:
                        best_d = nd
                        best_a = a
        if best_a is not None:
            return best_a
        if solidify:
            return solidify[0]
        if safe:
            return safe[0]
        return acts[0]


class ItemHoarderAgent(BaseAgent):
    """Aggressively collect items, fire railgun opportunistically."""
    name = "item_hoarder"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        coor = snake.coor_list
        head = coor[0]
        if ACT_RAILGUN in acts and snake.has_railgun() and len(coor) > 2:
            return ACT_RAILGUN
        my_count = sum(1 for s in obs["snakes"] if s["camp"] == snake.camp)
        if ACT_SPLIT in acts and len(coor) >= 14 and my_count < 3:
            return ACT_SPLIT
        safe = []
        for d in range(4):
            a = ACT_MOVE_BASE + d
            if a not in acts:
                continue
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if 0 <= nx < obs["length"] and 0 <= ny < obs["width"]:
                if obs["snake_map"][nx][ny] != snake_id:
                    safe.append(a)
        best_a = None
        best_d = 10 ** 9
        for it in obs["items"]:
            if it["type"] != 0:
                continue
            dist = abs(it["x"] - head[0]) + abs(it["y"] - head[1])
            if dist < best_d:
                for a in safe:
                    d2 = a - ACT_MOVE_BASE
                    nx, ny = head[0] + DX[d2], head[1] + DY[d2]
                    nd = abs(it["x"] - nx) + abs(it["y"] - ny)
                    if nd < best_d:
                        best_d = nd
                        best_a = a
        if best_a is not None:
            return best_a
        if safe:
            return safe[0]
        return acts[0]



BUILTIN_AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "wall_hugger": WallHuggerAgent,
    "center_seeker": CenterSeekerAgent,
    "aggressive_chaser": AggressiveChaserAgent,
    "territory_max": TerritoryMaximizerAgent,
    "item_hoarder": ItemHoarderAgent,
}

__all__ = ["BaseAgent", "RandomAgent", "GreedyAgent",
           "WallHuggerAgent", "CenterSeekerAgent", "AggressiveChaserAgent",
           "TerritoryMaximizerAgent", "ItemHoarderAgent", "BUILTIN_AGENTS"]

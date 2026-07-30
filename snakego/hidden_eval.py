"""Hidden evaluation: test the final strategy against unseen opponents.

The iteration loop only sees the "training" opponents.  After the best
version is selected, we run a *blind* evaluation against a held-out set
of opponents the coding agent never encountered.  This simulates a
tournament / deployment scenario and prevents overfitting to the
training pool.

    python -m snakego.hidden_eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .agents import BaseAgent, GreedyAgent, RandomAgent
from .iteration import evaluate_strategy
from .versioning import VersionStore


# --------------------------------------------------------------------------- #
# hidden opponent pool -- NEVER used during iteration
# --------------------------------------------------------------------------- #


class WallHuggerAgent(BaseAgent):
    """Always tries to move along the map edge (hugs walls)."""
    name = "wall_hugger"

    def act(self, obs, snake_id, game):
        from .env import DX, DY, ACT_MOVE_BASE
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        head = snake.coor_list[0]
        L, W = obs["length"], obs["width"]
        # prefer moves that keep us near a wall
        best_a = None
        best_wall_dist = 99
        for a in acts:
            if a > 4:
                continue
            d = a - ACT_MOVE_BASE
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if not (0 <= nx < L and 0 <= ny < W):
                continue
            if obs["snake_map"][nx][ny] == snake_id:
                continue
            wall_dist = min(nx, L - 1 - nx, ny, W - 1 - ny)
            if wall_dist < best_wall_dist:
                best_wall_dist = wall_dist
                best_a = a
        if best_a is not None:
            return best_a
        return acts[0]


class CenterSeekerAgent(BaseAgent):
    """Always moves toward the center of the map."""
    name = "center_seeker"

    def act(self, obs, snake_id, game):
        from .env import DX, DY, ACT_MOVE_BASE
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        head = snake.coor_list[0]
        L, W = obs["length"], obs["width"]
        cx, cy = L // 2, W // 2
        best_a = None
        best_dist = abs(head[0] - cx) + abs(head[1] - cy)
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
            if nd < best_dist:
                best_dist = nd
                best_a = a
        if best_a is not None:
            return best_a
        return acts[0]


def get_hidden_opponents() -> Dict[str, BaseAgent]:
    """Return the secret opponent pool for blind evaluation."""
    return {
        "greedy_builtin": GreedyAgent(),
        "wall_hugger": WallHuggerAgent(),
        "center_seeker": CenterSeekerAgent(),
        "random_99": RandomAgent(99),
        "random_77": RandomAgent(77),
    }


# --------------------------------------------------------------------------- #
# hidden evaluation
# --------------------------------------------------------------------------- #


def hidden_evaluate(
    store: VersionStore,
    strategy_name: str,
    version: int,
    n_seeds: int = 4,
    verbose: bool = True,
) -> dict:
    """Blind-evaluate one saved version against the hidden opponent pool.

    Returns a dict with win_rate, h2h, and per-opponent details.
    """
    agent = store.load_agent(strategy_name, version)
    opponents = get_hidden_opponents()

    win_rate, h2h = evaluate_strategy(agent, opponents, n_seeds=n_seeds)

    result = {
        "strategy": strategy_name,
        "version": version,
        "win_rate": win_rate,
        "h2h": h2h,
        "n_seeds": n_seeds,
        "n_opponents": len(opponents),
        "opponent_names": list(opponents.keys()),
    }

    if verbose:
        print(f"\nHidden Evaluation: {strategy_name}/v{version}")
        print(f"  overall win_rate: {win_rate:.1%}")
        print(f"  opponents ({len(opponents)}, unseen during iteration):")
        for name, rate in sorted(h2h.items(), key=lambda x: -x[1]):
            tag = "WIN" if rate > 0.5 else ("TIE" if rate == 0.5 else "LOSE")
            print(f"    {name:20s} {rate:.1%}  [{tag}]")

    return result


__all__ = [
    "hidden_evaluate", "get_hidden_opponents",
    "WallHuggerAgent", "CenterSeekerAgent",
]

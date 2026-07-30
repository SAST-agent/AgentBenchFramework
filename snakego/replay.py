"""Replay recording + self-contained HTML visualizer.

A ``ReplayRecorder`` wraps the game loop, snapshots the board after every
action, and dumps a JSON replay file.  ``build_html`` turns that JSON into
a single ``.html`` file (canvas + step controls, no server needed) that you
open in a browser to scrub through the game.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .env import SnakeGoGame


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #


class ReplayRecorder:
    """Snapshots board state at every step for later playback."""

    def __init__(self, p0_name: str = "P0", p1_name: str = "P1",
                 seed: int = 0) -> None:
        self.p0_name = p0_name
        self.p1_name = p1_name
        self.seed = seed
        self.frames: List[dict] = []
        self.winner: int = -1
        self.scores: List[int] = [0, 0]
        self._snap_initial = True

    def record_step(self, game: SnakeGoGame, action: int,
                    info: dict) -> None:
        """Capture the state *before* this action was applied.

        We snapshot pre-action so the viewer can show the decision point,
        then animate to the resulting board.
        """
        sid = game.current_snake_id() if not game.is_over() else info.get("snake")
        snake = game._get_snake(sid) if sid is not None and not game.is_over() else None
        frame = {
            "step": len(self.frames),
            "turn": game.turn,
            "player": game.current_player,
            "snake_id": sid,
            "snake_camp": snake.camp if snake else None,
            "action": action,
            "result": info.get("result") or info.get("illegal") or "turn_end",
            # board layers
            "wall_map": [row[:] for row in game.wall_map],
            "snake_map": [row[:] for row in game.snake_map],
            "item_map": [row[:] for row in game.item_map],
            # snake bodies for crisp rendering
            "snakes": [
                {"id": s.id, "camp": s.camp,
                 "coor": list(s.coor_list),
                 "len_bank": s.length_bank,
                 "railgun": s.has_railgun()}
                for s in game.snakes
            ],
        }
        self.frames.append(frame)

    def record_final(self, game: SnakeGoGame) -> None:
        """Capture the terminal board (after the last action)."""
        frame = {
            "step": len(self.frames),
            "turn": game.turn,
            "player": -1,
            "snake_id": None,
            "snake_camp": None,
            "action": -1,
            "result": "game_over",
            "wall_map": [row[:] for row in game.wall_map],
            "snake_map": [row[:] for row in game.snake_map],
            "item_map": [row[:] for row in game.item_map],
            "snakes": [
                {"id": s.id, "camp": s.camp,
                 "coor": list(s.coor_list),
                 "len_bank": s.length_bank,
                 "railgun": s.has_railgun()}
                for s in game.snakes
            ],
        }
        self.frames.append(frame)
        self.winner = game.winner()
        self.scores = list(game.scores())

    def to_dict(self) -> dict:
        return {
            "p0_name": self.p0_name,
            "p1_name": self.p1_name,
            "seed": self.seed,
            "winner": self.winner,
            "scores": self.scores,
            "n_frames": len(self.frames),
            "frames": self.frames,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        return path


# --------------------------------------------------------------------------- #
# a game driver that also records
# --------------------------------------------------------------------------- #


def play_with_replay(p0, p1, seed: int = 0,
                     config=None) -> tuple[dict, ReplayRecorder]:
    """Play a full game and return (result_dict, recorder)."""
    from .env import GameConfig
    from .agents import BaseAgent
    game = SnakeGoGame(config or GameConfig(seed=seed))
    game.reset(seed)
    rec = ReplayRecorder(p0.name if hasattr(p0, "name") else "P0",
                         p1.name if hasattr(p1, "name") else "P1", seed)
    agents = {0: p0, 1: p1}
    steps = 0
    while not game.is_over():
        player = game.current_player
        sid = game.current_snake_id()
        if sid is None:
            info = game.act(0)
            rec.record_step(game, 0, info)
            continue
        obs = game.obs()
        action = agents[player].act(obs, sid, game)
        info = game.act(action)
        rec.record_step(game, action, info)
        steps += 1
        if steps > 200000:
            break
    rec.record_final(game)
    s0, s1 = game.scores()
    result = {
        "winner": game.winner(), "scores": (s0, s1),
        "turns": game.turn, "steps": steps,
    }
    return result, rec


__all__ = ["ReplayRecorder", "play_with_replay"]

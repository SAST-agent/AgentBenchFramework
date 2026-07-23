"""
PayoffMatrix: pairwise win-rate matrix with incremental updates.

Each unordered strategy pair is played once with Match; the win rate is stored
from the lower-id side's perspective. When a new strategy is registered, only
the pairs that involve it (and have not been played) are computed, so adding a
strategy to a large population is O(n) rather than O(n^2).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.arena.match import Match


class PayoffMatrix:
    def __init__(self, game: str = "30_antwar2"):
        self.game = game
        self.results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.ids: List[str] = []

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def has(self, a: str, b: str) -> bool:
        return self._key(a, b) in self.results

    def get(self, a: str, b: str) -> float:
        """Win rate of a against b (0..1, 0.5 for self-play)."""
        if a == b:
            return 0.5
        lo, hi = self._key(a, b)
        entry = self.results.get((lo, hi))
        if entry is None:
            return float("nan")
        wr = float(entry["win_rate"])
        return wr if a == lo else 1.0 - wr

    def register_id(self, sid: str):
        if sid not in self.ids:
            self.ids.append(sid)

    def update(self,
               env,
               strategies: Dict[str, Any],
               n_games: int = 10,
               seed: int = 42,
               only_ids: Optional[List[str]] = None,
               force: bool = False) -> Dict[str, Any]:
        """Play every unordered pair once, skipping pairs already computed."""
        pool = only_ids if only_ids is not None else list(strategies.keys())
        for sid in pool:
            self.register_id(sid)
        pairs_played = 0
        started = time.time()
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if not force and self.has(a, b):
                    continue
                if a not in strategies or b not in strategies:
                    continue
                result = Match(env, strategies[a], strategies[b], seed=seed).run(n_games=n_games)
                lo, hi = self._key(a, b)
                if a == lo:
                    wr = result.win_rate
                    lo_wins, hi_wins = result.agent1_wins, result.agent2_wins
                else:
                    wr = 1.0 - result.win_rate
                    lo_wins, hi_wins = result.agent2_wins, result.agent1_wins
                self.results[(lo, hi)] = {
                    "win_rate": wr,
                    "lo_wins": lo_wins,
                    "hi_wins": hi_wins,
                    "draws": result.draws,
                    "games": result.games_played,
                    "avg_length": result.avg_game_length,
                }
                pairs_played += 1
        return {
            "pairs_played": pairs_played,
            "total_pairs": len(self.results),
            "elapsed_s": round(time.time() - started, 2),
        }

    def add_strategy(self,
                     env,
                     sid: str,
                     strategy,
                     strategies: Dict[str, Any],
                     n_games: int = 10,
                     seed: int = 42) -> Dict[str, Any]:
        """Incrementally evaluate a freshly added strategy against the rest."""
        self.register_id(sid)
        opponents = [s for s in self.ids if s != sid]
        return self.update(env, strategies, n_games=n_games, seed=seed, only_ids=[sid] + opponents)

    def matrix(self, ids: Optional[List[str]] = None) -> List[List[float]]:
        ids = ids or self.ids
        return [[self.get(a, b) for b in ids] for a in ids]

    def _wins_for(self, sid: str, opps: List[str]) -> Tuple[int, int]:
        wins = 0
        games = 0
        for o in opps:
            lo, hi = self._key(sid, o)
            entry = self.results[(lo, hi)]
            wins += entry["lo_wins"] if sid == lo else entry["hi_wins"]
            games += entry["games"]
        return wins, games

    def ranking(self, ids: Optional[List[str]] = None) -> List[Tuple[str, float, int, int]]:
        ids = ids or self.ids
        rows = []
        for sid in ids:
            opps = [o for o in ids if o != sid and self.has(sid, o)]
            if not opps:
                rows.append((sid, float("nan"), 0, 0))
                continue
            avg = sum(self.get(sid, o) for o in opps) / len(opps)
            wins, games = self._wins_for(sid, opps)
            rows.append((sid, avg, wins, games))
        rows.sort(key=lambda r: (r[1] if r[1] == r[1] else -1), reverse=True)
        return rows

    def save(self, path: str):
        payload = {
            "game": self.game,
            "ids": self.ids,
            "results": [{"pair": list(k), **v} for k, v in self.results.items()],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PayoffMatrix":
        with open(path) as f:
            payload = json.load(f)
        m = cls(game=payload.get("game", "30_antwar2"))
        m.ids = list(payload.get("ids", []))
        for row in payload.get("results", []):
            pair = tuple(row["pair"])
            m.results[pair] = {k: v for k, v in row.items() if k != "pair"}
        return m

    def summary(self, ids: Optional[List[str]] = None) -> Dict[str, Any]:
        ids = ids or self.ids
        return {
            "game": self.game,
            "n_strategies": len(ids),
            "n_pairs": len(self.results),
            "matrix": self.matrix(ids),
            "ranking": [{"strategy_id": s, "avg_win_rate": wr, "wins": w, "games": g}
                        for s, wr, w, g in self.ranking(ids)],
        }

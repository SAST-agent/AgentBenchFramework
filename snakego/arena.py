"""Arena: run matches and round-robin tournaments, produce payoff matrix + Elo.

A *match* is one game between two agents.  Because SnakeGo has a first-player
asymmetry (P0 starts top-right, P1 bottom-left), each pairing is played twice
with sides swapped, and the win-rate is averaged.

A *round-robin* plays every pair of agents both ways across N seeds, then
computes a payoff matrix (row beats column) and an Elo ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .env import GameConfig, SnakeGoGame
from .agents import BaseAgent
from .smoke_test import play_game


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #


class EloTracker:
    """Standard Elo with K=32, anchor 1500."""

    K = 32.0
    BASE = 1500.0

    def __init__(self) -> None:
        self.ratings: Dict[str, float] = {}

    def ensure(self, name: str) -> float:
        return self.ratings.setdefault(name, self.BASE)

    def update(self, name: str, score: float, opp: str) -> None:
        """*score* in {0, 0.5, 1} (loss/draw/win)."""
        ra = self.ensure(name)
        rb = self.ensure(opp)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        self.ratings[name] = ra + self.K * (score - ea)

    def get(self, name: str) -> float:
        return self.ensure(name)

    def rankings(self) -> List[Tuple[int, str, float]]:
        ranked = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        return [(i + 1, name, round(r)) for i, (name, r) in enumerate(ranked)]


# --------------------------------------------------------------------------- #
# match / tournament results
# --------------------------------------------------------------------------- #


@dataclass
class MatchResult:
    p0_name: str
    p1_name: str
    seed: int
    winner: int            # 0 or 1
    scores: Tuple[int, int]
    turns: int
    steps: int
    error: Optional[str] = None


@dataclass
class PayoffMatrix:
    """win_rate[a][b] = fraction of games *a* beat *b* (swapped sides averaged)."""
    agents: List[str]
    win_rate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    games_played: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, winner_name: str, loser_name: str, draw: bool = False) -> None:
        wr = self.win_rate.setdefault(winner_name, {}).setdefault(loser_name, 0.0)
        wr_l = self.win_rate.setdefault(loser_name, {}).setdefault(winner_name, 0.0)
        gp = self.games_played.setdefault(winner_name, {}).setdefault(loser_name, 0)
        gp_l = self.games_played.setdefault(loser_name, {}).setdefault(winner_name, 0)
        if draw:
            self.win_rate[winner_name][loser_name] = wr + 0.5
            self.win_rate[loser_name][winner_name] = wr_l + 0.5
        else:
            self.win_rate[winner_name][loser_name] = wr + 1.0
        self.games_played[winner_name][loser_name] = gp + 1
        self.games_played[loser_name][winner_name] = gp_l + 1

    def finalize(self) -> None:
        """Convert raw win-counts to rates."""
        for a in self.agents:
            for b in self.agents:
                if a == b:
                    continue
                gp = self.games_played.get(a, {}).get(b, 0)
                if gp > 0:
                    self.win_rate.setdefault(a, {})[b] = round(
                        self.win_rate.get(a, {}).get(b, 0.0) / gp, 4)
                else:
                    self.win_rate.setdefault(a, {})[b] = 0.0

    def table(self) -> List[List[float]]:
        return [[round(self.win_rate.get(a, {}).get(b, 0.0), 3)
                 for b in self.agents] for a in self.agents]


@dataclass
class TournamentResult:
    matches: List[MatchResult] = field(default_factory=list)
    payoff: PayoffMatrix = field(default_factory=lambda: PayoffMatrix(agents=[]))
    elo: EloTracker = field(default_factory=EloTracker)


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #


class Arena:
    """Organise matches between named agents and run round-robins."""

    def __init__(self, agents: Dict[str, BaseAgent], config: Optional[GameConfig] = None):
        self.agents = agents
        self.config = config or GameConfig()

    # -- single match -------------------------------------------------------

    def match(self, p0_name: str, p1_name: str, seed: int) -> MatchResult:
        p0 = self.agents[p0_name]
        p1 = self.agents[p1_name]
        r = play_game(p0, p1, seed=seed, config=self.config)
        return MatchResult(
            p0_name=p0_name, p1_name=p1_name, seed=seed,
            winner=r["winner"], scores=r["scores"], turns=r["turns"],
            steps=r["steps"], error=r.get("error"),
        )

    # -- round robin --------------------------------------------------------

    def round_robin(self, n_seeds: int = 4, base_seed: int = 0,
                    verbose: bool = False) -> TournamentResult:
        names = list(self.agents.keys())
        result = TournamentResult()
        result.payoff = PayoffMatrix(agents=names)

        pair_idx = 0
        total_pairs = len(names) * (len(names) - 1) // 2
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pair_idx += 1
                if verbose:
                    print(f"  [{pair_idx}/{total_pairs}] {a} vs {b} ...", flush=True)
                for s in range(n_seeds):
                    seed = base_seed + s
                    # game 1: a=P0, b=P1
                    m1 = self.match(a, b, seed)
                    result.matches.append(m1)
                    self._apply(result, m1)
                    # game 2: swap sides — b=P0, a=P1
                    m2 = self.match(b, a, seed + 1000)
                    result.matches.append(m2)
                    self._apply(result, m2)
        result.payoff.finalize()
        return result

    def _apply(self, result: TournamentResult, m: MatchResult) -> None:
        """Feed one match into the payoff matrix and Elo."""
        if m.winner == 0:
            win_n, lose_n = m.p0_name, m.p1_name
            draw = False
        elif m.winner == 1:
            win_n, lose_n = m.p1_name, m.p0_name
            draw = False
        else:
            win_n, lose_n = m.p0_name, m.p1_name
            draw = True
        # payoff counts 1 game per side; we normalise later in finalize
        result.payoff.record(win_n, lose_n, draw=draw)
        # Elo: treat each game as a standalone rated game
        s_win = 0.5 if draw else 1.0
        s_lose = 0.5 if draw else 0.0
        result.elo.update(win_n, s_win, lose_n)
        result.elo.update(lose_n, s_lose, win_n)


__all__ = ["Arena", "MatchResult", "PayoffMatrix", "TournamentResult", "EloTracker"]

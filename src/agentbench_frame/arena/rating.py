"""
Elo rating system for tracking agent skill levels.

Standard Elo with configurable K-factor.
Tracks rating history for all registered players.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class EloTracker:
    """
    Elo rating system for agent skill tracking.

    Default parameters:
    - Initial rating: 1500
    - K-factor: 32 (standard for chess; lower for more stable ratings)
    - Scale: 400 (standard)

    Example:
        elo = EloTracker()
        elo.add_player("AgentA")
        elo.add_player("AgentB")
        elo.update("AgentA", "AgentB", winner=0)  # AgentA won
        print(elo.get_rating("AgentA"))
    """

    def __init__(self,
                 initial_rating: float = 1500.0,
                 k_factor: float = 32.0,
                 scale: float = 400.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.scale = scale
        self._ratings: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}
        self._games_played: Dict[str, int] = {}

    def add_player(self, name: str, initial_rating: Optional[float] = None):
        """Register a new player in the rating system."""
        if name not in self._ratings:
            rating = initial_rating if initial_rating is not None else self.initial_rating
            self._ratings[name] = rating
            self._history[name] = [rating]
            self._games_played[name] = 0

    def update(self, player1: str, player2: str, winner: int):
        """
        Update ratings after a game.

        Args:
            player1: Name of first player
            player2: Name of second player
            winner: 0 if player1 won, 1 if player2 won, -1 if draw
        """
        # Ensure players exist
        self.add_player(player1)
        self.add_player(player2)

        r1 = self._ratings[player1]
        r2 = self._ratings[player2]

        # Expected scores
        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / self.scale))
        e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / self.scale))

        # Actual scores
        if winner == 0:
            s1, s2 = 1.0, 0.0
        elif winner == 1:
            s1, s2 = 0.0, 1.0
        else:  # draw
            s1, s2 = 0.5, 0.5

        # Dynamic K-factor: higher for new players
        k1 = max(16, self.k_factor / (1 + self._games_played[player1] * 0.1))
        k2 = max(16, self.k_factor / (1 + self._games_played[player2] * 0.1))

        # Update ratings
        self._ratings[player1] += k1 * (s1 - e1)
        self._ratings[player2] += k2 * (s2 - e2)

        self._history[player1].append(self._ratings[player1])
        self._history[player2].append(self._ratings[player2])
        self._games_played[player1] += 1
        self._games_played[player2] += 1

    def get_rating(self, name: str) -> float:
        """Get current rating of a player."""
        return self._ratings.get(name, self.initial_rating)

    def get_history(self, name: Optional[str] = None) -> Dict[str, List[float]]:
        """Get rating history. If name is None, return all histories."""
        if name:
            return {name: self._history.get(name, [])}
        return self._history

    def get_rankings(self) -> List[Tuple[str, float, int]]:
        """Get current rankings sorted by rating (highest first)."""
        ranked = [(name, rating, self._games_played.get(name, 0))
                  for name, rating in self._ratings.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def predict_win_probability(self, player1: str, player2: str) -> float:
        """Predict win probability for player1 against player2."""
        r1 = self._ratings.get(player1, self.initial_rating)
        r2 = self._ratings.get(player2, self.initial_rating)
        return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / self.scale))


@dataclass(frozen=True)
class EloGameRecord:
    role: str
    candidate: str
    opponent: str
    result: str
    act_id: str
    version_id: str
    seed: int
    game_index: int
    rating_before: float
    rating_after: float
    opponent_rating_before: float
    opponent_rating_after: float


class RoleEloLedger:
    """Per-game Elo with explicit asymmetric role and optional fixed anchors."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
    ) -> None:
        if k_factor <= 0 or scale <= 0:
            raise ValueError("k_factor and scale must be positive")
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.scale = float(scale)
        self._ratings: Dict[Tuple[str, str], float] = {}
        self._records: List[EloGameRecord] = []

    def rating(self, role: str, player: str) -> float:
        return self._ratings.get((role, player), self.initial_rating)

    def history(self, role: str, player: str) -> Tuple[EloGameRecord, ...]:
        return tuple(
            record
            for record in self._records
            if record.role == role and record.candidate == player
        )

    def update_game(
        self,
        *,
        role: str,
        candidate: str,
        opponent: str,
        result: str,
        act_id: str,
        version_id: str,
        seed: int,
        anchor_opponent: bool = False,
    ) -> EloGameRecord:
        if not role or not candidate or not opponent:
            raise ValueError("role, candidate, and opponent are required")
        scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
        if result not in scores:
            raise ValueError("Elo accepts only valid win, draw, or loss games")
        candidate_key = (role, candidate)
        opponent_key = (role, opponent)
        before = self.rating(role, candidate)
        opponent_before = self.rating(role, opponent)
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_before - before) / self.scale))
        after = before + self.k_factor * (scores[result] - expected)
        opponent_after = opponent_before
        if not anchor_opponent:
            opponent_after = opponent_before + self.k_factor * (
                (1.0 - scores[result]) - (1.0 - expected)
            )
        self._ratings[candidate_key] = after
        self._ratings[opponent_key] = opponent_after
        record = EloGameRecord(
            role=role,
            candidate=candidate,
            opponent=opponent,
            result=result,
            act_id=act_id,
            version_id=version_id,
            seed=int(seed),
            game_index=len(self._records) + 1,
            rating_before=before,
            rating_after=after,
            opponent_rating_before=opponent_before,
            opponent_rating_after=opponent_after,
        )
        self._records.append(record)
        return record

    def update_series(
        self,
        *,
        role: str,
        candidate: str,
        games: List[Dict[str, Any]],
        anchor_opponents: bool = False,
    ) -> Tuple[EloGameRecord, ...]:
        records = []
        for game in games:
            records.append(
                self.update_game(
                    role=role,
                    candidate=candidate,
                    opponent=str(game["opponent"]),
                    result=str(game["result"]),
                    act_id=str(game["act_id"]),
                    version_id=str(game["version_id"]),
                    seed=int(game["seed"]),
                    anchor_opponent=anchor_opponents,
                )
            )
        return tuple(records)

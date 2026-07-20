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

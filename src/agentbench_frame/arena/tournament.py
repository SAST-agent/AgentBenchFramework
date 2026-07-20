"""
Tournament/Arena system for multi-agent competition.

Supports:
- Round-robin: Every agent plays every other agent
- Elimination: Single/double elimination brackets
- Ladder: Agents challenge higher-ranked opponents
- Custom scheduling
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import itertools
import time
import json

from agentbench_frame.env.base import BaseEnv
from agentbench_frame.agent.base import BaseAgent
from agentbench_frame.arena.match import Match, MatchResult
from agentbench_frame.arena.rating import EloTracker


@dataclass
class TournamentResult:
    """Results from a tournament."""
    name: str
    agents: List[str]
    matches: List[MatchResult]
    final_ratings: Dict[str, float]
    rankings: List[Tuple[int, str, float]]  # (rank, agent_name, elo)
    duration_seconds: float


class Arena:
    """
    Tournament arena for multi-agent competition.

    Example:
        env = GeneralsEnv()
        agents = [agent1, agent2, agent3, agent4]

        arena = Arena(env, agents)
        result = arena.round_robin(n_games=20)
        print(f"Winner: {result.rankings[0]}")
    """

    def __init__(self,
                 env: BaseEnv,
                 agents: List[BaseAgent],
                 name: str = "Arena"):
        self.env = env
        self.agents = agents
        self.name = name
        self.elo = EloTracker()
        self._results: List[MatchResult] = []

        # Register all agents in Elo system
        for agent in agents:
            self.elo.add_player(agent.name)

    def round_robin(self, n_games: int = 10, seed: int = 42) -> TournamentResult:
        """
        Run a round-robin tournament where every agent plays every other agent.

        Args:
            n_games: Number of games per match pair
            seed: Random seed

        Returns:
            TournamentResult with rankings
        """
        start_time = time.time()
        print(f"Starting round-robin tournament: {self.name}")
        print(f"Agents: {[a.name for a in self.agents]}")
        print(f"Games per match: {n_games}")

        self._results = []
        pairs = list(itertools.combinations(range(len(self.agents)), 2))

        for i, j in pairs:
            agent1, agent2 = self.agents[i], self.agents[j]
            print(f"  Match: {agent1.name} vs {agent2.name}")

            match = Match(self.env, agent1, agent2, seed=seed + i * 100 + j)
            result = match.run(n_games=n_games)
            self._results.append(result)

            # Update Elo
            if result.win_rate > 0.5:
                self.elo.update(agent1.name, agent2.name, 0)
            elif result.win_rate < 0.5:
                self.elo.update(agent1.name, agent2.name, 1)
            # draws don't change Elo much

        # Generate rankings
        rankings = self._compute_rankings()

        duration = time.time() - start_time
        print(f"\nTournament complete in {duration:.1f}s")
        print(f"Top 3:")
        for rank, name, elo in rankings[:3]:
            print(f"  {rank}. {name} (Elo: {elo:.0f})")

        return TournamentResult(
            name=self.name,
            agents=[a.name for a in self.agents],
            matches=self._results,
            final_ratings={name: self.elo.get_rating(name)
                          for name in [a.name for a in self.agents]},
            rankings=rankings,
            duration_seconds=duration,
        )

    def elimination(self, n_games: int = 10, seed: int = 42) -> TournamentResult:
        """
        Run a single-elimination tournament.

        Agents are randomly seeded. Winners advance to the next round.
        """
        import random
        rng = random.Random(seed)

        remaining = list(range(len(self.agents)))
        rng.shuffle(remaining)
        round_num = 1

        while len(remaining) > 1:
            print(f"\nElimination Round {round_num}")
            next_round = []

            for i in range(0, len(remaining), 2):
                if i + 1 >= len(remaining):
                    next_round.append(remaining[i])  # bye
                    continue

                a1, a2 = self.agents[remaining[i]], self.agents[remaining[i + 1]]
                print(f"  {a1.name} vs {a2.name}")

                match = Match(self.env, a1, a2, seed=seed + round_num * 100 + i)
                result = match.run(n_games=n_games)

                winner_idx = remaining[i] if result.win_rate >= 0.5 else remaining[i + 1]
                next_round.append(winner_idx)

                self._results.append(result)
                self.elo.update(a1.name, a2.name,
                               0 if result.win_rate >= 0.5 else 1)

            remaining = next_round
            round_num += 1

        rankings = self._compute_rankings()
        return TournamentResult(
            name=f"{self.name} (Elimination)",
            agents=[a.name for a in self.agents],
            matches=self._results,
            final_ratings={name: self.elo.get_rating(name)
                          for name in [a.name for a in self.agents]},
            rankings=rankings,
            duration_seconds=0,
        )

    def ladder(self, n_challenges: int = 5, n_games: int = 10,
               seed: int = 42) -> TournamentResult:
        """
        Run a ladder-style tournament.

        Agents start ranked by initial Elo. Lower-ranked agents can challenge
        higher-ranked ones to move up.
        """
        import random
        rng = random.Random(seed)
        rankings = list(range(len(self.agents)))

        for _ in range(n_challenges):
            # Pick a random agent to challenge someone 1-3 ranks above
            challenger_idx = rng.randint(0, len(rankings) - 2)
            target_rank = max(0, challenger_idx - rng.randint(1, 3))
            defender_idx = rankings[target_rank]

            if challenger_idx == target_rank:
                continue

            a1 = self.agents[rankings[challenger_idx]]
            a2 = self.agents[defender_idx]

            match = Match(self.env, a1, a2, seed=seed)
            result = match.run(n_games=n_games)

            # If challenger wins, swap positions
            if result.win_rate > 0.5:
                rankings[challenger_idx], rankings[target_rank] = \
                    rankings[target_rank], rankings[challenger_idx]

            self._results.append(result)

        final_rankings = self._compute_rankings()
        return TournamentResult(
            name=f"{self.name} (Ladder)",
            agents=[a.name for a in self.agents],
            matches=self._results,
            final_ratings={name: self.elo.get_rating(name)
                          for name in [a.name for a in self.agents]},
            rankings=final_rankings,
            duration_seconds=0,
        )

    def _compute_rankings(self) -> List[Tuple[int, str, float]]:
        """Compute rankings from Elo ratings."""
        ratings = [(a.name, self.elo.get_rating(a.name)) for a in self.agents]
        ratings.sort(key=lambda x: x[1], reverse=True)
        return [(i + 1, name, elo) for i, (name, elo) in enumerate(ratings)]

    def save_results(self, path: str):
        """Save tournament results to JSON."""
        output = {
            "name": self.name,
            "matches": [
                {
                    "agent1": m.agent1_name,
                    "agent2": m.agent2_name,
                    "games": m.games_played,
                    "win_rate": m.win_rate,
                    "wins_1": m.agent1_wins,
                    "wins_2": m.agent2_wins,
                    "draws": m.draws,
                }
                for m in self._results
            ],
            "ratings": {name: self.elo.get_rating(name)
                       for name in [a.name for a in self.agents]},
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

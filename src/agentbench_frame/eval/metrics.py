"""
Metrics Calculator

Computes performance metrics for agent evaluation:
- Win rates by opponent
- Elo ratings over time
- Per-strategy statistics
- Action distribution analysis
- Game length analysis
"""

from typing import Any, Dict, List, Optional, Tuple
import statistics


class MetricsCalculator:
    """
    Computes comprehensive performance metrics for agent evaluation.

    Example:
        calc = MetricsCalculator()
        calc.add_result("AgentA", "AgentB", winner=0, steps=42, reward=15.0)
        stats = calc.get_stats("AgentA")
    """

    def __init__(self):
        self._results: List[Dict[str, Any]] = []
        self._elo_history: Dict[str, List[float]] = {}

    def add_result(self,
                   agent_name: str,
                   opponent_name: str,
                   winner: int,
                   steps: int,
                   reward: float,
                   metadata: Optional[Dict[str, Any]] = None):
        """Record a single game result."""
        self._results.append({
            "agent": agent_name,
            "opponent": opponent_name,
            "winner": winner,
            "steps": steps,
            "reward": reward,
            "metadata": metadata or {},
        })

    def get_win_rate(self,
                     agent_name: str,
                     opponent_name: Optional[str] = None) -> float:
        """
        Calculate win rate for an agent.

        Args:
            agent_name: Agent to calculate for
            opponent_name: If provided, only consider games vs this opponent

        Returns:
            Win rate (0.0 to 1.0)
        """
        results = self._get_results(agent_name, opponent_name)
        if not results:
            return 0.0

        wins = sum(1 for r in results if r["winner"] == 0)
        return wins / len(results)

    def get_head_to_head(self, agent1: str, agent2: str) -> Dict[str, Any]:
        """Get head-to-head statistics between two agents."""
        results = self._get_results(agent1, opponent_name=agent2)
        if not results:
            return {"games": 0, "win_rate": 0.0}

        wins = sum(1 for r in results if r["winner"] == 0)
        draws = sum(1 for r in results if r["winner"] == -1)
        losses = len(results) - wins - draws

        return {
            "games": len(results),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(results),
            "avg_steps": statistics.mean([r["steps"] for r in results]),
            "avg_reward": statistics.mean([r["reward"] for r in results]),
        }

    def get_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for an agent.

        Returns:
            Dictionary with win_rate, avg_steps, per-opponent stats, etc.
        """
        results = self._get_results(agent_name)
        if not results:
            return {"games_played": 0}

        opponents = set(r["opponent"] for r in results)
        per_opponent = {}
        for opp in opponents:
            per_opponent[opp] = self.get_head_to_head(agent_name, opp)

        return {
            "games_played": len(results),
            "overall_win_rate": sum(1 for r in results if r["winner"] == 0) / len(results),
            "avg_steps": statistics.mean([r["steps"] for r in results]),
            "avg_reward": statistics.mean([r["reward"] for r in results]),
            "std_reward": statistics.stdev([r["reward"] for r in results]) if len(results) > 1 else 0,
            "unique_opponents": len(opponents),
            "per_opponent": per_opponent,
            "recent_win_rate": self._recent_win_rate(agent_name, n=20),
        }

    def get_action_stats(self, agent_name: str) -> Dict[str, Any]:
        """Analyze action distribution for an agent."""
        results = self._get_results(agent_name)
        if not results:
            return {}

        # Collect action types used
        action_counts = {}
        for r in results:
            metadata = r.get("metadata", {})
            actions = metadata.get("actions", [])
            for action in actions:
                action_type = str(action[0]) if isinstance(action, list) else str(action)
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

        total = sum(action_counts.values()) or 1
        return {
            "total_actions": total,
            "action_distribution": {k: v / total for k, v in action_counts.items()},
            "action_counts": action_counts,
        }

    def compute_elo_ratings(self, initial_elo: float = 1500.0,
                            k_factor: float = 32.0) -> Dict[str, float]:
        """Compute Elo ratings from all recorded results."""
        from agentbench_frame.arena.rating import EloTracker
        elo = EloTracker(initial_rating=initial_elo, k_factor=k_factor)

        for r in self._results:
            elo.add_player(r["agent"])
            elo.add_player(r["opponent"])
            elo.update(r["agent"], r["opponent"], r["winner"])

        return {name: elo.get_rating(name) for name in elo._ratings}

    def _get_results(self, agent_name: str,
                     opponent_name: Optional[str] = None) -> List[Dict]:
        """Filter results for a specific agent and optional opponent."""
        results = [r for r in self._results if r["agent"] == agent_name]
        if opponent_name:
            results = [r for r in results if r["opponent"] == opponent_name]
        return results

    def _recent_win_rate(self, agent_name: str, n: int = 20) -> float:
        """Get win rate over the last n games."""
        results = self._get_results(agent_name)
        recent = results[-n:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r["winner"] == 0) / len(recent)

    def clear(self):
        """Clear all recorded results."""
        self._results = []

    def save(self, path: str):
        """Save results to JSON."""
        import json
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)

    def load(self, path: str):
        """Load results from JSON."""
        import json
        with open(path, "r") as f:
            self._results = json.load(f)

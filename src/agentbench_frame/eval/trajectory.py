"""
Trajectory Recorder

Records game episodes for later analysis, replay, and training data generation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class Episode:
    """A single recorded game episode."""
    episode_id: str
    agent_name: str
    opponent_name: str
    game_name: str
    timestamp: float
    trajectory: List[Dict[str, Any]]
    total_reward: float
    winner: int
    steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryRecorder:
    """
    Records game episodes (state, action, reward sequences) for analysis.

    Supports:
    - Per-episode recording
    - JSON export for analysis
    - Training data generation
    - Replay support

    Example:
        recorder = TrajectoryRecorder()
        recorder.add_episode("MyAgent", "Opponent", trajectory, 10.0, 0)
        recorder.save("trajectories.json")
    """

    def __init__(self, max_episodes: int = 10000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        self._episode_counter = 0

    def add_episode(self,
                    agent_name: str,
                    opponent_name: str,
                    trajectory: List[Dict[str, Any]],
                    reward: float,
                    winner: int,
                    game_name: str = "unknown",
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record a completed episode.

        Args:
            agent_name: Name of the primary agent
            opponent_name: Name of the opponent
            trajectory: List of (state, action, reward) steps
            reward: Total episode reward
            winner: Winner ID (0=agent, 1=opponent, -1=draw)
            game_name: Name of the game
            metadata: Additional info
        """
        episode = Episode(
            episode_id=f"ep_{self._episode_counter:06d}",
            agent_name=agent_name,
            opponent_name=opponent_name,
            game_name=game_name,
            timestamp=time.time(),
            trajectory=trajectory,
            total_reward=reward,
            winner=winner,
            steps=len(trajectory),
            metadata=metadata or {},
        )

        self.episodes.append(episode)
        self._episode_counter += 1

        # Trim old episodes if needed
        while len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def save(self, path: str):
        """Save all recorded episodes to a JSON file."""
        data = {
            "num_episodes": len(self.episodes),
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "agent_name": ep.agent_name,
                    "opponent_name": ep.opponent_name,
                    "game_name": ep.game_name,
                    "timestamp": ep.timestamp,
                    "total_reward": ep.total_reward,
                    "winner": ep.winner,
                    "steps": ep.steps,
                    "trajectory": ep.trajectory,
                    "metadata": ep.metadata,
                }
                for ep in self.episodes
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str):
        """Load episodes from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.episodes = []
        for ep_data in data.get("episodes", []):
            ep = Episode(
                episode_id=ep_data["episode_id"],
                agent_name=ep_data["agent_name"],
                opponent_name=ep_data["opponent_name"],
                game_name=ep_data["game_name"],
                timestamp=ep_data["timestamp"],
                trajectory=ep_data["trajectory"],
                total_reward=ep_data["total_reward"],
                winner=ep_data["winner"],
                steps=ep_data["steps"],
                metadata=ep_data.get("metadata", {}),
            )
            self.episodes.append(ep)

    def get_episodes_by_agent(self, agent_name: str) -> List[Episode]:
        """Get all episodes for a specific agent."""
        return [ep for ep in self.episodes if ep.agent_name == agent_name]

    def get_episodes_by_opponent(self, opponent_name: str) -> List[Episode]:
        """Get all episodes against a specific opponent."""
        return [ep for ep in self.episodes if ep.opponent_name == opponent_name]

    def get_win_rate(self, agent_name: Optional[str] = None) -> float:
        """Calculate win rate over recorded episodes."""
        if agent_name:
            episodes = self.get_episodes_by_agent(agent_name)
        else:
            episodes = self.episodes

        if not episodes:
            return 0.0

        wins = sum(1 for ep in episodes if ep.winner == 0)
        return wins / len(episodes)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all recorded episodes."""
        if not self.episodes:
            return {"num_episodes": 0}

        rewards = [ep.total_reward for ep in self.episodes]
        steps = [ep.steps for ep in self.episodes]
        win_rate = sum(1 for ep in self.episodes if ep.winner == 0) / len(self.episodes)

        return {
            "num_episodes": len(self.episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_steps": sum(steps) / len(steps),
            "win_rate": win_rate,
            "unique_agents": len(set(ep.agent_name for ep in self.episodes)),
            "unique_opponents": len(set(ep.opponent_name for ep in self.episodes)),
        }

    def clear(self):
        """Clear all recorded episodes."""
        self.episodes = []
        self._episode_counter = 0

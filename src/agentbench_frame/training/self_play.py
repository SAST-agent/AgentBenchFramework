"""
Self-Play Training

Implements self-play training where the agent plays against historical
versions of itself, creating an automatic curriculum.

Pattern:
1. Maintain a pool of opponent checkpoints
2. Train against randomly sampled opponents from the pool
3. Periodically add the current agent to the opponent pool
4. Track Elo ratings of all checkpoints
"""

import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import copy
import random

from agentbench_frame.env.base import BaseEnv
from agentbench_frame.agent.base import BaseAgent
from agentbench_frame.arena.rating import EloTracker
from agentbench_frame.arena.match import Match


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    pool_size: int = 10
    add_to_pool_every: int = 100  # episodes
    n_eval_games: int = 20
    total_episodes: int = 1000
    initial_elo: float = 1500.0
    log_dir: str = "./self_play_logs"


class SelfPlayTrainer:
    """
    Self-play training loop.

    The agent trains by playing against a pool of historical opponents.
    This prevents overfitting to a single strategy and creates a natural
    curriculum as the agent improves.

    Example:
        trainer = SelfPlayTrainer(env, agent)
        trainer.train()
    """

    def __init__(self,
                 env: BaseEnv,
                 agent: BaseAgent,
                 config: Optional[SelfPlayConfig] = None):
        self.env = env
        self.agent = agent
        self.config = config or SelfPlayConfig()
        self.elo = EloTracker()

        # Opponent pool management
        self._opponent_pool: List[BaseAgent] = []
        self._opponent_versions: List[int] = []
        self._current_version = 0
        self._episodes_since_add = 0
        self._training_history: List[Dict] = []

        # Setup
        os.makedirs(self.config.log_dir, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """
        Run the self-play training loop.

        Returns:
            Training summary
        """
        print(f"Starting self-play training: {self.config.total_episodes} episodes")

        # Add initial agent to pool
        self._add_to_pool("initial")

        for episode in range(self.config.total_episodes):
            # Sample opponent from pool
            opponent = self._sample_opponent()

            # Run training game
            result = self._play_training_game(opponent)

            # Learn from game
            learn_metrics = self.agent.learn(result)

            # Track Elo
            self.elo.update(self.agent.name, opponent.name, result["winner"])

            # Periodically evaluate against full pool
            if episode % 50 == 0:
                elo = self._evaluate_pool()
                print(f"Episode {episode}: ELO={elo:.0f}, pool_size={len(self._opponent_pool)}")

            # Add checkpoint to pool periodically
            self._episodes_since_add += 1
            if self._episodes_since_add >= self.config.add_to_pool_every:
                self._add_to_pool(f"v{self._current_version}")
                self._current_version += 1
                self._episodes_since_add = 0

            # Maintain pool size
            while len(self._opponent_pool) > self.config.pool_size:
                self._opponent_pool.pop(0)

            self._training_history.append({
                "episode": episode,
                "opponent": opponent.name,
                "winner": result.get("winner"),
                **learn_metrics,
            })

        return {
            "final_elo": self.elo.get_rating(self.agent.name),
            "opponents_trained_against": len(set(a.name for a in self._opponent_pool)),
            "history": self._training_history,
            "elo_history": self.elo.get_history(),
        }

    def _sample_opponent(self) -> BaseAgent:
        """Sample an opponent from the pool."""
        if not self._opponent_pool:
            return self.agent  # self-play with same agent

        # Weighted sampling: prefer recent opponents but include older ones
        n = len(self._opponent_pool)
        weights = [0.5 + 0.5 * i / max(1, n - 1) for i in range(n)]
        total = sum(weights)
        weights = [w / total for w in weights]
        idx = random.choices(range(n), weights=weights, k=1)[0]
        return self._opponent_pool[idx]

    def _play_training_game(self, opponent: BaseAgent) -> Dict[str, Any]:
        """Play a single training game."""
        obs = self.env.reset()
        done = False
        current_agent = 0  # Our agent is player 0, opponent is player 1

        while not done:
            if current_agent == 0:
                action = self.agent.act(obs.to_dict())
            else:
                action = opponent.act(obs.to_dict())

            obs, reward, done, info = self.env.step(action)
            current_agent = 1 - current_agent

        return {
            "winner": obs.state.get("winner", -1),
            "reward": reward,
            "steps": obs.round_num,
            "opponent": opponent.name,
        }

    def _evaluate_pool(self) -> float:
        """Evaluate agent against all pool opponents."""
        total_elo = 0
        n_matches = 0

        for opponent in self._opponent_pool[-3:]:  # evaluate against recent 3
            match = Match(self.env, self.agent, opponent)
            result = match.run(n_games=self.config.n_eval_games)
            total_elo += result.get("elo_change", 0)
            n_matches += 1

        return total_elo / max(1, n_matches) if n_matches > 0 else 0

    def _add_to_pool(self, label: str):
        """Add current agent checkpoint to opponent pool."""
        checkpoint = copy.deepcopy(self.agent)
        checkpoint.name = f"{self.agent.name}_{label}"
        self._opponent_pool.append(checkpoint)
        self.elo.add_player(checkpoint.name)

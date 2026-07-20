"""
Reinforcement Learning Training Loop

Provides a training orchestrator that wraps the environment and agent
into a standard RL training loop. Supports SB3 and Tianshou backends.

For AgentBench games, the training loop:
1. Runs game episodes to collect experience
2. Updates the policy using the configured algorithm
3. Evaluates against baseline opponents
4. Logs metrics (win rate, reward, game length)
5. Saves checkpoints
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics

from agentbench_frame.env.base import BaseEnv, Observation
from agentbench_frame.agent.base import BaseAgent
from agentbench_frame.eval.trajectory import TrajectoryRecorder
from agentbench_frame.eval.metrics import MetricsCalculator


@dataclass
class RLTrainConfig:
    """Configuration for RL training."""
    total_timesteps: int = 100_000
    n_envs: int = 1
    eval_freq: int = 10_000
    eval_episodes: int = 20
    save_freq: int = 50_000
    log_dir: str = "./training_logs"
    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42


class RLTrainer:
    """
    Trainer for RL-based agents.

    Handles the training loop, evaluation, checkpointing, and logging.
    Can be configured with different algorithms (PPO, A2C, DQN).

    Example:
        env = GeneralsEnv(mode=EnvMode.DIRECT)
        agent = RLAgent(name="GeneralsRL")
        trainer = RLTrainer(env, agent)
        trainer.train()
    """

    def __init__(self,
                 env: BaseEnv,
                 agent: BaseAgent,
                 config: Optional[RLTrainConfig] = None,
                 eval_opponents: Optional[List[BaseAgent]] = None):
        self.env = env
        self.agent = agent
        self.config = config or RLTrainConfig()
        self.eval_opponents = eval_opponents or []
        self.metrics = MetricsCalculator()
        self.trajectory_recorder = TrajectoryRecorder()

        # Training state
        self._timesteps = 0
        self._episodes = 0
        self._best_eval_score = -float("inf")
        self._training_history: List[Dict] = []

        # Setup logging
        os.makedirs(self.config.log_dir, exist_ok=True)

    def train(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            callback: Optional callback called after each episode with (trainer, metrics)

        Returns:
            Training summary dictionary
        """
        print(f"Starting RL training: {self.config.algorithm}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Environment: {self.env.game_name}")

        start_time = time.time()

        while self._timesteps < self.config.total_timesteps:
            # Run episode
            episode_metrics = self._run_episode()
            self._episodes += 1
            self._timesteps += episode_metrics.get("steps", 0)

            # Record history
            self._training_history.append({
                "episode": self._episodes,
                "timesteps": self._timesteps,
                **episode_metrics,
            })

            # Evaluate periodically
            if self._timesteps % self.config.eval_freq == 0:
                eval_score = self._evaluate()
                print(f"Timestep {self._timesteps}: eval_score={eval_score:.3f}")

                if eval_score > self._best_eval_score:
                    self._best_eval_score = eval_score
                    self._save_checkpoint("best")

            # Save checkpoint periodically
            if self._timesteps % self.config.save_freq == 0:
                self._save_checkpoint(f"step_{self._timesteps}")

            # Callback
            if callback:
                callback(self, episode_metrics)

        elapsed = time.time() - start_time
        summary = {
            "total_timesteps": self._timesteps,
            "total_episodes": self._episodes,
            "elapsed_seconds": elapsed,
            "best_eval_score": self._best_eval_score,
            "history": self._training_history,
        }

        # Save final model
        self._save_checkpoint("final")
        self._save_summary(summary)

        print(f"Training complete. Best eval score: {self._best_eval_score:.3f}")
        return summary

    def _run_episode(self) -> Dict[str, float]:
        """Run a single episode and update the policy."""
        obs = self.env.reset(seed=self.config.seed + self._episodes)
        done = False
        total_reward = 0.0
        steps = 0
        episode_data = []

        while not done:
            action = self.agent.act(obs.to_dict())
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            episode_data.append({
                "state": obs.to_dict(),
                "action": action,
                "reward": reward,
                "done": done,
            })

        # Learn from the episode
        learn_metrics = self.agent.learn({"episode": episode_data})

        # Record trajectory
        self.trajectory_recorder.add_episode(
            agent_name=self.agent.name,
            opponent_name="self",
            trajectory=self.env.get_trajectory(),
            reward=total_reward,
            winner=obs.state.get("winner", -1),
        )

        return {
            "steps": steps,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(1, steps),
            "winner": obs.state.get("winner", -1),
            **learn_metrics,
        }

    def _evaluate(self) -> float:
        """Evaluate the agent against baseline opponents."""
        from agentbench_frame.arena.match import Match

        scores = []
        for opponent in self.eval_opponents or []:
            match = Match(self.env, self.agent, opponent)
            result = match.run(n_games=self.config.eval_episodes)
            scores.append(result["win_rate"])

        if not scores:
            # Self-evaluation: check if agent can complete game
            wins = 0
            for _ in range(self.config.eval_episodes):
                obs = self.env.reset()
                done = False
                while not done:
                    action = self.agent.act(obs.to_dict())
                    obs, _, done, _ = self.env.step(action)
                    if done and obs.state.get("winner") == 0:
                        wins += 1
            scores.append(wins / self.config.eval_episodes)

        return statistics.mean(scores) if scores else 0.0

    def _save_checkpoint(self, name: str):
        """Save agent checkpoint."""
        path = os.path.join(self.config.log_dir, f"checkpoint_{name}")
        self.agent.save(path)

    def _save_summary(self, summary: Dict):
        """Save training summary to JSON."""
        path = os.path.join(self.config.log_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def get_history(self) -> List[Dict]:
        """Return training history for analysis."""
        return self._training_history

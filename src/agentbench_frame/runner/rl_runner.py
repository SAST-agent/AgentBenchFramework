"""
BaseRLRunner — training runner for reinforcement learning.

Wires up the RL training loop with tracking. Uses PPOTrainer or
RLTrainer under the hood.
"""

import os
from typing import Any, Dict, Optional

from agentbench_frame.runner.base import BaseRunner
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.records import EpisodeRecord

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseRLRunner(BaseRunner):
    """Runner for RL training experiments.

    Usage:
        runner = BaseRLRunner(game="generals", agent="ppo-v1",
                              total_timesteps=200_000)
        runner.run()
    """

    run_type = "rl"

    def __init__(self, total_timesteps: int = 200_000, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self._eval_opponent = None

    def _create_env(self):
        from agentbench_frame.env.base import EnvMode
        from agentbench_frame.env.registry import make_env
        return make_env(self.game, mode=EnvMode.DIRECT)

    def _create_agent(self):
        from agentbench_frame.agent.rl_agent import RLAgent
        from agentbench_frame.agent.policy_network import create_generals_policy
        if HAS_TORCH:
            return RLAgent(name=self.agent_name,
                           policy_network=create_generals_policy(),
                           deterministic=False)
        return RLAgent(name=self.agent_name, deterministic=False)

    @staticmethod
    def _is_ppo_compatible(env) -> bool:
        """Return whether ``env`` matches the built-in PPO input contract."""
        from agentbench_frame.env.generals_env import GeneralsEnv

        underlying = getattr(env, "_env", env)
        return isinstance(underlying, GeneralsEnv) and hasattr(
            underlying, "to_feature_vector"
        )

    def _execute(self, run: Run) -> Dict[str, Any]:
        env = run._tracked_env
        agent = run._timed_agent

        if env is None:
            raise RuntimeError("No environment available")

        # Use the real PPO implementation whenever this environment exposes
        # the feature conversion required by the Generals policy.  The
        # tracked wrapper delegates that API to the underlying environment.
        if HAS_TORCH and self._is_ppo_compatible(env):
            from agentbench_frame.training.ppo_trainer import (
                HAS_NUMPY,
                PPOConfig,
                PPOTrainer,
            )

            if HAS_NUMPY:
                ppo_config = PPOConfig(
                    total_timesteps=self.total_timesteps,
                    learning_rate=self.config.get("learning_rate", 3e-4),
                    log_dir=self.config.get(
                        "log_dir", os.path.join(run.run_dir, "ppo")
                    ),
                    verbose=self.config.get("verbose", False),
                )
                trainer = PPOTrainer(env, config=ppo_config)
                result = trainer.train()

                for episode in (
                    trainer.get_episode_results()
                    if hasattr(trainer, "get_episode_results")
                    else []
                ):
                    run.log_episode(
                        reward=float(episode.get("reward", 0.0)),
                        steps=int(episode.get("steps", 0)),
                        winner=int(episode.get("winner", -1)),
                    )

                # Make the trained policy available through the wrapped Agent
                # for callers that continue using the runner-owned instance.
                if agent is not None and hasattr(trainer, "policy"):
                    target_agent = getattr(agent, "_agent", agent)
                    if hasattr(target_agent, "_policy_network"):
                        target_agent._policy_network = trainer.policy
                    if hasattr(target_agent, "env"):
                        target_agent.env = env

                run._total_steps = max(
                    run._total_steps,
                    int(result.get("total_timesteps", self.total_timesteps)),
                )
                run._episodes = max(
                    run._episodes,
                    int(result.get("total_episodes", run._episodes)),
                )
                run.write("training_summary", **result)
                return result

        # Generic fallback for environments that are not PPO-compatible or
        # installations without the optional numerical dependencies.
        total_episodes = 0
        total_reward = 0.0

        for ep in range(self.config.get("max_episodes", 1000)):
            if run._total_steps >= self.total_timesteps:
                break

            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            episode_data = []
            info = {}

            while not done and run._total_steps + ep_steps < self.total_timesteps:
                action = agent.act(obs.to_dict()) if agent else [[8]]
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1
                episode_data.append({
                    "state": obs.to_dict(),
                    "action": action,
                    "reward": reward,
                    "done": done,
                })

            learn_metrics = (
                agent.learn({"episode": episode_data})
                if agent and hasattr(agent, "learn")
                else {}
            )
            if learn_metrics:
                info = {**info, "learn": learn_metrics}

            winner = obs.state.get("winner", -1) if hasattr(obs, "state") else -1
            run.log_episode(reward=ep_reward, steps=ep_steps,
                            winner=winner, info=info)

            total_episodes += 1
            total_reward += ep_reward

            # Check total timesteps budget
            if run._total_steps >= self.total_timesteps:
                break

        return {
            "total_episodes": total_episodes,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(1, total_episodes),
        }

"""
BaseRLRunner — training runner for reinforcement learning.

Wires up the RL training loop with tracking. Uses PPOTrainer or
RLTrainer under the hood.
"""

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

    def _execute(self, run: Run) -> Dict[str, Any]:
        env = run._tracked_env
        agent = run._timed_agent

        if env is None:
            raise RuntimeError("No environment available")

        total_episodes = 0
        total_reward = 0.0

        for ep in range(self.config.get("max_episodes", 1000)):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action = agent.act(obs.to_dict()) if agent else [[8]]
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1

            winner = obs.state.get("winner", -1) if hasattr(obs, "state") else -1
            run.log_episode(reward=ep_reward, steps=ep_steps,
                            winner=winner, info=info)

            total_episodes += 1
            total_reward += ep_reward

            if agent and hasattr(agent, "_policy_network") and agent._policy_network:
                # Learn step (simple: just continue)
                pass

            # Check total timesteps budget
            if run._total_steps >= self.total_timesteps:
                break

        return {
            "total_episodes": total_episodes,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(1, total_episodes),
        }

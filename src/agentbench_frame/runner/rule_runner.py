"""
BaseRuleRunner — iteration runner for rule-based agents.

Runs a rule iteration loop with tracking.
"""

from typing import Any, Dict, Optional

from agentbench_frame.runner.base import BaseRunner
from agentbench_frame.tracking.run import Run


class BaseRuleRunner(BaseRunner):
    """Runner for rule-based agent iteration experiments.

    Usage:
        runner = BaseRuleRunner(game="generals", agent="expansionist",
                                opponent="random")
        runner.run()
    """

    def __init__(self, opponent: str = "random", **kwargs):
        super().__init__(**kwargs)
        self.opponent_name = opponent

    def _create_env(self):
        from agentbench_frame.env.base import EnvMode
        from agentbench_frame.env.registry import make_env
        return make_env(self.game, mode=EnvMode.DIRECT)

    def _create_agent(self):
        from agentbench_frame.agent.rule_based import RuleBasedAgent
        return RuleBasedAgent(name=self.agent_name)

    def _create_opponent(self):
        from agentbench_frame.agent.registry import AgentRegistry
        from agentbench_frame.agent.base import RandomAgent
        try:
            return AgentRegistry.create(self.opponent_name)
        except KeyError:
            return RandomAgent(name=self.opponent_name)

    def _execute(self, run: Run) -> Dict[str, Any]:
        env = run._tracked_env
        agent = run._timed_agent
        opponent = self._create_opponent()

        if env is None:
            raise RuntimeError("No environment available")

        n_games = self.config.get("n_games", 100)
        wins = 0
        total_reward = 0.0

        for game_idx in range(n_games):
            obs = env.reset(seed=42 + game_idx)
            done = False
            ep_reward = 0.0
            ep_steps = 0

            # Alternate starting side
            if game_idx % 2 == 0:
                agents = {0: agent, 1: opponent}
            else:
                agents = {0: opponent, 1: agent}

            current_player = 0
            while not done:
                current_agent = agents[current_player]
                action = current_agent.act(obs.to_dict()) if current_agent else [[8]]
                obs, reward, done, info = env.step(action)
                if current_player == (0 if game_idx % 2 == 0 else 1):
                    ep_reward += float(reward)
                ep_steps += 1
                current_player = obs.player_id

            winner = obs.state.get("winner", -1) if hasattr(obs, "state") else -1
            run.log_episode(reward=ep_reward, steps=ep_steps,
                            winner=winner, info=info)

            my_is_player0 = (game_idx % 2 == 0)
            if (my_is_player0 and winner == 0) or (not my_is_player0 and winner == 1):
                wins += 1
            total_reward += ep_reward

            if game_idx % 10 == 0:
                run.write("progress", game=game_idx, wins=wins,
                          win_rate=wins / max(1, game_idx + 1))

        n = max(1, n_games)
        return {
            "n_games": n_games,
            "wins": wins,
            "losses": n_games - wins,
            "win_rate": wins / n,
            "avg_reward": total_reward / n,
        }

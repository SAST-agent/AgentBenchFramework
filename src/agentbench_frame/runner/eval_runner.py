"""
BaseEvalRunner — evaluation runner.

Runs fixed evaluation against baseline opponents with tracking.
"""

from typing import Any, Dict, List, Optional

from agentbench_frame.runner.base import BaseRunner
from agentbench_frame.tracking.run import Run


class BaseEvalRunner(BaseRunner):
    """Runner for agent evaluation experiments.

    Usage:
        runner = BaseEvalRunner(game="generals", agent="ppo-v1",
                                opponents=["random", "aggressive"])
        runner.run()
    """

    def __init__(self, opponents: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.opponents = opponents or ["random"]

    def _create_env(self):
        from agentbench_frame.env.base import EnvMode
        from agentbench_frame.env.registry import make_env
        return make_env(self.game, mode=EnvMode.DIRECT)

    def _create_agent(self):
        from agentbench_frame.agent.rl_agent import RLAgent
        from agentbench_frame.agent.registry import AgentRegistry
        from agentbench_frame.agent.random import RandomAgent  # noqa

        try:
            return AgentRegistry.create(self.agent_name)
        except KeyError:
            return RLAgent(name=self.agent_name)

    def _execute(self, run: Run) -> Dict[str, Any]:
        from agentbench_frame.agent.base import RandomAgent
        from agentbench_frame.agent.registry import AgentRegistry
        from agentbench_frame.arena.match import Match

        env = run._tracked_env
        agent = run._timed_agent

        if env is None:
            raise RuntimeError("No environment available")

        n_games = self.config.get("n_eval_games", 50)
        all_results = {}

        for opp_name in self.opponents:
            try:
                opp = AgentRegistry.create(opp_name)
            except KeyError:
                opp = RandomAgent(name=opp_name)

            match = Match(env, agent, opp, seed=42)
            result = match.run(n_games=n_games)

            all_results[opp_name] = {
                "win_rate": result.win_rate,
                "wins": result.agent1_wins,
                "losses": result.agent2_wins,
                "draws": result.draws,
                "avg_game_length": result.avg_game_length,
            }

            run.write("eval_result", opponent=opp_name,
                      win_rate=result.win_rate,
                      games=n_games)

        return {"eval_results": all_results}

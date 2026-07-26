"""
BaseEvalRunner — evaluation runner.

Runs fixed evaluation against baseline opponents with tracking.
"""

from dataclasses import asdict
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

    run_type = "eval"

    def __init__(self, opponents: Optional[List[str]] = None,
                 benchmark_spec=None, **kwargs):
        super().__init__(**kwargs)
        self.opponents = opponents or ["random"]
        self.benchmark_spec = benchmark_spec or self.config.get("benchmark_spec")

    def _create_env(self):
        from agentbench_frame.env.base import EnvMode
        from agentbench_frame.env.registry import make_env
        return make_env(self.game, mode=EnvMode.DIRECT)

    def _create_agent(self):
        from agentbench_frame.agent.rl_agent import RLAgent
        from agentbench_frame.agent.registry import AgentRegistry
        from agentbench_frame.agent.base import RandomAgent  # noqa

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

        if self.benchmark_spec is not None:
            return self._execute_benchmark(run, env, agent)

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

    def _execute_benchmark(self, run: Run, env, agent) -> Dict[str, Any]:
        """Execute exactly the cases in a frozen ``BenchmarkSpec``."""
        from agentbench_frame.agent.base import RandomAgent
        from agentbench_frame.agent.registry import AgentRegistry
        from agentbench_frame.arena.match import Match
        from agentbench_frame.eval.benchmark import GameResult, evaluate_benchmark

        spec = self.benchmark_spec
        raw_results = []
        for case in spec.cases:
            try:
                try:
                    opponent = AgentRegistry.create(case.opponent)
                except KeyError:
                    opponent = RandomAgent(name=case.opponent)

                # ``first_player`` identifies the environment side occupied by
                # the evaluated agent. Match's agent1 is always environment
                # player 0 when alternate_starts is disabled.
                if case.first_player == 0:
                    side0, side1 = agent, opponent
                elif case.first_player == 1:
                    side0, side1 = opponent, agent
                else:
                    raise ValueError("benchmark first_player must be 0 or 1")

                match = Match(env, side0, side1, alternate_starts=False, seed=case.seed)
                match_result = match.run(n_games=1)
                game = match_result.game_results[0]
                winner = game.get("winner", -1)
                if winner == case.first_player:
                    outcome = "win"
                elif winner in (0, 1):
                    outcome = "loss"
                else:
                    outcome = "draw"
                result = GameResult(
                    case_id=case.case_id,
                    outcome=outcome,
                    metadata={"game_result": game},
                )
                run.log_episode(
                    reward=float(game.get("agent1_reward", 0.0)
                                 if case.first_player == 0
                                 else game.get("agent2_reward", 0.0)),
                    steps=int(game.get("steps", 0)),
                    winner=winner,
                    agent_player_id=case.first_player,
                )
                if hasattr(run, "log_budget"):
                    run.log_budget(
                        "evaluation",
                        episodes=0,
                        env_steps=0,
                        time_s=match_result.duration_seconds,
                    )
            except Exception as exc:
                result = GameResult(
                    case_id=case.case_id,
                    outcome="draw",
                    valid=False,
                    error=str(exc),
                )
            raw_results.append(result)
            run.write(
                "benchmark_game_result",
                benchmark_version=spec.version,
                case_id=result.case_id,
                outcome=result.outcome,
                valid=result.valid,
                error=result.error,
            )

        evaluation = evaluate_benchmark(spec, raw_results)
        return {
            "benchmark_version": spec.version,
            "evaluation_status": evaluation.status,
            "benchmark_score": evaluation.score,
            "wins": evaluation.wins,
            "losses": evaluation.losses,
            "draws": evaluation.draws,
            "benchmark_results": [asdict(result) for result in raw_results],
        }

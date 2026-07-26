"""
BaseRunner — abstract runner that all execution runners inherit from.

Provides the run() template method: creates a Run, wraps env/agent,
then calls _execute() for the subclass-specific logic.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agentbench_frame.tracking.run import Run


class BaseRunner(ABC):
    """Abstract base class for all experiment runners.

    Subclasses: BaseRLRunner, BaseRuleRunner, BaseEvalRunner.

    Usage:
        runner = MyRunner(game="generals", agent="ppo-v1")
        runner.run()
    """

    run_type = "eval"

    def __init__(self,
                 game: str,
                 agent: str,
                 data_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self.game = game
        self.agent_name = agent
        self.data_dir = data_dir or "./runs"
        self.config = config or {}
        self.kwargs = kwargs

    @abstractmethod
    def _execute(self, run: Run) -> Dict[str, Any]:
        """Subclass hook: execute the run logic.

        Args:
            run: The active Run instance (writer, sampler already set up).

        Returns:
            Result dict merged into summary.
        """
        ...

    def run(self) -> Dict[str, Any]:
        """Entry point: create Run, wrap env/agent, execute, finish.

        Returns:
            Summary dict written to summary.json.
        """
        run = Run.start(
            game=self.game,
            agent=self.agent_name,
            run_type=self.config.get("run_type", self.run_type),
            data_dir=self.data_dir,
            config=self.config,
        )

        env = self._create_env()
        agent = self._create_agent()
        if env is not None:
            tracked_env = run.wrap_env(env)
        else:
            tracked_env = None
        if agent is not None:
            timed_agent = run.wrap_agent(agent)
        else:
            timed_agent = None

        run.start_sampler(interval_s=5)

        t0 = time.time()
        try:
            result = self._execute(run)
        except Exception as e:
            run.write("error", error=str(e))
            result = {"error": str(e)}

        result["wall_time_s"] = time.time() - t0
        summary = run.finish(extra_summary=result)
        return summary

    def _create_env(self):
        """Override to provide the environment instance."""
        return None

    def _create_agent(self):
        """Override to provide the agent instance."""
        return None

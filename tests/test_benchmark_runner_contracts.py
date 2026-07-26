import unittest
from unittest import mock
from types import SimpleNamespace


class _Observation:
    def __init__(self, winner=1):
        self.player_id = 1
        self.round_num = 1
        self.state = {"winner": winner}

    def to_dict(self):
        return {"state": self.state, "player_id": self.player_id}


class _OneStepEnv:
    def reset(self, seed=None):
        return _Observation()

    def step(self, action):
        return _Observation(), 1.0, True, {"termination_reason": "rule"}

    def get_trajectory(self):
        return []


class BenchmarkRunnerContractTests(unittest.TestCase):
    def test_eval_runner_can_execute_a_frozen_case_list_and_persist_aggregate(self):
        from agentbench_frame.eval.benchmark import BenchmarkCase, BenchmarkSpec
        from agentbench_frame.runner.eval_runner import BaseEvalRunner

        evaluated = SimpleNamespace(
            name="evaluated",
            act=lambda _observation: [[8]],
            reset=lambda: None,
        )
        opponent = SimpleNamespace(
            name="opponent",
            act=lambda _observation: [[8]],
            reset=lambda: None,
        )
        run = SimpleNamespace(
            _tracked_env=_OneStepEnv(),
            _timed_agent=evaluated,
            log_episode=lambda **_kwargs: None,
            write=lambda *_args, **_kwargs: None,
            log_budget=lambda *_args, **_kwargs: None,
        )
        spec = BenchmarkSpec(
            "bench-v1",
            [BenchmarkCase("case-0", "opponent", 7, 0)],
        )
        runner = BaseEvalRunner(
            game="test",
            agent="evaluated",
            benchmark_spec=spec,
        )

        with mock.patch(
            "agentbench_frame.agent.registry.AgentRegistry.create",
            return_value=opponent,
        ):
            summary = runner._execute(run)

        self.assertEqual(summary["evaluation_status"], "complete")
        self.assertAlmostEqual(summary["benchmark_score"], 0.0)
        self.assertEqual(summary["benchmark_version"], "bench-v1")
        self.assertEqual(len(summary["benchmark_results"]), 1)


if __name__ == "__main__":
    unittest.main()

import unittest


class BenchmarkContractTests(unittest.TestCase):
    def test_complete_benchmark_uses_half_point_for_draws(self):
        from agentbench_frame.eval.benchmark import (
            BenchmarkCase,
            BenchmarkSpec,
            GameResult,
            evaluate_benchmark,
        )

        spec = BenchmarkSpec(
            version="bench-v1",
            cases=[
                BenchmarkCase("c1", "weak", 1, 0),
                BenchmarkCase("c2", "strong", 2, 1),
                BenchmarkCase("c3", "strong", 3, 0),
            ],
        )
        evaluation = evaluate_benchmark(
            spec,
            [
                GameResult("c1", "win"),
                GameResult("c2", "draw"),
                GameResult("c3", "loss"),
            ],
        )

        self.assertEqual(evaluation.status, "complete")
        self.assertEqual((evaluation.wins, evaluation.losses, evaluation.draws), (1, 1, 1))
        self.assertAlmostEqual(evaluation.score, 0.5)

    def test_incomplete_benchmark_has_no_aggregate_score(self):
        from agentbench_frame.eval.benchmark import (
            BenchmarkCase,
            BenchmarkSpec,
            GameResult,
            evaluate_benchmark,
        )

        spec = BenchmarkSpec("bench-v1", [BenchmarkCase("c1", "weak", 1, 0)])
        evaluation = evaluate_benchmark(spec, [GameResult("c1", "win", valid=False, error="timeout")])

        self.assertEqual(evaluation.status, "incomplete")
        self.assertIsNone(evaluation.score)
        self.assertEqual(evaluation.per_case["c1"].error, "timeout")

    def test_benchmark_spec_rejects_duplicate_case_ids(self):
        from agentbench_frame.eval.benchmark import BenchmarkCase, BenchmarkSpec

        with self.assertRaises(ValueError):
            BenchmarkSpec(
                "bench-v1",
                [BenchmarkCase("same", "a", 1, 0), BenchmarkCase("same", "b", 2, 1)],
            )

    def test_trapezoid_auc_uses_only_adjacent_complete_points(self):
        from agentbench_frame.eval.curves import trapezoid_auc

        self.assertAlmostEqual(
            trapezoid_auc([(0.0, 0.0), (2.0, 1.0), (3.0, None), (5.0, 0.5), (7.0, 1.0)]),
            1.0 + 1.5,
        )


class InformationGainContractTests(unittest.TestCase):
    def test_deterministic_action_uses_only_the_complete_declared_support(self):
        from agentbench_frame.eval.information_gain import (
            deterministic_measurement_distribution,
            policy_kl,
        )
        from agentbench_frame.eval.measurement import ActionSupport

        support = ActionSupport(("0", "1", "2", "3", "4"))
        old = deterministic_measurement_distribution("1", support, epsilon=0.05)
        same = deterministic_measurement_distribution("1", support, epsilon=0.05)
        new = deterministic_measurement_distribution("4", support, epsilon=0.05)

        self.assertEqual(
            old,
            {"0": 0.01, "1": 0.96, "2": 0.01, "3": 0.01, "4": 0.01},
        )
        self.assertAlmostEqual(
            policy_kl(list(same.values()), list(old.values())),
            0.0,
        )
        self.assertGreater(policy_kl(list(new.values()), list(old.values())), 0.0)
        self.assertLess(policy_kl(list(new.values()), list(old.values())), float("inf"))

        with self.assertRaises(ValueError):
            deterministic_measurement_distribution("target-nearest-bean", support, epsilon=0.05)

    def test_epsilon_regularization_gives_finite_common_support(self):
        from agentbench_frame.eval.information_gain import epsilon_regularize, policy_kl

        self.assertEqual(len(epsilon_regularize([1.0, 0.0], epsilon=0.1)), 2)
        self.assertAlmostEqual(epsilon_regularize([1.0, 0.0], epsilon=0.1)[0], 0.95)
        self.assertAlmostEqual(epsilon_regularize([1.0, 0.0], epsilon=0.1)[1], 0.05)
        self.assertAlmostEqual(policy_kl([1.0, 0.0], [1.0, 0.0], epsilon=0.1), 0.0)

    def test_episode_trace_is_recorded_at_real_contexts(self):
        from agentbench_frame.eval.information_gain import episode_policy_kl_trace

        new_policy = lambda context: {"a": 0.75, "b": 0.25} if context == "s0" else {"a": 0.5, "b": 0.5}
        old_policy = lambda context: {"a": 0.5, "b": 0.5}
        trace = episode_policy_kl_trace(
            new_policy,
            old_policy,
            contexts=["s0", "s1"],
            legal_actions=lambda _context: ["a", "b"],
            epsilon=0.01,
        )

        self.assertEqual(len(trace), 2)
        self.assertGreater(trace[0], 0.0)
        self.assertAlmostEqual(trace[1], 0.0)

    def test_occupancy_shift_and_trajectory_summary_are_derived(self):
        from agentbench_frame.eval.information_gain import occupancy_shift, trajectory_kl_from_trace

        self.assertAlmostEqual(occupancy_shift(["a", "b"], ["a", "b"]), 0.0)
        self.assertGreater(occupancy_shift(["a", "a"], ["b", "b"]), 0.0)
        self.assertAlmostEqual(trajectory_kl_from_trace([0.1, 0.2, 0.3]), 0.6)


if __name__ == "__main__":
    unittest.main()

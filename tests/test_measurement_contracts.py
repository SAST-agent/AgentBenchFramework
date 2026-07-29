import unittest
from decimal import Decimal


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
    def test_canonical_action_space_allows_count_only_support(self):
        from agentbench_frame.eval.action_space import CanonicalActionSpace

        class CountOnlySpace:
            spec_id = "toy-v1"

            def canonicalize(self, state, action):
                return tuple(tuple(command) for command in action)

            def contains(self, state, action):
                return True

            def cardinality(self, state):
                return 7

        self.assertIsInstance(CountOnlySpace(), CanonicalActionSpace)

    def test_deterministic_uniform_kl_matches_explicit_distribution(self):
        from agentbench_frame.eval.information_gain import (
            policy_kl,
            uniform_smoothed_deterministic_kl,
        )

        epsilon = Decimal("0.01")
        support_size = 5
        measured = uniform_smoothed_deterministic_kl(
            new_action=("b",),
            old_action=("a",),
            support_size=support_size,
            epsilon=epsilon,
        )
        q = 1.0 - float(epsilon) + float(epsilon) / support_size
        r = float(epsilon) / support_size
        explicit = policy_kl([r, q, r, r, r], [q, r, r, r, r])
        self.assertAlmostEqual(float(measured), explicit)

    def test_deterministic_uniform_kl_is_zero_for_equal_action(self):
        from agentbench_frame.eval.information_gain import (
            uniform_smoothed_deterministic_kl,
        )

        self.assertEqual(
            uniform_smoothed_deterministic_kl(
                new_action=((8,),),
                old_action=((8,),),
                support_size=10**30,
                epsilon="0.01",
            ),
            Decimal(0),
        )

    def test_deterministic_uniform_kl_rejects_invalid_support(self):
        from agentbench_frame.eval.information_gain import (
            uniform_smoothed_deterministic_kl,
        )

        with self.assertRaisesRegex(ValueError, "support_size"):
            uniform_smoothed_deterministic_kl("b", "a", 0, "0.01")
        with self.assertRaisesRegex(ValueError, "one legal action"):
            uniform_smoothed_deterministic_kl("b", "a", 1, "0.01")

    def test_deterministic_uniform_kl_rejects_invalid_epsilon(self):
        from agentbench_frame.eval.information_gain import (
            uniform_smoothed_deterministic_kl,
        )

        for epsilon in ("0", "1", "-0.01", "NaN"):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "epsilon"):
                    uniform_smoothed_deterministic_kl("b", "a", 2, epsilon)

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

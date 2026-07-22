import contextlib
import io
import unittest
from pathlib import Path

from agentbench_frame.aquawar.cli import build_parser, main, parse_opponent
from agentbench_frame.aquawar.evaluator import AquaWarEvaluationResult, Opponent


class FakeEvaluator:
    instances = []
    error_count = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def evaluate(self):
        return AquaWarEvaluationResult(
            run_dir=Path("/tmp/fake-run"),
            summary={},
            matches=[],
            error_count=self.error_count,
        )


class CliTest(unittest.TestCase):
    def setUp(self):
        FakeEvaluator.instances.clear()
        FakeEvaluator.error_count = 0

    def test_parse_opponent_splits_only_first_equals(self):
        self.assertEqual(
            parse_opponent("rank01=python ai.py --mode=a"),
            Opponent("rank01", "python ai.py --mode=a"),
        )

    def test_parse_opponent_rejects_missing_name_or_command(self):
        for value in ("rank01", "=command", "rank01="):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NAME=COMMAND"):
                    parse_opponent(value)

    def test_parser_accepts_all_evaluation_options(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--logic",
                "logic",
                "--candidate-name",
                "candidate-v2",
                "--candidate",
                "candidate-command",
                "--opponent",
                "alpha=alpha-command",
                "--opponent",
                "beta=beta-command --flag=x",
                "--pairs",
                "7",
                "--seats",
                "1",
                "--timeout",
                "2.5",
                "--data-dir",
                "/tmp/results",
                "--save-replays",
                "--save-traces",
            ]
        )

        self.assertEqual(args.logic, "logic")
        self.assertEqual(args.candidate_name, "candidate-v2")
        self.assertEqual(args.candidate, "candidate-command")
        self.assertEqual(
            args.opponent,
            [
                Opponent("alpha", "alpha-command"),
                Opponent("beta", "beta-command --flag=x"),
            ],
        )
        self.assertEqual(args.pairs, 7)
        self.assertEqual(args.seats, "1")
        self.assertEqual(args.timeout, 2.5)
        self.assertEqual(args.data_dir, Path("/tmp/results"))
        self.assertTrue(args.save_replays)
        self.assertTrue(args.save_traces)

    def test_main_invokes_evaluator_and_prints_run_dir(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "--logic",
                    "logic",
                    "--candidate-name",
                    "candidate",
                    "--candidate",
                    "candidate-command",
                    "--opponent",
                    "sample=sample-command",
                    "--pairs",
                    "2",
                ],
                evaluator_class=FakeEvaluator,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "/tmp/fake-run")
        self.assertEqual(
            FakeEvaluator.instances[0].kwargs,
            {
                "logic_command": "logic",
                "candidate_name": "candidate",
                "candidate_command": "candidate-command",
                "opponents": [Opponent("sample", "sample-command")],
                "pairs": 2,
                "seats": "both",
                "timeout": 5.0,
                "data_dir": None,
                "save_replays": False,
                "save_traces": False,
            },
        )

    def test_main_returns_nonzero_when_any_match_errors(self):
        FakeEvaluator.error_count = 1
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "--logic",
                    "logic",
                    "--candidate-name",
                    "candidate",
                    "--candidate",
                    "candidate",
                    "--opponent",
                    "sample=sample",
                ],
                evaluator_class=FakeEvaluator,
            )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

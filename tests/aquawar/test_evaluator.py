import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from agentbench_frame.aquawar.evaluator import AquaWarEvaluator, Opponent
from agentbench_frame.aquawar.match import AquaWarMatchError


class DeterministicMatchRunner:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        logic_command,
        ai_commands,
        timeout,
        replay_path,
        trace_path=None,
    ):
        index = len(self.calls)
        self.calls.append(
            {
                "logic": logic_command,
                "ais": list(ai_commands),
                "timeout": timeout,
                "replay": replay_path,
                "trace": trace_path,
            }
        )
        if index == 3:
            raise AquaWarMatchError("fixture failure")
        winners = [0, 0, None, None, 0, 1, 1, None]
        winner = winners[index]
        end_info = {"0": 50, "1": 50}
        if winner is not None:
            end_info = {"0": 100, "1": 0} if winner == 0 else {"0": 0, "1": 100}
        return {"winner": winner, "end_info": end_info, "turns": 10 + index}


class EvaluatorTest(unittest.TestCase):
    def test_schedules_ab_ba_and_writes_standard_result(self):
        runner = DeterministicMatchRunner()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            evaluator = AquaWarEvaluator(
                logic_command="logic-command",
                candidate_name="candidate",
                candidate_command="candidate-command",
                opponents=[
                    Opponent("alpha", "alpha-command"),
                    Opponent("beta", "beta-command"),
                ],
                pairs=2,
                seats="both",
                timeout=1.25,
                data_dir=data_dir,
                match_runner=runner,
            )

            result = evaluator.evaluate()

            self.assertEqual(len(runner.calls), 8)
            self.assertEqual(
                [call["ais"] for call in runner.calls],
                [
                    ["candidate-command", "alpha-command"],
                    ["alpha-command", "candidate-command"],
                    ["candidate-command", "alpha-command"],
                    ["alpha-command", "candidate-command"],
                    ["candidate-command", "beta-command"],
                    ["beta-command", "candidate-command"],
                    ["candidate-command", "beta-command"],
                    ["beta-command", "candidate-command"],
                ],
            )
            self.assertTrue(all(call["logic"] == "logic-command" for call in runner.calls))
            self.assertTrue(all(call["timeout"] == 1.25 for call in runner.calls))
            self.assertTrue(all(call["trace"] is None for call in runner.calls))

            run_dir = result.run_dir
            self.assertTrue((run_dir / "run.toml").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "events.jsonl").is_file())
            self.assertTrue((run_dir / "matches.jsonl").is_file())

            metadata = tomllib.loads((run_dir / "run.toml").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            matches = [
                json.loads(line)
                for line in (run_dir / "matches.jsonl").read_text().splitlines()
            ]

            self.assertEqual(metadata["run"]["game"], "25_aquawar")
            self.assertEqual(metadata["run"]["agent"], "candidate")
            self.assertEqual(metadata["run"]["type"], "eval")
            self.assertEqual(summary["game"], "25_aquawar")
            self.assertEqual(summary["agent"], "candidate")
            self.assertEqual(summary["run_type"], "eval")
            self.assertEqual(summary["total_episodes"], 8)
            self.assertEqual(summary["total_steps"], 95)
            self.assertAlmostEqual(summary["win_rate"], 4 / 7)
            self.assertEqual(
                summary["h2h"],
                {"candidate": {"alpha": 0.5, "beta": 0.625}},
            )
            self.assertEqual(
                summary["aquawar"]["aggregate"],
                {
                    "wins": 3,
                    "draws": 2,
                    "losses": 2,
                    "errors": 1,
                    "valid_games": 7,
                    "attempted_games": 8,
                    "score_rate": 4 / 7,
                    "avg_turns": 95 / 7,
                },
            )
            self.assertEqual(summary["aquawar"]["by_seat"]["0"]["wins"], 2)
            self.assertAlmostEqual(summary["aquawar"]["by_seat"]["0"]["score_rate"], 0.625)
            self.assertEqual(summary["aquawar"]["by_seat"]["1"]["errors"], 1)
            self.assertAlmostEqual(summary["aquawar"]["by_seat"]["1"]["score_rate"], 0.5)
            self.assertEqual(len(matches), 8)
            self.assertEqual(matches[3]["candidate_result"], "error")
            self.assertEqual(result.error_count, 1)
            self.assertEqual(result.summary, summary)

    def test_artifact_paths_are_relative_to_run_directory(self):
        def successful_runner(
            logic_command,
            ai_commands,
            timeout,
            replay_path,
            trace_path=None,
        ):
            replay_path.write_text("{}")
            self.assertIsNotNone(trace_path)
            trace_path.write_text("{}\n")
            return {"winner": 0, "end_info": {"0": 1, "1": 0}, "turns": 2}

        with tempfile.TemporaryDirectory() as directory:
            result = AquaWarEvaluator(
                logic_command="logic",
                candidate_name="candidate",
                candidate_command="candidate",
                opponents=[Opponent("sample", "sample")],
                pairs=1,
                seats="0",
                data_dir=Path(directory),
                save_replays=True,
                save_traces=True,
                match_runner=successful_runner,
            ).evaluate()

            record = json.loads((result.run_dir / "matches.jsonl").read_text())
            self.assertEqual(record["replay"], "artifacts/sample-pair000-seat0.json")
            self.assertEqual(record["trace"], "artifacts/sample-pair000-seat0.trace.jsonl")
            self.assertTrue((result.run_dir / record["replay"]).is_file())
            self.assertTrue((result.run_dir / record["trace"]).is_file())

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least one opponent"):
            AquaWarEvaluator("logic", "candidate", "candidate", [])
        with self.assertRaisesRegex(ValueError, "pairs must be positive"):
            AquaWarEvaluator(
                "logic", "candidate", "candidate", [Opponent("sample", "sample")], pairs=0
            )
        with self.assertRaisesRegex(ValueError, "seats must be"):
            AquaWarEvaluator(
                "logic",
                "candidate",
                "candidate",
                [Opponent("sample", "sample")],
                seats="invalid",
            )
        with self.assertRaisesRegex(ValueError, "candidate name"):
            AquaWarEvaluator(
                "logic",
                "../candidate",
                "candidate",
                [Opponent("sample", "sample")],
            )
        with self.assertRaisesRegex(ValueError, "opponent name"):
            AquaWarEvaluator(
                "logic",
                "candidate",
                "candidate",
                [Opponent("../sample", "sample")],
            )
        with self.assertRaisesRegex(ValueError, "opponent names must be unique"):
            AquaWarEvaluator(
                "logic",
                "candidate",
                "candidate",
                [Opponent("sample", "one"), Opponent("sample", "two")],
            )
        for field, kwargs in (
            ("logic command", {"logic_command": ""}),
            ("candidate command", {"candidate_command": ""}),
            ("opponent command", {"opponents": [Opponent("sample", "")]}),
        ):
            values = {
                "logic_command": "logic",
                "candidate_name": "candidate",
                "candidate_command": "candidate",
                "opponents": [Opponent("sample", "sample")],
            }
            values.update(kwargs)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    AquaWarEvaluator(**values)


if __name__ == "__main__":
    unittest.main()

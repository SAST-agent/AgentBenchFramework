import json
import tempfile
import unittest
from pathlib import Path

from agentbench_frame.lostspace.evaluator import LostSpaceEvaluator, Opponent
from agentbench_frame.lostspace.match import LostSpaceMatchError


def _parse_run_toml(text):
    """Minimal [run]-section parser (avoids the Python 3.11-only tomllib)."""
    meta = {}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            continue
        if "=" in line and section == "run":
            key, _, val = line.partition("=")
            meta[key.strip()] = val.strip().strip('"')
    return {"run": meta}


class DeterministicMatchRunner:
    """Returns canned 4-player rankings; call 3 errors out."""

    # candidate's finishing rank (1=1st ... 4=4th) per call index; None at idx 3
    candidate_ranks = [1, 4, 2, None, 1, 3, 4, 2]

    def __init__(self):
        self.calls = []

    def __call__(self, logic_command, ai_commands, timeout, replay_path, trace_path=None, **kwargs):
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
            raise LostSpaceMatchError("fixture failure")
        candidate_seat = ai_commands.index("candidate-command")
        cand_rank = self.candidate_ranks[index]
        others = [s for s in range(4) if s != candidate_seat]
        ranking = [None, None, None, None]
        ranking[cand_rank - 1] = candidate_seat
        for spot, pid in zip([p for p in range(4) if p != cand_rank - 1], others):
            ranking[spot] = pid
        end_info = {str(seat): 4 - ranking.index(seat) for seat in range(4)}
        return {
            "winner": ranking[0],
            "ranking": ranking,
            "end_info": end_info,
            "turns": 100 + index,
        }


class EvaluatorTest(unittest.TestCase):
    def test_schedules_four_seats_and_writes_standard_result(self):
        runner = DeterministicMatchRunner()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            evaluator = LostSpaceEvaluator(
                logic_command="logic-command",
                candidate_name="candidate",
                candidate_command="candidate-command",
                opponents=[
                    Opponent("alpha", "alpha-command"),
                    Opponent("beta", "beta-command"),
                ],
                filler_command="filler-command",
                pairs=1,
                seats="all",
                timeout=1.25,
                data_dir=data_dir,
                match_runner=runner,
            )

            result = evaluator.evaluate()

            self.assertEqual(len(runner.calls), 8)

            # candidate rotates through seats 0..3 per opponent; opponent occupies
            # one other seat; the remaining two seats are filler.
            for call in runner.calls:
                ais = call["ais"]
                self.assertEqual(ais.count("candidate-command"), 1)
                self.assertEqual(ais.count("filler-command"), 2)
                self.assertEqual(
                    sorted(
                        ais.count(cmd)
                        for cmd in ("alpha-command", "beta-command")
                    ),
                    [0, 1],
                )
            self.assertTrue(all(call["logic"] == "logic-command" for call in runner.calls))
            self.assertTrue(all(call["timeout"] == 1.25 for call in runner.calls))
            self.assertTrue(all(call["trace"] is None for call in runner.calls))

            run_dir = result.run_dir
            self.assertTrue((run_dir / "run.toml").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "events.jsonl").is_file())
            self.assertTrue((run_dir / "matches.jsonl").is_file())

            metadata = _parse_run_toml((run_dir / "run.toml").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            matches = [
                json.loads(line)
                for line in (run_dir / "matches.jsonl").read_text().splitlines()
            ]

            self.assertEqual(metadata["run"]["game"], "25_lostspace")
            self.assertEqual(metadata["run"]["agent"], "candidate")
            self.assertEqual(metadata["run"]["type"], "eval")
            self.assertEqual(summary["game"], "25_lostspace")
            self.assertEqual(summary["agent"], "candidate")
            self.assertEqual(summary["run_type"], "eval")
            self.assertEqual(summary["total_episodes"], 8)

            agg = summary["lostspace"]["aggregate"]
            self.assertEqual(agg["wins"], 2)
            self.assertEqual(agg["losses"], 5)
            self.assertEqual(agg["errors"], 1)
            self.assertEqual(agg["valid_games"], 7)
            self.assertEqual(agg["attempted_games"], 8)
            self.assertAlmostEqual(agg["win_rate"], 2 / 7)
            self.assertAlmostEqual(agg["avg_rank"], 17 / 7)
            self.assertAlmostEqual(agg["avg_score"], 18 / 7)
            # turns for valid indices 0,1,2,4,5,6,7
            self.assertAlmostEqual(agg["avg_turns"], 725 / 7)
            self.assertAlmostEqual(summary["win_rate"], 2 / 7)
            self.assertEqual(
                summary["h2h"], {"candidate": {"alpha": 1 / 3, "beta": 1 / 4}}
            )
            self.assertEqual(summary["lostspace"]["by_opponent"]["alpha"]["valid_games"], 3)
            self.assertAlmostEqual(summary["lostspace"]["by_opponent"]["alpha"]["win_rate"], 1 / 3)
            self.assertAlmostEqual(summary["lostspace"]["by_seat"]["0"]["win_rate"], 1.0)
            self.assertEqual(summary["lostspace"]["by_seat"]["3"]["valid_games"], 1)
            self.assertEqual(len(matches), 8)
            self.assertEqual(matches[3]["candidate_result"], "error")
            self.assertEqual(result.error_count, 1)
            self.assertEqual(result.summary, summary)

    def test_artifact_paths_are_relative_to_run_directory(self):
        def successful_runner(logic_command, ai_commands, timeout, replay_path, trace_path=None, **kwargs):
            replay_path.write_text("[]")
            self.assertIsNotNone(trace_path)
            trace_path.write_text("{}\n")
            return {
                "winner": 0,
                "ranking": [0, 1, 2, 3],
                "end_info": {"0": 4, "1": 3, "2": 2, "3": 1},
                "turns": 2,
            }

        with tempfile.TemporaryDirectory() as directory:
            result = LostSpaceEvaluator(
                logic_command="logic",
                candidate_name="candidate",
                candidate_command="candidate",
                opponents=[Opponent("sample", "sample")],
                filler_command="filler",
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
        good = dict(
            logic_command="logic",
            candidate_name="candidate",
            candidate_command="candidate",
            opponents=[Opponent("sample", "sample")],
            filler_command="filler",
        )
        with self.assertRaisesRegex(ValueError, "at least one opponent"):
            LostSpaceEvaluator(**{**good, "opponents": []})
        with self.assertRaisesRegex(ValueError, "filler command must not be empty"):
            LostSpaceEvaluator(**{**good, "filler_command": "  "})
        with self.assertRaisesRegex(ValueError, "pairs must be positive"):
            LostSpaceEvaluator(**{**good, "pairs": 0})
        with self.assertRaisesRegex(ValueError, "seats must be"):
            LostSpaceEvaluator(**{**good, "seats": "invalid"})
        with self.assertRaisesRegex(ValueError, "candidate name"):
            LostSpaceEvaluator(**{**good, "candidate_name": "../candidate"})
        with self.assertRaisesRegex(ValueError, "opponent names must be unique"):
            LostSpaceEvaluator(
                **{**good, "opponents": [Opponent("sample", "a"), Opponent("sample", "b")]}
            )


if __name__ == "__main__":
    unittest.main()

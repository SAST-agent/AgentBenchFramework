import json
import os
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path

from agentbench_frame.aquawar.match import AquaWarMatchError, run_match


FIXTURES = Path(__file__).parent / "fixtures"


def command(script: str, *args: str) -> str:
    parts = [sys.executable, str(FIXTURES / script), *args]
    return " ".join(shlex.quote(part) for part in parts)


class MatchTest(unittest.TestCase):
    def test_routes_one_turn_and_returns_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"

            result = run_match(
                command("fake_logic.py"),
                [command("fake_ai.py"), command("fake_ai.py")],
                timeout=0.5,
                replay_path=root / "replay.json",
                trace_path=trace,
            )

            self.assertEqual(
                result,
                {"winner": 0, "end_info": {"0": 100, "1": 0}, "turns": 1},
            )
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([record["type"] for record in records], ["observation", "action"])
            self.assertEqual(records[0]["content"], {"turn": 1})
            self.assertEqual(records[1]["content"], {"action": "pass"})

    def test_reports_malformed_logic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AquaWarMatchError, "logic emitted invalid JSON"):
                run_match(
                    command("fake_logic.py", "--mode", "malformed"),
                    [command("fake_ai.py"), command("fake_ai.py")],
                    timeout=0.5,
                    replay_path=Path(directory) / "replay.json",
                )

    def test_reports_ai_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AquaWarMatchError, "player 0 timed out"):
                run_match(
                    command("fake_logic.py"),
                    [command("fake_ai.py", "--sleep", "0.5"), command("fake_ai.py")],
                    timeout=0.1,
                    replay_path=Path(directory) / "replay.json",
                )

    def test_cleans_up_ai_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_paths = [root / "ai0.pid", root / "ai1.pid"]
            run_match(
                command("fake_logic.py", "--delay", "0.1"),
                [
                    command("fake_ai.py", "--pid-file", str(pid_paths[0])),
                    command("fake_ai.py", "--pid-file", str(pid_paths[1])),
                ],
                timeout=0.5,
                replay_path=root / "replay.json",
            )

            for path in pid_paths:
                pid = int(path.read_text())
                deadline = time.monotonic() + 1
                while True:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    if time.monotonic() >= deadline:
                        self.fail(f"AI process {pid} was not cleaned up")
                    time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agentbench_frame.lostspace.match import LostSpaceMatchError, run_match


FIXTURES = Path(__file__).parent / "fixtures"


def command(script: str, *args: str) -> str:
    parts = [sys.executable, str(FIXTURES / script), *args]
    # shlex.quote emits POSIX single-quotes which cmd.exe does not understand;
    # use subprocess.list2cmdline on Windows so the shell=True launcher works.
    if _is_windows():
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def process_alive(pid: int) -> bool:
    """Cross-platform 'is this pid still running?' check."""
    if _is_windows():
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class MatchTest(unittest.TestCase):
    def _four_ais(self, *extra: str) -> list[str]:
        return [command("fake_ai.py", *extra) for _ in range(4)]

    def test_routes_one_turn_and_returns_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"

            result = run_match(
                command("fake_logic.py"),
                self._four_ais(),
                timeout=1.0,
                replay_path=root / "replay.json",
                trace_path=trace,
            )

            self.assertEqual(result["winner"], 0)
            self.assertEqual(result["ranking"], [0, 1, 2, 3])
            self.assertEqual(
                result["end_info"], {"0": 4, "1": 3, "2": 2, "3": 1}
            )
            self.assertEqual(result["turns"], 1)

            records = [json.loads(line) for line in trace.read_text().splitlines()]
            actions = [r for r in records if r["type"] == "action"]
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["player"], 0)
            self.assertEqual(actions[0]["content"]["type"], "action")

    def test_notification_frame_does_not_wait_for_reply(self):
        # An ``other_death`` status update is addressed to the in-turn player
        # (listen=[0], player=[0]) but expects no reply: the player consumes
        # it silently. The harness must route it through without blocking on
        # a reply that never comes, or the match times out.
        with tempfile.TemporaryDirectory() as directory:
            result = run_match(
                command("fake_logic.py", "--mode", "notify"),
                self._four_ais(),
                timeout=1.0,
                replay_path=Path(directory) / "replay.json",
            )
            self.assertEqual(result["winner"], 0)
            self.assertEqual(result["turns"], 1)

    def test_reports_malformed_logic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LostSpaceMatchError, "logic emitted invalid JSON"):
                run_match(
                    command("fake_logic.py", "--mode", "malformed"),
                    self._four_ais(),
                    timeout=1.0,
                    replay_path=Path(directory) / "replay.json",
                )

    def test_reports_ai_timeout(self):
        # A player that does not answer an action request in time is reported
        # to the logic as an ``ai_error`` (TimeOutError) and the match
        # continues — mirroring the saiblo per-round TLE, instead of
        # deadlocking the whole game. The timeout is recorded in the trace.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            result = run_match(
                command("fake_logic.py"),
                [
                    command("fake_ai.py", "--sleep", "2.0"),
                    *self._four_ais()[1:],
                ],
                timeout=0.5,
                replay_path=root / "replay.json",
                trace_path=trace,
            )
            self.assertEqual(result["winner"], 0)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            errors = [
                r for r in records
                if r.get("type") == "ai_error" and r.get("player") == 0
            ]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["content"], "timeOutError")

    def test_cleans_up_ai_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_paths = [root / f"ai{i}.pid" for i in range(4)]
            run_match(
                command("fake_logic.py"),
                [
                    command("fake_ai.py", "--pid-file", str(pid_paths[i]))
                    for i in range(4)
                ],
                timeout=1.0,
                replay_path=root / "replay.json",
            )

            for path in pid_paths:
                pid = int(path.read_text())
                self.assertFalse(process_alive(pid), f"pid {pid} still alive")


if __name__ == "__main__":
    unittest.main()

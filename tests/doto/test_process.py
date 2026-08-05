import os
import sys
import time
from pathlib import Path

from agentbench_frame.doto.process import ManagedProcess


FIXTURES = Path(__file__).parent / "fixtures"


def test_managed_process_terminates_child_group(tmp_path):
    pid_path = tmp_path / "child.pid"
    process = ManagedProcess.start(
        [sys.executable, str(FIXTURES / "spawn_child.py"), str(pid_path)],
        cwd=tmp_path,
        label="tree",
    )
    deadline = time.monotonic() + 2
    while not pid_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child_pid = int(pid_path.read_text())

    process.terminate()

    deadline = time.monotonic() + 2
    while True:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"child process {child_pid} survived cleanup")
        time.sleep(0.01)

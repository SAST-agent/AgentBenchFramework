"""Owned subprocess groups for native DOTO components."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence


@dataclass
class ManagedProcess:
    process: subprocess.Popen[bytes]
    label: str
    stderr_file: BinaryIO | None = None

    @classmethod
    def start(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        label: str,
        stderr_path: Path | None = None,
    ) -> "ManagedProcess":
        stderr_file = None
        stderr: int | BinaryIO = subprocess.PIPE
        if stderr_path is not None:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_file = stderr_path.open("wb")
            stderr = stderr_file
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
        except Exception:
            if stderr_file is not None:
                stderr_file.close()
            raise
        return cls(process=process, label=label, stderr_file=stderr_file)

    @property
    def stdin(self):
        if self.process.stdin is None:
            raise RuntimeError(f"{self.label} stdin is unavailable")
        return self.process.stdin

    @property
    def stdout(self):
        if self.process.stdout is None:
            raise RuntimeError(f"{self.label} stdout is unavailable")
        return self.process.stdout

    def terminate(self, grace_seconds: float = 1.0) -> None:
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
        if self.stderr_file is not None and not self.stderr_file.closed:
            self.stderr_file.close()

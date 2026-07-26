"""Bounded local player subprocess lifecycle.

These controls improve experiment reliability. They are not a security sandbox.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence

from .models import AgentProcessSpec, ProcessLimits
from .protocol import PacketTooLarge, encode_peer_commands, parse_command_packet


class PlayerProcessError(RuntimeError):
    pass


class DecisionTimeout(PlayerProcessError):
    pass


class PrematureExit(PlayerProcessError):
    pass


class StartupFailure(PlayerProcessError):
    pass


class ManagedAgentProcess:
    def __init__(
        self,
        spec: AgentProcessSpec,
        limits: ProcessLimits,
        artifact_dir: Path,
    ) -> None:
        self.spec = spec
        self.limits = limits
        self.artifact_dir = artifact_dir
        self.process: subprocess.Popen[bytes] | None = None
        self._protocol = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_buffer = bytearray()

    def __enter__(self) -> "ManagedAgentProcess":
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        allowed = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "PYTHONPATH", "LANG"}
        }
        allowed.setdefault("LANG", "C.UTF-8")
        allowed.update({str(key): str(value) for key, value in self.spec.env.items()})
        try:
            self.process = subprocess.Popen(
                self.spec.argv,
                cwd=self.spec.cwd,
                env=allowed,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as exc:
            self._write_metadata(error=str(exc))
            raise StartupFailure(f"cannot start {self.spec.agent_id}: {exc}") from exc
        self._protocol = (self.artifact_dir / "agent.protocol.bin").open("wb")
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._write_metadata()
        return self

    def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        retained = 0
        with (self.artifact_dir / "agent.stderr.log").open("wb") as output:
            while True:
                chunk = self.process.stderr.read(8192)
                if not chunk:
                    break
                if retained < self.limits.max_artifact_bytes:
                    bounded = chunk[: self.limits.max_artifact_bytes - retained]
                    output.write(bounded)
                    retained += len(bounded)

    def _record_protocol(self, direction: bytes, payload: bytes) -> None:
        assert self._protocol is not None
        self._protocol.write(direction + struct.pack(">I", len(payload)) + payload)
        self._protocol.flush()

    def _write(self, payload: bytes) -> None:
        if self.process is None or self.process.stdin is None:
            raise PlayerProcessError("player is not running")
        if self.process.poll() is not None:
            raise PrematureExit(f"{self.spec.agent_id} exited with {self.process.returncode}")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise PrematureExit(f"{self.spec.agent_id} closed stdin") from exc
        self._record_protocol(b">", payload)

    def send_initial(self, observation: Mapping[str, object]) -> None:
        payload = json.dumps(observation, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        self._write(payload)

    def send_peer_commands(self, commands: Sequence[Sequence[int]]) -> None:
        self._write(encode_peer_commands(commands))

    def _read_exact_until(self, size: int, deadline: float) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        fd = self.process.stdout.fileno()
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)
        try:
            while len(self._stdout_buffer) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DecisionTimeout(f"{self.spec.agent_id} decision timed out")
                if not selector.select(remaining):
                    raise DecisionTimeout(f"{self.spec.agent_id} decision timed out")
                chunk = os.read(fd, max(4096, size - len(self._stdout_buffer)))
                if not chunk:
                    code = self.process.poll()
                    raise PrematureExit(f"{self.spec.agent_id} exited with {code}")
                self._stdout_buffer.extend(chunk)
        finally:
            selector.close()
        result = bytes(self._stdout_buffer[:size])
        del self._stdout_buffer[:size]
        return result

    def request_turn(self) -> tuple[tuple[int, ...], ...]:
        deadline = time.monotonic() + self.limits.decision_timeout_s
        header = self._read_exact_until(4, deadline)
        length = struct.unpack(">I", header)[0]
        if length > self.limits.max_packet_bytes:
            raise PacketTooLarge(f"packet length {length} exceeds {self.limits.max_packet_bytes}")
        payload = self._read_exact_until(length, deadline)
        self._record_protocol(b"<", header + payload)
        return parse_command_packet(payload)

    def _write_metadata(self, error: str | None = None) -> None:
        payload = {
            "agent_id": self.spec.agent_id,
            "argv": list(self.spec.argv),
            "cwd": str(self.spec.cwd),
            "pid": self.process.pid if self.process else None,
            "returncode": self.process.poll() if self.process else None,
            "error": error,
        }
        (self.artifact_dir / "process.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=1)
            except ProcessLookupError:
                pass
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)
        if self._protocol:
            self._protocol.close()
        self._write_metadata()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_baseline_process(
    workspace: Path,
    engine_root: Path,
    python_executable: Path,
    sdk_root: Path | None = None,
) -> AgentProcessSpec:
    python_path = [str(workspace), str(engine_root)]
    if sdk_root is not None:
        python_path.append(str(sdk_root))
    return AgentProcessSpec(
        agent_id="baseline",
        argv=(str(python_executable), str(workspace / "main.py")),
        cwd=workspace,
        env={"PYTHONPATH": os.pathsep.join(python_path), "PYTHONUNBUFFERED": "1"},
    )


def build_calibration_process(
    source_root: Path,
    engine_root: Path,
    python_executable: Path,
    sdk_root: Path,
    mode: str,
) -> AgentProcessSpec:
    from .assets import CALIBRATION_MODES

    if mode not in CALIBRATION_MODES:
        raise ValueError(f"unsupported calibration mode: {mode}")
    python_path = (str(source_root), str(engine_root), str(sdk_root))
    return AgentProcessSpec(
        agent_id=f"calibration-{mode}",
        argv=(str(python_executable), str(source_root / "main.py")),
        cwd=source_root,
        env={
            "AGENTBENCH_CALIBRATION_MODE": mode,
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
        },
    )

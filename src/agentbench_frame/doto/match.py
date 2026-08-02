"""Run two native DOTO policies through the historical official protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .assets import official_server_dir, verify_assets
from .process import ManagedProcess
from .protocol import (
    DotoProtocolError,
    read_ai_frame,
    read_server_frame,
    write_ai_observation,
    write_server_action,
)


class DotoMatchError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


@dataclass(frozen=True)
class MatchResult:
    winner: int | None
    scores: tuple[float, float]
    frames: int
    duration: float
    terminated_by: str
    errors: tuple[str, ...]
    replay_path: Path
    trace_path: Path
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["scores"] = list(self.scores)
        value["errors"] = list(self.errors)
        value["replay_path"] = str(self.replay_path)
        value["trace_path"] = str(self.trace_path)
        value["score_diff"] = self.scores[0] - self.scores[1]
        return value


def _argv(path: Path) -> list[str]:
    return [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(raw: Any) -> tuple[float, float]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise DotoMatchError("protocol", "invalid final scores") from exc
    if not isinstance(raw, list) or len(raw) != 2:
        raise DotoMatchError("protocol", "final scores must contain two values")
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise DotoMatchError("protocol", "final scores must be numeric") from exc


def run_match(
    agent0: Path,
    agent1: Path,
    *,
    seed: int,
    output_dir: Path,
    tag: str,
    frame_timeout: float = 1.0,
    server_timeout: float = 330.0,
    server_dir: Path | None = None,
    test_only: bool = False,
) -> MatchResult:
    if frame_timeout <= 0 or server_timeout <= 0:
        raise ValueError("timeouts must be positive")
    agents = (Path(agent0).resolve(), Path(agent1).resolve())
    if not all(path.is_file() for path in agents):
        raise ValueError("both DOTO agent executables must exist")
    output_dir = Path(output_dir).resolve()
    if test_only and "runs" in output_dir.parts and "23_doto" in output_dir.parts:
        raise ValueError("test-only matches cannot be written into 23_doto Results")
    selected_server = Path(server_dir).resolve() if server_dir else official_server_dir().resolve()
    if selected_server != official_server_dir().resolve() and not test_only:
        raise ValueError("a custom server requires test_only=True")
    if not (selected_server / "main.py").is_file():
        raise ValueError("DOTO server directory must contain main.py")
    if not test_only:
        verify_assets()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(char if char.isalnum() or char in "._-" else "-" for char in tag)
    replay_path = output_dir / f"{safe_tag}_seed{seed}.zip"
    trace_path = output_dir / f"{safe_tag}_seed{seed}.trace.jsonl"
    stderr_paths = [output_dir / f"{safe_tag}.server.stderr"] + [
        output_dir / f"{safe_tag}.agent{index}.stderr" for index in range(2)
    ]
    launcher = Path(__file__).with_name("server_launcher.py")
    server = ManagedProcess.start(
        [
            sys.executable,
            str(launcher),
            "--server-dir",
            str(selected_server),
            "--replay",
            str(replay_path),
            "--seed",
            str(seed),
        ],
        cwd=selected_server,
        label="server",
        stderr_path=stderr_paths[0],
    )
    ai_processes = [
        ManagedProcess.start(
            _argv(path), cwd=path.parent, label=f"agent{index}",
            stderr_path=stderr_paths[index + 1],
        )
        for index, path in enumerate(agents)
    ]
    processes = [server, *ai_processes]
    started = time.monotonic()
    seq = 0
    frames = 0
    final_scores: tuple[float, float] | None = None

    def record(stream, kind: str, faction: int | None, frame: int, payload: Any) -> None:
        nonlocal seq
        row = {
            "seq": seq,
            "timestamp": round(time.monotonic() - started, 6),
            "kind": kind,
            "faction": faction,
            "frame": frame,
            "payload": payload,
        }
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        seq += 1

    try:
        with trace_path.open("w", encoding="utf-8") as trace:
            while final_scores is None:
                elapsed = time.monotonic() - started
                if elapsed >= server_timeout:
                    raise DotoMatchError("server_timeout", f"exceeded {server_timeout} seconds")
                try:
                    packet = read_server_frame(
                        server.stdout,
                        min(frame_timeout, server_timeout - elapsed),
                        "server",
                    )
                except DotoProtocolError as exc:
                    if server.process.poll() is not None:
                        raise DotoMatchError(
                            "process_exit", f"server exited with {server.process.returncode}"
                        ) from exc
                    raise DotoMatchError("protocol", str(exc)) from exc
                try:
                    message = json.loads(packet.payload)
                except json.JSONDecodeError as exc:
                    raise DotoMatchError("protocol", f"server emitted invalid JSON: {exc}") from exc
                frame = int(message.get("frame", -2))
                if packet.message_type == 2 or frame == -1:
                    final_scores = _scores(message.get("scores"))
                    record(trace, "final", None, -1, message)
                    for ai in ai_processes:
                        write_ai_observation(ai.stdin, packet.payload)
                    break

                recipients = range(2) if packet.target == -1 else (packet.target,)
                if any(faction not in (0, 1) for faction in recipients):
                    raise DotoMatchError("protocol", f"invalid target {packet.target}")
                for faction in recipients:
                    record(trace, "observation", faction, frame, message)
                    write_ai_observation(ai_processes[faction].stdin, packet.payload)
                if frame == 0:
                    continue
                frames = max(frames, frame)
                for faction in recipients:
                    try:
                        raw_action = read_ai_frame(
                            ai_processes[faction].stdout, frame_timeout, f"agent{faction}"
                        )
                    except DotoProtocolError as exc:
                        stage = "process_exit" if ai_processes[faction].process.poll() is not None else "ai_timeout"
                        raise DotoMatchError(stage, str(exc)) from exc
                    try:
                        action: Any = json.loads(raw_action)
                    except json.JSONDecodeError:
                        action = raw_action.decode("utf-8", errors="replace")
                    record(trace, "action", faction, frame, action)
                    write_server_action(server.stdin, faction, raw_action)

        deadline = time.monotonic() + min(5.0, frame_timeout + 4.0)
        while not replay_path.is_file() and time.monotonic() < deadline:
            if server.process.poll() is not None and not replay_path.is_file():
                break
            time.sleep(0.01)
        if not replay_path.is_file():
            raise DotoMatchError("replay_missing", "server reached final frame without replay ZIP")
        assert final_scores is not None
        winner = 0 if final_scores[0] > final_scores[1] else 1 if final_scores[1] > final_scores[0] else None
        metadata = {
            "seed": seed,
            "test_only": test_only,
            "server_dir": str(selected_server),
            "server_main_sha256": _hash(selected_server / "main.py"),
            "map_sha256": (_hash(selected_server / "Maps" / "0.json")
                           if (selected_server / "Maps" / "0.json").is_file() else None),
            "agent_sha256": [_hash(path) for path in agents],
            "realtime_scale": 1.0,
        }
        return MatchResult(
            winner=winner,
            scores=final_scores,
            frames=frames,
            duration=time.monotonic() - started,
            terminated_by="normal",
            errors=(),
            replay_path=replay_path,
            trace_path=trace_path,
            metadata=metadata,
        )
    finally:
        for process in reversed(processes):
            process.terminate()

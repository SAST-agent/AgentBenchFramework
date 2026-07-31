"""Isolated deterministic matches through the frozen Saiblo protocol."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from agentbench_frame.eval.measurement import canonical_state_id
from agentbench_frame.games.rollman.protocol import (
    ProtocolError,
    decode_ai_frame,
    decode_logic_frame,
    write_logic_input,
)
from agentbench_frame.games.rollman.replay import RollmanReplay, load_replay


SAFE_ENV_NAMES = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SYSTEMROOT",
    "TMPDIR",
)


class MatchError(RuntimeError):
    """The match has no scientifically valid game result."""


@dataclasses.dataclass(frozen=True)
class ProcessSpec:
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    untrusted: bool = False
    memory_limit_mb: int = 512
    read_roots: tuple[Path, ...] = ()
    denied_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(str(value) for value in self.argv))
        if not self.argv or not self.argv[0]:
            raise ValueError("process argv cannot be empty")
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd).resolve())
        object.__setattr__(
            self,
            "read_roots",
            tuple(Path(path).resolve() for path in self.read_roots),
        )
        object.__setattr__(
            self,
            "denied_paths",
            tuple(Path(path).resolve() for path in self.denied_paths),
        )
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")


class DecisionStateTracker(Protocol):
    def reset(self, state: Mapping[str, Any]) -> None:
        ...

    def state(self) -> Mapping[str, Any]:
        ...

    def step(self, rollman_action: int, ghosts_action: tuple[int, int, int]) -> None:
        ...


@dataclasses.dataclass(frozen=True)
class MatchResult:
    status: str
    seed: int
    rollman_score: int
    ghosts_score: int
    result: str
    end_state: tuple[str, str]
    replay: RollmanReplay
    rollman_decisions: tuple[dict[str, Any], ...]
    trace_path: Path


def _safe_environment(extra: Mapping[str, str]) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in SAFE_ENV_NAMES if name in os.environ
    }
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _descendant_rss(root_pid: int) -> dict[int, int] | None:
    try:
        usage = subprocess.run(
            ("ps", "-axo", "pid=,ppid=,rss="),
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except OSError:
        return None
    rows: dict[int, tuple[int, int]] = {}
    for line in usage.stdout.splitlines():
        try:
            pid, ppid, rss = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        rows[pid] = (ppid, rss)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return {pid: rows.get(pid, (0, 0))[1] for pid in descendants}


def _start(spec: ProcessSpec, label: str) -> subprocess.Popen[bytes]:
    argv = spec.argv
    scratch: Path | None = None
    if spec.untrusted:
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
            raise MatchError("untrusted process sandbox is unavailable")
        scratch = Path(tempfile.mkdtemp(prefix="agentbench-untrusted-")).resolve()
        read_roots = {
            Path("/System"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/Library"),
            Path("/opt/homebrew"),
            Path("/dev"),
            *spec.read_roots,
        }
        if spec.cwd is not None:
            read_roots.add(spec.cwd)
        for value in spec.argv:
            candidate = Path(value)
            if not candidate.is_absolute() or not candidate.exists():
                continue
            original = candidate.absolute()
            read_roots.add(
                original.parent.parent
                if original.parent.name == "bin"
                else original.parent
            )
            resolved = candidate.resolve()
            read_roots.add(resolved if resolved.is_dir() else resolved.parent)
            if resolved.parent.name == "bin":
                read_roots.add(resolved.parent.parent)
        clauses = " ".join(
            f"(subpath {json.dumps(str(root))})"
            for root in sorted(read_roots, key=lambda item: str(item))
            if root.exists()
        )
        denied = "".join(
            f"(deny file-read* (literal {json.dumps(str(path))}))"
            for path in spec.denied_paths
        )
        profile = (
            '(version 1)(import "system.sb")(deny default)'
            "(allow process-exec)(deny process-fork)"
            "(allow signal (target self))"
            "(allow sysctl-read)(allow mach-lookup)(allow ipc-posix*)"
            f"(allow file-read* {clauses})"
            f"(allow file-write* (subpath {json.dumps(str(scratch))}))"
            f"{denied}"
        )
        argv = ("/usr/bin/sandbox-exec", "-p", profile, *spec.argv)
    try:
        child_environment = _safe_environment(spec.env)
        if scratch is not None:
            child_environment["TMPDIR"] = str(scratch)
        process = subprocess.Popen(
            argv,
            cwd=spec.cwd,
            env=child_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise MatchError(f"failed to start {label}: {exc}") from exc
    if process.stdin is None or process.stdout is None:
        raise MatchError(f"failed to open pipes for {label}")
    if scratch is not None:
        setattr(process, "_agentbench_scratch", scratch)
    if spec.untrusted:
        limit_kib = spec.memory_limit_mb * 1024

        def monitor_memory() -> None:
            while process.poll() is None:
                descendants = _descendant_rss(process.pid)
                if descendants is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    return
                rss_kib = sum(descendants.values())
                if rss_kib > limit_kib or len(descendants) > 64:
                    for pid in sorted(descendants, reverse=True):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                    return
                time.sleep(0.05)

        threading.Thread(target=monitor_memory, daemon=True).start()
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    running = process.poll() is None
    descendants = (
        _descendant_rss(process.pid) or {process.pid: 0}
        if running
        else {}
    )
    for pid in sorted(descendants, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait()
    for pid in sorted(descendants, reverse=True):
        if pid == process.pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    scratch = getattr(process, "_agentbench_scratch", None)
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_raw(stream: Any, content: str) -> None:
    stream.write(content.encode("utf-8"))
    stream.flush()


def _parse_action(payload: bytes, *, player: int) -> tuple[dict[str, Any], Any]:
    try:
        message = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MatchError(f"player {player} emitted invalid JSON") from exc
    if not isinstance(message, dict):
        raise MatchError(f"player {player} action must be an object")
    role = message.get("role")
    expected_role = player
    if role != expected_role:
        raise MatchError(
            f"player {player} declared role {role!r}, expected {expected_role}"
        )
    raw_action = message.get("action")
    if not isinstance(raw_action, str):
        raise MatchError(f"player {player} action must be a string")
    parts = raw_action.split()
    expected_count = 1 if role == 0 else 3
    if len(parts) != expected_count:
        raise MatchError(
            f"player {player} action requires {expected_count} directions"
        )
    try:
        actions = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise MatchError(f"player {player} action contains a non-integer") from exc
    if any(action not in range(5) for action in actions):
        raise MatchError(f"player {player} action is outside 0..4")
    return message, actions[0] if role == 0 else actions


def _judger_ai_error(exc: ProtocolError) -> tuple[int, str]:
    message = str(exc).lower()
    if "timed out" in message:
        return 1, "TLE"
    if "unreasonable frame length" in message:
        return 2, "OLE"
    return 0, "RE"


def _visible_state_from_watch(value: Mapping[str, Any]) -> dict[str, Any] | None:
    required = {
        "level",
        "round",
        "pacman_coord",
        "ghosts_coord",
        "score",
        "portal_available",
    }
    if not required.issubset(value) or value.get("StopReason") is not None:
        return None
    return dict(value)


def run_match(
    *,
    logic: ProcessSpec,
    rollman: ProcessSpec,
    ghosts: ProcessSpec,
    seed: int,
    timeout_s: float,
    replay_path: str | Path,
    trace_path: str | Path,
    state_tracker: DecisionStateTracker | None = None,
) -> MatchResult:
    """Run one role-fixed game: player 0 is Rollman and player 1 is Ghosts."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    replay_file = Path(replay_path).resolve()
    trace_file = Path(trace_path).resolve()
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    seeded_logic = ProcessSpec(
        argv=logic.argv,
        cwd=logic.cwd,
        env={**logic.env, "AGENTBENCH_ROLLMAN_SEED": str(int(seed))},
    )
    logic_process = _start(seeded_logic, "logic")
    players = [_start(rollman, "Rollman"), _start(ghosts, "Ghosts")]
    processes = [logic_process, *players]
    trace: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    latest_visible_state: dict[str, Any] | None = None
    pending_actions: dict[int, Any] = {}
    announced_ai_timeout = float(timeout_s)
    announced_ai_max_length = 1024
    try:
        init = json.dumps(
            {
                "player_list": [1, 1],
                "player_num": 2,
                "replay": str(replay_file),
                "config": {"random_seed": int(seed)},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        write_logic_input(logic_process.stdin, init)

        while True:
            target, payload = decode_logic_frame(
                logic_process.stdout,
                timeout=timeout_s,
                label="logic",
            )
            if target in {0, 1}:
                content = payload.decode("utf-8")
                _write_raw(players[target].stdin, content)
                trace.append(
                    {
                        "type": "observation",
                        "state": None,
                        "player": target,
                        "content": content.rstrip("\n"),
                    }
                )
                try:
                    direct_value = json.loads(content)
                except json.JSONDecodeError:
                    direct_value = None
                if isinstance(direct_value, dict) and "board" in direct_value:
                    latest_visible_state = dict(direct_value)
                    if state_tracker is not None:
                        state_tracker.reset(direct_value)
                continue
            try:
                message = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MatchError("logic emitted invalid JSON") from exc
            if not isinstance(message, dict):
                raise MatchError("logic message must be an object")

            if "watch" in message:
                try:
                    watch = json.loads(str(message["watch"]))
                except json.JSONDecodeError as exc:
                    raise MatchError("logic emitted invalid watch JSON") from exc
                visible = _visible_state_from_watch(watch)
                if visible is not None:
                    latest_visible_state = visible
                trace.append({"type": "watch", "content": watch})
                continue

            state = int(message.get("state", 0))
            if state == -1:
                raw_end_info = message.get("end_info", "{}")
                raw_end_state = message.get("end_state", '["RE","RE"]')
                end_info = (
                    json.loads(raw_end_info)
                    if isinstance(raw_end_info, str)
                    else raw_end_info
                )
                end_state = (
                    json.loads(raw_end_state)
                    if isinstance(raw_end_state, str)
                    else raw_end_state
                )
                if not isinstance(end_info, dict):
                    raise MatchError("logic end_info must be an object")
                if not isinstance(end_state, list) or len(end_state) != 2:
                    raise MatchError("logic end_state must contain two player states")
                allowed_end_states = {"OK", "RE", "TLE", "OLE", "IA"}
                if any(str(value) not in allowed_end_states for value in end_state):
                    raise MatchError(
                        f"logic emitted unknown player end state: {end_state}"
                    )
                scores = (int(end_info["0"]), int(end_info["1"]))
                replay = load_replay(replay_file)
                if end_state == ["OK", "OK"] and replay.final_score != scores:
                    raise MatchError(
                        "logic end_info score disagrees with replay terminal score"
                    )
                result = (
                    "win"
                    if scores[0] > scores[1]
                    else "loss"
                    if scores[0] < scores[1]
                    else "draw"
                )
                trace_file.write_text(
                    "".join(
                        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                        for item in trace
                    ),
                    encoding="utf-8",
                )
                return MatchResult(
                    status="complete",
                    seed=int(seed),
                    rollman_score=scores[0],
                    ghosts_score=scores[1],
                    result=result,
                    end_state=(str(end_state[0]), str(end_state[1])),
                    replay=replay,
                    rollman_decisions=tuple(decisions),
                    trace_path=trace_file,
                )
            if state == 0 and "time" in message:
                announced_ai_timeout = float(message["time"])
                announced_ai_max_length = int(message.get("length", 1024))
                if announced_ai_timeout <= 0 or announced_ai_max_length <= 0:
                    raise MatchError("logic emitted invalid round limits")
                trace.append(
                    {
                        "type": "round_config",
                        "time": announced_ai_timeout,
                        "length": announced_ai_max_length,
                    }
                )
                continue

            send_players = message.get("player", [])
            contents = message.get("content", [])
            if len(send_players) != len(contents):
                raise MatchError("logic player/content lengths differ")
            for player_value, content_value in zip(send_players, contents):
                player = int(player_value)
                content = str(content_value)
                _write_raw(players[player].stdin, content)
                trace.append(
                    {
                        "type": "observation",
                        "state": state,
                        "player": player,
                        "content": content.rstrip("\n"),
                    }
                )
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and "board" in parsed:
                    latest_visible_state = dict(parsed)
                    if state_tracker is not None:
                        state_tracker.reset(parsed)

            listeners = [int(value) for value in message.get("listen", [])]
            responses: dict[int, bytes] = {}
            ai_fault = False
            for player in listeners:
                try:
                    response = decode_ai_frame(
                        players[player].stdout,
                        timeout=announced_ai_timeout,
                        label=f"player {player}",
                        max_frame_size=announced_ai_max_length,
                    )
                except ProtocolError as exc:
                    error_code, error_name = _judger_ai_error(exc)
                    trace.append(
                        {
                            "type": "ai_fault",
                            "state": state,
                            "player": player,
                            "error": error_name,
                            "detail": str(exc),
                        }
                    )
                    routed_error = json.dumps(
                        {
                            "player": -1,
                            "content": json.dumps(
                                {"error": error_code, "player": player},
                                separators=(",", ":"),
                            ),
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                    write_logic_input(logic_process.stdin, routed_error)
                    pending_actions.clear()
                    ai_fault = True
                    break
                action_message, action = _parse_action(response, player=player)
                responses[player] = response
                pending_actions[player] = action
                action_event: dict[str, Any] = {
                    "type": "action",
                    "state": state,
                    "player": player,
                    "role": player,
                    "action": action,
                }
                if player == 0:
                    context = (
                        dict(state_tracker.state())
                        if state_tracker is not None
                        else dict(latest_visible_state or {})
                    )
                    if not context:
                        raise MatchError("Rollman action has no pre-decision state")
                    decision = {
                        "state": context,
                        "state_id": canonical_state_id(context),
                        "action": int(action),
                        "memory_id": action_message.get("memory_id"),
                    }
                    decisions.append(decision)
                    action_event["decision"] = decision
                trace.append(action_event)

            if ai_fault:
                continue
            if 0 in pending_actions and 1 in pending_actions:
                if state_tracker is not None:
                    state_tracker.step(pending_actions[0], pending_actions[1])
                pending_actions.clear()

            if len(listeners) != 1:
                continue
            player = listeners[0]
            routed = json.dumps(
                {
                    "player": player,
                    "content": responses[player].decode("utf-8"),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            write_logic_input(logic_process.stdin, routed)
    except (ProtocolError, OSError, KeyError, ValueError) as exc:
        raise MatchError(str(exc)) from exc
    finally:
        for process in processes:
            _stop(process)

"""Isolated deterministic matches through the frozen Saiblo protocol."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(str(value) for value in self.argv))
        if not self.argv or not self.argv[0]:
            raise ValueError("process argv cannot be empty")
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd).resolve())


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
    environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _start(spec: ProcessSpec, label: str) -> subprocess.Popen[bytes]:
    try:
        process = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=_safe_environment(spec.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise MatchError(f"failed to start {label}: {exc}") from exc
    if process.stdin is None or process.stdout is None:
        raise MatchError(f"failed to open pipes for {label}")
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


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
                scores = (int(end_info["0"]), int(end_info["1"]))
                replay = load_replay(replay_file)
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
            for player in listeners:
                response = decode_ai_frame(
                    players[player].stdout,
                    timeout=timeout_s,
                    label=f"player {player}",
                )
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

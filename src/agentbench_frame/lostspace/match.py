"""Run one LostSpace game (4-player FFA) via the historical Saiblo wire protocol.

The match runner spawns the official game logic plus four AI client subprocesses
and shuttles length-prefixed frames between them, acting as the judger. It is a
4-player generalisation of ``agentbench_frame.aquawar.match``; the wire framing
and routing logic are identical, only the player count and end-of-game score
parsing differ.

Process management is cross-platform: on POSIX children are started in a new
session and stopped with the process group; on Windows a new process group is
created and the tree is torn down with ``taskkill /T``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentbench_frame.lostspace.protocol import read_frame, write_frame

_IS_WINDOWS = sys.platform.startswith("win")

# Content-frame types that oblige the addressed player to reply with its next
# action. LostSpace's logic (communicate.py) emits these then re-enters its
# ``inround`` read loop waiting for exactly one action frame back; every other
# type (``id``, ``offround``, ``other_death``, ``getkey``, ``escaped``,
# ``death``, ``ai_error``, witness ``see``, ...) is a notification the player
# consumes silently. ``roundbegin`` starts a turn, ``action`` is the per-action
# response, and ``format error`` is a re-prompt after a malformed action.
_ACTION_REQUEST_TYPES = frozenset({"roundbegin", "action", "format error"})


class LostSpaceMatchError(RuntimeError):
    """Raised when a LostSpace match cannot reach a valid result."""


def _start(command: str, label: str) -> subprocess.Popen[bytes]:
    kwargs: dict[str, Any] = dict(
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if _IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    if process.stdin is None or process.stdout is None:
        raise LostSpaceMatchError(f"failed to open pipes for {label}")
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (FileNotFoundError, OSError):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        if _IS_WINDOWS:
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def _decode_content(content: str) -> Any:
    """Parse a per-player content string for the trace.

    LostSpace wraps AI-facing payloads with ``convert_byte_str_for_ai``
    (4 ASCII digits + json). The original bytes (prefix included) are still
    forwarded to the AI verbatim; this helper only strips the prefix so the
    trace stores clean JSON.
    """
    if len(content) >= 4 and content[:4].isdigit():
        return json.loads(content[4:])
    return json.loads(content)


def run_match(
    logic_command: str,
    ai_commands: list[str],
    timeout: float,
    replay_path: Path,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Run a complete 4-player game.

    Returns a dict with:
      - ``winner``: player id with the top score, or ``None`` on a tie.
      - ``ranking``: player ids ordered best (1st) to worst (4th).
      - ``end_info``: the raw score dict ``{"0": pts, ...}`` (4=1st ... 1=4th).
      - ``turns``: number of action frames routed back to the logic.
    """
    if len(ai_commands) != 4:
        raise ValueError("LostSpace requires exactly four AI commands")

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    logic = _start(logic_command, "logic")
    ais = [
        _start(command, f"player {index}")
        for index, command in enumerate(ai_commands)
    ]
    processes = [logic, *ais]
    turns = 0
    trace: list[dict[str, Any]] = []
    try:
        init = json.dumps(
            {
                "player_list": [1, 1, 1, 1],
                "player_num": 4,
                "replay": str(replay_path.resolve()),
            },
            separators=(",", ":"),
        ).encode()
        write_frame(logic.stdin, init)

        while True:
            packet = read_frame(
                logic.stdout,
                timeout,
                "logic",
                has_target=True,
            )
            try:
                message = json.loads(packet)
            except json.JSONDecodeError as exc:
                raise LostSpaceMatchError(
                    f"logic emitted invalid JSON: {exc}"
                ) from exc

            state = int(message.get("state", 0))
            if state == -1:
                raw_end = message.get("end_info", "{}")
                end_info = (
                    json.loads(raw_end) if isinstance(raw_end, str) else raw_end
                )
                scores = {int(k): float(v) for k, v in end_info.items()}
                ranking = sorted(scores, key=lambda p: (-scores[p], p))
                top_two = ranking[:2]
                if (
                    len(top_two) == 2
                    and scores[top_two[0]] == scores[top_two[1]]
                ):
                    winner = None
                else:
                    winner = ranking[0] if ranking else None
                return {
                    "winner": winner,
                    "ranking": ranking,
                    "end_info": end_info,
                    "turns": turns,
                }
            if state == 0 and "time" in message:
                continue

            players = message.get("player", [])
            contents = message.get("content", [])
            listeners = message.get("listen", [])
            content_by_player: dict[int, str] = {
                int(p): c for p, c in zip(players, contents)
            }
            for player_value, content in zip(players, contents):
                player = int(player_value)
                parsed_content = _decode_content(content)
                trace.append(
                    {
                        "state": state,
                        "type": "observation",
                        "player": player,
                        "content": parsed_content,
                    }
                )
                ais[player].stdin.write(content.encode())
                ais[player].stdin.flush()

            # A listener actually replies only when the frame is an action
            # request addressed to it. LostSpace carries the in-turn player in
            # `listen` on nearly every frame (including death/escape/off-round
            # notifications), but the in-turn player only emits its next action
            # in response to: round-begin (turn start), an "action" response
            # (respond_action after each action), or a re-prompt after a format
            # error. Everything else (other_death, escaped, ai_error, offround,
            # witness "see", id) is a notification the player consumes silently.
            for listener in listeners:
                responder = int(listener)
                if responder not in content_by_player:
                    continue
                ctype = _decode_content(content_by_player[responder]).get("type")
                if ctype not in _ACTION_REQUEST_TYPES:
                    continue
                response = read_frame(
                    ais[responder].stdout,
                    timeout,
                    f"player {responder}",
                )
                try:
                    action: Any = json.loads(response)
                except json.JSONDecodeError:
                    action = response.decode("utf-8", errors="replace")
                trace.append(
                    {
                        "state": state,
                        "type": "action",
                        "player": responder,
                        "content": action,
                    }
                )
                routed = json.dumps(
                    {
                        "player": responder,
                        "content": response.decode("utf-8", errors="replace"),
                    },
                    separators=(",", ":"),
                ).encode()
                write_frame(logic.stdin, routed)
                turns += 1
    except LostSpaceMatchError:
        raise
    except Exception as exc:
        exited = [
            f"process {index} exit={process.returncode}"
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        suffix = f" ({'; '.join(exited)})" if exited else ""
        raise LostSpaceMatchError(f"{exc}{suffix}") from exc
    finally:
        for process in processes:
            _stop(process)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(
                json.dumps(item, ensure_ascii=False) for item in trace
            )
            trace_path.write_text(content + ("\n" if content else ""))

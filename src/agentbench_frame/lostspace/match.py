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
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentbench_frame.lostspace.protocol import (
    LostSpaceProtocolError,
    read_frame,
    write_frame,
)

_IS_WINDOWS = sys.platform.startswith("win")

# Content-frame types that oblige the addressed player to reply with its next
# action. LostSpace's logic (communicate.py) emits these then re-enters its
# ``inround`` read loop waiting for exactly one action frame back; every other
# type (``id``, ``offround``, ``other_death``, ``getkey``, ``escaped``,
# ``death``, ``ai_error``, witness ``see``, ...) is a notification the player
# consumes silently. ``roundbegin`` starts a turn, ``action`` is the per-action
# response, and ``format error`` is a re-prompt after a malformed action.
_ACTION_REQUEST_TYPES = frozenset({"roundbegin", "action", "format error"})

# LostSpace PlayerStatus values: 0=Alive, 1=Died, 2=Escaped, 3=Skipped,
# 4=WaitForEscape, 5=Error. The logic only opens its in-round action loop
# (``inround()``, which actually waits for the player's move) when the
# in-turn player is Alive or WaitForEscape. For every other status the
# logic emits the ``roundbegin`` with ``listen=[player]`` but never blocks —
# so the harness must not block either, or it will burn a full TLE every
# round for each dead/errored/escaped player.
_ACTION_STATUSES = frozenset({0, 4})


def _expects_reply(content: str) -> bool:
    """Whether the in-turn player must emit an action for this frame."""
    parsed = _decode_content(content)
    if not isinstance(parsed, dict):
        return False
    ctype = parsed.get("type")
    if ctype not in _ACTION_REQUEST_TYPES:
        return False
    if ctype == "roundbegin":
        # No status field ⇒ assume actionable (first-turn / unknown form).
        status = parsed.get("status")
        if status is not None and status not in _ACTION_STATUSES:
            return False
    return True


def _report_ai_error(
    logic: subprocess.Popen[bytes],
    trace: list[dict[str, Any]],
    player: int,
    state: int,
    error_log: str,
) -> None:
    """Tell the logic a player has failed (TLE or crash) and the game goes on.

    Mirrors the saiblo judger's ``ai_error`` message: the logic's in-round loop
    treats it as a fatal error for that player (hp=0, marked lose) and, if it
    is the in-turn player, ends the turn. Idempotent for players already out.
    """
    error_code = 0 if error_log == "runError" else 1
    inner = json.dumps(
        {
            "player": player,
            "state": state,
            "error": error_code,
            "error_log": error_log,
        },
        separators=(",", ":"),
    )
    routed = json.dumps(
        {"player": -1, "content": inner},
        separators=(",", ":"),
    ).encode()
    write_frame(logic.stdin, routed)
    trace.append(
        {
            "state": state,
            "type": "ai_error",
            "player": player,
            "content": error_log,
        }
    )


class LostSpaceMatchError(RuntimeError):
    """Raised when a LostSpace match cannot reach a valid result."""


def _to_argv(rest: str) -> list[str]:
    """Split a (post-``cd``) command string into an argv list, Windows-safe.

    ``shlex.split(posix=False)`` preserves backslashes in Windows paths but
    leaves the surrounding double-quotes that ``subprocess.list2cmdline``
    emits around tokens containing spaces; we strip one pair per token.
    """
    argv: list[str] = []
    for token in shlex.split(rest, posix=False):
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            token = token[1:-1]
        argv.append(token)
    return argv


def _split_cwd(command: str) -> tuple[str, str | None]:
    """Pull a leading ``cd <dir> &&`` (or ``cd /d <dir> &&``) off ``command``.

    Returns ``(rest, cwd)``. We strip the ``cd`` chain because leaving it in
    the ``shell=True`` command on Windows (``cmd /c "cd /d X && prog"``)
    mediates the stdin pipe through cmd.exe, which intermittently rejects
    large writes with ``OSError [Errno 22] Invalid argument`` once the
    per-turn ``roundbegin`` payload (a few hundred bytes) is flushed before
    the child has drained it. Passing ``cwd`` to ``Popen`` instead launches
    the program directly and keeps the pipe unmediated.
    """
    rest = command.strip()
    cwd: str | None = None
    if rest.startswith("cd "):
        # posix=False keeps backslashes in Windows paths (D:\pymol\python.exe);
        # we strip the surrounding quotes that list2cmdline added.
        tokens = _to_argv(rest)
        idx = 1
        if idx < len(tokens) and tokens[idx] == "/d":  # Windows flag
            idx += 1
        if idx < len(tokens):
            cwd = tokens[idx]
            idx += 1
        if idx < len(tokens) and tokens[idx] == "&&":
            idx += 1
        rest = " ".join(tokens[idx:])
    return rest, cwd


def _resolve_windows_executable(argv: list[str], cwd: str | None) -> None:
    """Resolve a bare program name (e.g. ``main``) under Windows.

    ``cmd /c`` would use PATHEXT to turn ``main`` into ``main.exe`` in the
    current directory; a direct ``CreateProcess`` call does not, so a
    cwd-local C++ binary built by ``make`` (``main``) would not be found.
    Append ``.exe`` when the bare name resolves in ``cwd`` (or on PATH).
    """
    if not argv:
        return
    exe = argv[0]
    if os.path.sep in exe or "/" in exe:
        return  # already a path
    # A bare program name (``main`` or ``main.exe``) launched with an explicit
    # application name is resolved by CreateProcess against the *calling*
    # process cwd, NOT the ``cwd=`` we pass — so it must be made absolute
    # against the intended working directory first.
    candidates = [exe, f"{exe}.exe"] if "." not in os.path.basename(exe) else [exe]
    search_dirs: list[str] = []
    if cwd:
        search_dirs.append(cwd)
    search_dirs += os.environ.get("PATH", "").split(os.pathsep)
    for directory in search_dirs:
        if not directory:
            continue
        for candidate in candidates:
            if os.path.isfile(os.path.join(directory, candidate)):
                argv[0] = os.path.join(directory, candidate)
                return


def _stderr_for(label: str) -> Any:
    """Return a stderr target for a child process.

    When ``LOSTSPACE_DEBUG_DIR`` is set, each child's stderr is written to a
    file named after its label so a stall/death can be post-mortemed.
    """
    debug_dir = os.environ.get("LOSTSPACE_DEBUG_DIR")
    if not debug_dir:
        return subprocess.DEVNULL
    path = Path(debug_dir) / f"{label}.stderr"
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "ab", buffering=0)


def _child_env() -> dict[str, str]:
    """Environment for child processes (logic + AIs).

    ``PYTHONHOME`` / ``PYTHONPATH`` are stripped so a Python child launched
    with its own interpreter (e.g. the bundled logic under
    ``D:\\pymol\\python.exe``) is not poisoned by the parent's interpreter —
    a real failure mode when the harness itself runs under ``uv`` (whose
    ``PYTHONHOME`` points at a different CPython and breaks the child's
    ``importlib``).
    """
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH"}}
    return env


def _start(command: str, label: str) -> subprocess.Popen[bytes]:
    shell_command, cwd = _split_cwd(command)
    env = _child_env()
    if _IS_WINDOWS:
        # Launch the program directly with an argv + cwd instead of routing
        # through ``cmd /c``. cmd.exe as a pipe middleman intermittently
        # rejects large stdin writes (Errno 22) on the per-turn ``roundbegin``
        # frame; going argv-first removes the race and keeps the pipe direct.
        argv = _to_argv(shell_command)
        _resolve_windows_executable(argv, cwd)
        kwargs: dict[str, Any] = dict(
            args=argv,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=_stderr_for(label),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=env,
        )
        if cwd:
            kwargs["cwd"] = cwd
    else:
        kwargs = dict(
            args=shell_command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=_stderr_for(label),
            start_new_session=True,
            env=env,
        )
        if cwd:
            kwargs["cwd"] = cwd
    process = subprocess.Popen(**kwargs)
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

        pending_responder: int | None = None
        while True:
            try:
                packet = read_frame(
                    logic.stdout,
                    timeout,
                    "logic",
                    has_target=True,
                )
            except LostSpaceProtocolError as exc:
                # Symmetric-deadlock break: the logic emits a frame after every
                # action it accepts, so a logic-read stall means the logic is
                # itself blocked waiting for the player who just acted (a
                # multi-action turn where the AI went quiet without
                # ``finish``). Attribute the timeout to that player and let the
                # logic's ai_error handling advance the game, exactly as the
                # saiblo TLE would.
                if pending_responder is None:
                    raise
                error_log = (
                    "runError" if "exited" in str(exc) else "timeOutError"
                )
                error_code = 0 if error_log == "runError" else 1
                inner = json.dumps(
                    {
                        "player": pending_responder,
                        "state": state,
                        "error": error_code,
                        "error_log": error_log,
                    },
                    separators=(",", ":"),
                )
                routed = json.dumps(
                    {"player": -1, "content": inner},
                    separators=(",", ":"),
                ).encode()
                write_frame(logic.stdin, routed)
                trace.append(
                    {
                        "state": state,
                        "type": "ai_error",
                        "player": pending_responder,
                        "content": error_log,
                    }
                )
                turns += 1
                pending_responder = None
                continue
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
                try:
                    ais[player].stdin.write(content.encode())
                    ais[player].stdin.flush()
                except OSError:
                    # The AI process has exited (broken pipe). Report it as a
                    # runError so the logic eliminates the player and the match
                    # continues, instead of crashing the whole game.
                    _report_ai_error(logic, trace, player, state, "runError")

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
                if not _expects_reply(content_by_player[responder]):
                    continue
                try:
                    response = read_frame(
                        ais[responder].stdout,
                        timeout,
                        f"player {responder}",
                    )
                except LostSpaceProtocolError as exc:
                    # Saiblo enforces a per-round time limit: an AI that does
                    # not answer an action request in time is reported to the
                    # logic as an ``ai_error`` (TimeOutError / RunError) and the
                    # game continues — the logic marks the player errored and
                    # moves on. Several ranked algorithms legitimately stall in
                    # specific branches (e.g. rank01 returns from its get-key
                    # strategy without sending ``finish``); replicating the TLE
                    # instead of deadlocking the whole match is the faithful
                    # behaviour. A crashed/closed pipe is a RunError; a stall
                    # is a TimeOutError.
                    error_log = "runError" if "exited" in str(exc) else "timeOutError"
                    _report_ai_error(logic, trace, responder, state, error_log)
                    turns += 1
                    continue
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
                # This player just acted; if the logic subsequently stalls
                # (next read), the stall is attributed to them.
                pending_responder = responder
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

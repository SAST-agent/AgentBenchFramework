"""Run one AquaWar game using the historical Saiblo wire protocol."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from agentbench_frame.aquawar.protocol import read_frame, write_frame


class AquaWarMatchError(RuntimeError):
    """Raised when an AquaWar match cannot reach a valid result."""


def _start(command: str, label: str) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if process.stdin is None or process.stdout is None:
        raise AquaWarMatchError(f"failed to open pipes for {label}")
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


def run_match(
    logic_command: str,
    ai_commands: list[str],
    timeout: float,
    replay_path: Path,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Run a complete game and return its winner, score, and turn count."""
    if len(ai_commands) != 2:
        raise ValueError("AquaWar requires exactly two AI commands")

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
                "player_list": [1, 1],
                "player_num": 2,
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
                raise AquaWarMatchError(
                    f"logic emitted invalid JSON: {exc}"
                ) from exc

            state = int(message.get("state", 0))
            if state == -1:
                raw_end = message.get("end_info", "{}")
                end_info = json.loads(raw_end) if isinstance(raw_end, str) else raw_end
                winner = None
                if end_info.get("0", 0) > end_info.get("1", 0):
                    winner = 0
                elif end_info.get("1", 0) > end_info.get("0", 0):
                    winner = 1
                return {"winner": winner, "end_info": end_info, "turns": turns}
            if state == 0 and "time" in message:
                continue

            players = message.get("player", [])
            contents = message.get("content", [])
            listeners = message.get("listen", [])
            for player_value, content in zip(players, contents):
                player = int(player_value)
                parsed_content = json.loads(content)
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

            responses: dict[int, bytes] = {}
            for listener in listeners:
                player = int(listener)
                response = read_frame(
                    ais[player].stdout,
                    timeout,
                    f"player {player}",
                )
                responses[player] = response
                try:
                    action: Any = json.loads(response)
                except json.JSONDecodeError:
                    action = response.decode("utf-8", errors="replace")
                trace.append(
                    {
                        "state": state,
                        "type": "action",
                        "player": player,
                        "content": action,
                    }
                )

            # Multiple listeners indicate a finish broadcast. The historical
            # logic enters READY without reading their acknowledgements.
            if len(listeners) != 1:
                continue
            player = int(listeners[0])
            routed = json.dumps(
                {
                    "player": player,
                    "content": responses[player].decode("utf-8", errors="replace"),
                },
                separators=(",", ":"),
            ).encode()
            write_frame(logic.stdin, routed)
            turns += 1
    except AquaWarMatchError:
        raise
    except Exception as exc:
        exited = [
            f"process {index} exit={process.returncode}"
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        suffix = f" ({'; '.join(exited)})" if exited else ""
        raise AquaWarMatchError(f"{exc}{suffix}") from exc
    finally:
        for process in processes:
            _stop(process)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(
                json.dumps(item, ensure_ascii=False) for item in trace
            )
            trace_path.write_text(content + ("\n" if content else ""))

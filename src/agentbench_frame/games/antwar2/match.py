"""Native AntWar2 transport and role-correct generic match records."""

from __future__ import annotations

import dataclasses
import json
import os
import queue
import struct
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from agentbench_frame.hl.match_record import MatchRecord


class AntWarMatchError(RuntimeError):
    """The native run did not produce a scientifically valid match."""


@dataclasses.dataclass(frozen=True)
class ProcessSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv or not self.argv[0]:
            raise ValueError("process argv cannot be empty")
        object.__setattr__(self, "argv", tuple(str(item) for item in self.argv))
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())


@dataclasses.dataclass(frozen=True)
class NativeMatchArtifacts:
    record: MatchRecord
    replay_path: Path
    trace_path: Path
    events_path: Path
    game_returncode: int
    player_returncodes: tuple[int, int]


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF after {count - remaining}/{count} bytes")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _packet(value: object) -> bytes:
    body = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def _write(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(stream.fileno(), view)
        if written < 1:
            raise BrokenPipeError("process pipe accepted zero bytes")
        view = view[written:]


def _reader(
    stream: BinaryIO,
    output: queue.Queue[tuple[str, Any]],
    *,
    game: bool,
) -> None:
    try:
        while True:
            size = struct.unpack(">I", _read_exact(stream, 4))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError(f"frame exceeds 64 MiB: {size}")
            object_id = struct.unpack(">i", _read_exact(stream, 4))[0] if game else None
            body = _read_exact(stream, size)
            output.put(("frame", (object_id, body)))
    except EOFError:
        output.put(("eof", None))
    except BaseException as exc:  # delivered to the orchestration thread
        output.put(("error", exc))


def _drain_stderr(stream: BinaryIO, destination: Path, tail: bytearray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            try:
                chunk = stream.read(65536)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            handle.write(chunk)
            handle.flush()
            tail.extend(chunk)
            if len(tail) > 8192:
                del tail[:-8192]


def _safe_environment(extra: Mapping[str, str]) -> dict[str, str]:
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")
    value = {name: os.environ[name] for name in allowed if name in os.environ}
    value.update({str(key): str(item) for key, item in extra.items()})
    value["PYTHONUNBUFFERED"] = "1"
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    return value


def _start(spec: ProcessSpec) -> subprocess.Popen[bytes]:
    try:
        process = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=_safe_environment(spec.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise AntWarMatchError(f"failed to start {spec.argv[0]}: {exc}") from exc
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.terminate()
        raise AntWarMatchError("failed to open process pipes")
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _next_frame(
    frames: queue.Queue[tuple[str, Any]],
    *,
    deadline: float,
    label: str,
    processes: tuple[subprocess.Popen[bytes], ...],
) -> tuple[int | None, bytes]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AntWarMatchError(f"{label} timed out")
    try:
        kind, payload = frames.get(timeout=remaining)
    except queue.Empty as exc:
        codes = [process.poll() for process in processes]
        raise AntWarMatchError(f"{label} timed out; returncodes={codes}") from exc
    if kind == "error":
        raise AntWarMatchError(f"{label} reader failed: {payload}") from payload
    if kind == "eof":
        codes = [process.poll() for process in processes]
        raise AntWarMatchError(f"{label} closed early; returncodes={codes}")
    return payload


def _load_replay(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AntWarMatchError(f"cannot read replay {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise AntWarMatchError("replay must be a non-empty JSON array")
    if not all(isinstance(record, dict) for record in value):
        raise AntWarMatchError("replay contains a non-object round")
    return value


def record_from_replay(
    replay_path: str | Path,
    *,
    candidate: str,
    opponent: str,
    candidate_role: str,
    seed: int,
    trace_path: str | Path | None = None,
) -> MatchRecord:
    """Convert one complete official replay into the common match schema."""

    if candidate_role not in {"P0", "P1"}:
        raise ValueError("candidate_role must be P0 or P1")
    source = Path(replay_path).resolve()
    replay = _load_replay(source)
    terminal = replay[-1].get("round_state")
    if not isinstance(terminal, dict):
        raise AntWarMatchError("replay has no terminal round_state")
    winner = terminal.get("winner")
    if winner not in {0, 1}:
        raise AntWarMatchError("replay has no valid terminal winner")
    camps = terminal.get("camps")
    if (
        not isinstance(camps, list)
        or len(camps) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in camps)
    ):
        raise AntWarMatchError("replay has invalid terminal camps")
    role = 0 if candidate_role == "P0" else 1
    other = 1 - role
    won = winner == role
    candidate_camp = float(camps[role])
    opponent_camp = float(camps[other])
    return MatchRecord.from_mapping(
        {
            "schema_version": "1.0",
            "game": "30_antwar2",
            "candidate": candidate,
            "opponent": opponent,
            "candidate_role": candidate_role,
            "seed": int(seed),
            "status": "complete",
            "result": "win" if won else "loss",
            "points": 1.0 if won else 0.0,
            "candidate_score": candidate_camp,
            "opponent_score": opponent_camp,
            "dense_margin": candidate_camp - opponent_camp,
            "terminal_metrics": {
                "p0_camp_hp": float(camps[0]),
                "p1_camp_hp": float(camps[1]),
            },
            "rounds": len(replay),
            "replay": str(source),
            "trace": None if trace_path is None else str(Path(trace_path).resolve()),
            "faults": [],
            "live_opponent": True,
        }
    )


def write_public_trace(replay_path: str | Path, trace_path: str | Path) -> None:
    """Project the official replay into bounded-skill-compatible public JSONL."""

    replay = _load_replay(Path(replay_path))
    destination = Path(trace_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        sequence = 0
        for round_index, record in enumerate(replay):
            for player in (0, 1):
                operations = record.get(f"op{player}", [])
                if not isinstance(operations, list):
                    raise AntWarMatchError(
                        f"round {round_index} player {player} operations are invalid"
                    )
                item = {
                    "sequence": sequence,
                    "round": round_index,
                    "kind": "ai_operations",
                    "player": player,
                    "operations": operations,
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                sequence += 1
            public_state = record.get("round_state")
            if not isinstance(public_state, dict):
                raise AntWarMatchError(f"round {round_index} has no public state")
            item = {
                "sequence": sequence,
                "round": round_index,
                "kind": "public_state",
                "public_state": public_state,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            sequence += 1


def run_native_match(
    *,
    game: ProcessSpec,
    candidate_process: ProcessSpec,
    opponent_process: ProcessSpec,
    candidate: str,
    opponent: str,
    candidate_role: str,
    seed: int,
    replay_path: str | Path,
    trace_path: str | Path,
    events_path: str | Path,
    timeout_s: float = 120.0,
) -> NativeMatchArtifacts:
    """Run two live packages through the unchanged length-prefixed backend."""

    if candidate_role not in {"P0", "P1"}:
        raise ValueError("candidate_role must be P0 or P1")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    replay = Path(replay_path).resolve()
    trace = Path(trace_path).resolve()
    events = Path(events_path).resolve()
    replay.parent.mkdir(parents=True, exist_ok=True)
    events.parent.mkdir(parents=True, exist_ok=True)
    role = 0 if candidate_role == "P0" else 1
    player_specs = (
        (candidate_process, opponent_process)
        if role == 0
        else (opponent_process, candidate_process)
    )
    game_process = _start(game)
    players = (_start(player_specs[0]), _start(player_specs[1]))
    processes = (game_process, *players)
    game_frames: queue.Queue[tuple[str, Any]] = queue.Queue()
    player_frames = (queue.Queue(), queue.Queue())
    stderr_tails = (bytearray(), bytearray(), bytearray())
    readers = [
        threading.Thread(
            target=_reader,
            args=(game_process.stdout, game_frames),
            kwargs={"game": True},
            daemon=True,
        ),
        *(
            threading.Thread(
                target=_reader,
                args=(players[index].stdout, player_frames[index]),
                kwargs={"game": False},
                daemon=True,
            )
            for index in (0, 1)
        ),
    ]
    stderr_readers = [
        threading.Thread(
            target=_drain_stderr,
            args=(
                processes[index].stderr,
                events.with_suffix(f".{('game', 'p0', 'p1')[index]}.stderr.log"),
                stderr_tails[index],
            ),
            daemon=True,
        )
        for index in range(3)
    ]
    for thread in (*readers, *stderr_readers):
        thread.start()
    deadline = time.monotonic() + timeout_s
    ended = False
    try:
        with events.open("w", encoding="utf-8") as event_log:
            def event(kind: str, **details: object) -> None:
                event_log.write(
                    json.dumps({"kind": kind, **details}, ensure_ascii=False) + "\n"
                )
                event_log.flush()

            init = {
                "player_list": [1, 1],
                "player_num": 2,
                "config": {"random_seed": int(seed)},
                "replay": str(replay),
            }
            _write(game_process.stdin, _packet(init))
            event("send_init", seed=int(seed))
            while not ended:
                object_id, payload = _next_frame(
                    game_frames,
                    deadline=deadline,
                    label="game",
                    processes=processes,
                )
                event("game_packet", object=object_id, size=len(payload))
                if object_id in {0, 1}:
                    _write(players[object_id].stdin, payload)
                    event("forward_operation", player=object_id, size=len(payload))
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AntWarMatchError("game emitted invalid JSON") from exc
                if not isinstance(message, dict):
                    raise AntWarMatchError("game message is not an object")
                public_players = message.get("player", [])
                public_content = message.get("content", [])
                if public_players or public_content:
                    if (
                        not isinstance(public_players, list)
                        or not isinstance(public_content, list)
                        or len(public_players) != len(public_content)
                    ):
                        raise AntWarMatchError("game broadcast has invalid shape")
                    for player, content in zip(public_players, public_content):
                        player = int(player)
                        raw = str(content).encode("utf-8")
                        _write(players[player].stdin, raw)
                        event("broadcast_state", player=player, size=len(raw))
                for raw_player in message.get("listen", []):
                    player = int(raw_player)
                    _unused, body = _next_frame(
                        player_frames[player],
                        deadline=deadline,
                        label=f"player {player}",
                        processes=processes,
                    )
                    framed = struct.pack(">I", len(body)) + body
                    reply = {
                        "player": player,
                        "content": framed.decode("latin1"),
                        "time": 0,
                    }
                    _write(game_process.stdin, _packet(reply))
                    event("send_operation", player=player, size=len(framed))
                if "end_state" in message:
                    event("end", end_state=message["end_state"])
                    ended = True
        game_process.wait(timeout=max(deadline - time.monotonic(), 0.1))
        for process in players:
            process.stdin.close()
            process.wait(timeout=5)
        returncodes = (players[0].returncode, players[1].returncode)
        if game_process.returncode != 0 or any(code != 0 for code in returncodes):
            tails = [tail.decode("utf-8", errors="replace") for tail in stderr_tails]
            raise AntWarMatchError(
                "non-zero process return code: "
                f"game={game_process.returncode}, players={returncodes}, stderr={tails}"
            )
        if not ended:
            raise AntWarMatchError("game exited without end_state")
        write_public_trace(replay, trace)
        record = record_from_replay(
            replay,
            candidate=candidate,
            opponent=opponent,
            candidate_role=candidate_role,
            seed=seed,
            trace_path=trace,
        )
        return NativeMatchArtifacts(
            record=record,
            replay_path=replay,
            trace_path=trace,
            events_path=events,
            game_returncode=game_process.returncode,
            player_returncodes=returncodes,
        )
    except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
        tails = [tail.decode("utf-8", errors="replace") for tail in stderr_tails]
        raise AntWarMatchError(f"native match transport failed: {exc}; stderr={tails}") from exc
    finally:
        for process in processes:
            _stop(process)

"""
Stdio Protocol Handler for THUAC RFC (Saiblo Communication Protocol).

The THUAC/Saiblo protocol uses 4-byte length-prefixed messages over stdin/stdout:
- Message format: [4-byte big-endian length][4-byte big-endian target][JSON payload]
- The judger (评测机) sits between the logic and AI processes
- AI processes receive messages on stdin and respond on stdout

This module provides both server-side (game logic) and client-side (AI agent)
protocol handling.
"""

import json
import struct
import sys
import subprocess
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from enum import IntEnum


class MessageTarget(IntEnum):
    """Target identifiers in the Saiblo protocol."""
    JUDGER = -1
    ALL = -1


@dataclass
class ProtocolMessage:
    """A parsed message from the Saiblo protocol."""
    state: int          # round/state number; -1 for end
    listen: List[int]   # players that should process this
    player: List[int]   # players the content is for
    content: List[str]  # message content per player (JSON strings)
    raw: bytes = b""    # raw bytes


class StdioProtocol:
    """
    Handler for the Saiblo stdio communication protocol.

    Implements the THUAC RFC: 4-byte length prefix + optional target + JSON payload.

    Can be used in two modes:
    - Client mode: AI agent communicating with the game logic
    - Server mode: Game logic communicating with AI agents
    """

    HEADER_SIZE = 4  # bytes for length prefix
    TARGET_SIZE = 4  # bytes for target (only in logic->judger direction)

    def __init__(self,
                 stdin_stream=sys.stdin.buffer,
                 stdout_stream=sys.stdout.buffer,
                 debug: bool = False):
        self.stdin = stdin_stream
        self.stdout = stdout_stream
        self.debug = debug
        self._buffer = b""

    # ---- Reading ----

    def read_message(self) -> ProtocolMessage:
        """
        Read a protocol message from stdin.

        Reads 4-byte length prefix, then the JSON payload.
        Returns a parsed ProtocolMessage.
        """
        # Read 4-byte length header
        header = self._read_exact(self.HEADER_SIZE)
        if len(header) < self.HEADER_SIZE:
            raise EOFError("Unexpected end of input stream")

        length = struct.unpack(">I", header)[0]

        # Read the payload
        payload = self._read_exact(length)
        data_str = payload.decode("utf-8")

        if self.debug:
            print(f"[Protocol] Read: {data_str[:200]}...", file=sys.stderr)

        return self._parse_message(data_str, payload)

    def read_ai_message(self) -> Dict[str, Any]:
        """
        Read a message in AI-client format (4-byte ASCII length prefix + JSON).
        This is the format used by some games (e.g., LostSpace) for AI communication.
        """
        header = self._read_exact(4)
        length = int(header.decode("utf-8"))
        payload = self._read_exact(length)
        return json.loads(payload.decode("utf-8"))

    def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes from stdin."""
        data = self.stdin.read(n)
        return data

    def _parse_message(self, data_str: str, raw: bytes) -> ProtocolMessage:
        """Parse a JSON protocol message."""
        try:
            obj = json.loads(data_str)

            # Handle end-of-game message
            if obj.get("state") == -1:
                return ProtocolMessage(
                    state=-1,
                    listen=[],
                    player=[],
                    content=[obj.get("end_info", "")],
                    raw=raw,
                )

            return ProtocolMessage(
                state=obj.get("state", 0),
                listen=obj.get("listen", []),
                player=obj.get("player", []),
                content=obj.get("content", []),
                raw=raw,
            )
        except json.JSONDecodeError:
            return ProtocolMessage(
                state=-1, listen=[], player=[], content=[data_str], raw=raw
            )

    # ---- Writing ----

    def write_message(self, data: Any, target: int = -1):
        """
        Write a message to stdout in protocol format.

        Args:
            data: JSON-serializable object or string
            target: Target player ID (-1 for judger)
        """
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)

        payload = data_str.encode("utf-8")
        length = len(payload)

        header = struct.pack(">Ii", length, target)
        self.stdout.write(header)
        self.stdout.write(payload)
        self.stdout.flush()

        if self.debug:
            print(f"[Protocol] Write: {data_str[:200]}...", file=sys.stderr)

    def write_ai_message(self, data: Any):
        """
        Write a message in AI-client format (4-byte ASCII length + JSON).
        """
        data_str = json.dumps(data) if not isinstance(data, str) else data
        payload = data_str.encode("utf-8")
        length = len(payload)

        header = f"{length:04d}".encode("utf-8")
        self.stdout.write(header)
        self.stdout.write(payload)
        self.stdout.flush()

    def write_command_list(self, commands: List[List[int]]):
        """
        Write a command list in Generals format (space-separated ints, one per line).
        """
        lines = []
        for cmd in commands:
            lines.append(" ".join(str(x) for x in cmd))
        lines.append("8")  # end-of-operations marker
        data = "\n".join(lines) + "\n"
        self.write_message(data)


class SubprocessGameRunner:
    """
    Runs a game logic process and handles stdio communication.

    This wraps the game logic binary so that the framework can communicate with it
    via the Saibolo protocol, enabling subprocess-mode environments.

    Usage:
        runner = SubprocessGameRunner(["python", "game_logic/main.py"])
        runner.start()
        init_info = runner.receive_init()
        # ... run game loop ...
        runner.stop()
    """

    def __init__(self, command: List[str],
                 timeout: float = 30.0,
                 debug: bool = False):
        self.command = command
        self.timeout = timeout
        self.debug = debug
        self.process: Optional[subprocess.Popen] = None
        self._output_queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the game logic subprocess."""
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def send(self, data: Any):
        """Send data to the game logic."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Process not started")

        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)

        payload = data_str.encode("utf-8")
        length = struct.pack(">I", len(payload))
        self.process.stdin.write(length + payload)
        self.process.stdin.flush()

    def receive(self) -> bytes:
        """Receive raw bytes from the game logic."""
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Process not started")

        header = self.process.stdout.read(4)
        if len(header) < 4:
            raise EOFError("Game process ended unexpectedly")

        length = struct.unpack(">I", header)[0]
        return self.process.stdout.read(length)

    def receive_json(self) -> Dict[str, Any]:
        """Receive and parse a JSON message."""
        data = self.receive()
        return json.loads(data.decode("utf-8"))

    def stop(self):
        """Stop the game logic subprocess."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

"""Strict parser for the Generals SDK player protocol."""

from __future__ import annotations

from collections.abc import Sequence
import struct
from typing import BinaryIO


COMMAND_ARITY = {1: 5, 2: 4, 3: 3, 4: None, 5: 2, 6: None, 7: 3, 8: 1, 9: 1}


class ProtocolError(RuntimeError):
    pass


class PacketTooLarge(ProtocolError):
    pass


class MissingEndTurn(ProtocolError):
    pass


class InvalidEncoding(ProtocolError):
    pass


class InvalidCommand(ProtocolError):
    pass


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError(f"unexpected EOF after {len(chunks)} of {size} bytes")
        chunks.extend(block)
    return bytes(chunks)


def read_packet(stream: BinaryIO, max_bytes: int) -> bytes:
    header = _read_exact(stream, 4)
    length = struct.unpack(">I", header)[0]
    if length > max_bytes:
        raise PacketTooLarge(f"packet length {length} exceeds {max_bytes}")
    return _read_exact(stream, length)


def _valid_arity(command: tuple[int, ...]) -> bool:
    code = command[0]
    expected = COMMAND_ARITY.get(code)
    if expected is not None:
        return len(command) == expected
    if code == 4:
        return len(command) in (3, 5)
    if code == 6:
        return len(command) in (4, 6)
    return False


def parse_command_packet(payload: bytes) -> tuple[tuple[int, ...], ...]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidEncoding("command packet is not valid UTF-8") from exc
    if not text:
        raise MissingEndTurn("empty command packet")
    commands: list[tuple[int, ...]] = []
    ended = False
    for raw_line in text.splitlines():
        if not raw_line.strip():
            raise InvalidCommand("blank command line")
        if ended:
            raise InvalidCommand("command after terminal end-turn")
        try:
            command = tuple(int(part) for part in raw_line.split())
        except ValueError as exc:
            raise InvalidCommand("command contains a non-integer") from exc
        if not command or not _valid_arity(command):
            raise InvalidCommand(f"invalid command arity: {command!r}")
        commands.append(command)
        ended = command == (8,)
    if not ended:
        raise MissingEndTurn("command packet has no terminal end-turn")
    return tuple(commands)


def encode_peer_commands(commands: Sequence[Sequence[int]]) -> bytes:
    return "".join(" ".join(str(value) for value in command) + "\n" for command in commands).encode(
        "utf-8"
    )

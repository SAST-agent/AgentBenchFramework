"""Historical DOTO process framing with bounded reads."""

from __future__ import annotations

import os
import selectors
import struct
import time
from dataclasses import dataclass
from typing import BinaryIO


DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024


class DotoProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServerFrame:
    message_type: int
    target: int | None
    payload: bytes


def read_exact(stream: BinaryIO, size: int, timeout: float, label: str) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    try:
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise DotoProtocolError(f"{label} timed out while reading {size} bytes")
            chunk = os.read(stream.fileno(), size - len(data))
            if not chunk:
                raise DotoProtocolError(f"{label} exited while reading {size} bytes")
            data.extend(chunk)
    finally:
        selector.close()
    return bytes(data)


def _read_length(stream: BinaryIO, timeout: float, label: str) -> int:
    length = struct.unpack(">i", read_exact(stream, 4, timeout, label))[0]
    if length < 0:
        raise DotoProtocolError(f"{label} announced negative frame length {length}")
    if length > DEFAULT_MAX_FRAME_SIZE:
        raise DotoProtocolError(f"{label} announced unreasonable frame length {length}")
    return length


def read_server_frame(stream: BinaryIO, timeout: float, label: str = "server") -> ServerFrame:
    length = _read_length(stream, timeout, label)
    if length < 4:
        raise DotoProtocolError(f"{label} announced short server frame {length}")
    message_type = struct.unpack(">i", read_exact(stream, 4, timeout, label))[0]
    target = None
    header = 4
    if message_type != 2:
        if length < 8:
            raise DotoProtocolError(f"{label} announced short routed frame {length}")
        target = struct.unpack(">i", read_exact(stream, 4, timeout, label))[0]
        header = 8
    payload = read_exact(stream, length - header, timeout, label)
    return ServerFrame(message_type, target, payload)


def read_ai_frame(stream: BinaryIO, timeout: float, label: str = "ai") -> bytes:
    length = _read_length(stream, timeout, label)
    return read_exact(stream, length, timeout, label)


def write_ai_observation(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(">i", len(payload)))
    stream.write(payload)
    stream.flush()


def write_server_action(stream: BinaryIO, faction: int, payload: bytes) -> None:
    length = len(payload) + 8
    stream.write(struct.pack(">iii", length, 0, faction))
    stream.write(payload)
    stream.flush()

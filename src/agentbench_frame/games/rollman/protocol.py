"""Frozen Rollman/Saiblo wire framing."""

from __future__ import annotations

import os
import selectors
import struct
import time
from typing import BinaryIO


DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when a process violates or stalls the frozen protocol."""


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    timeout: float | None = None,
    label: str = "stream",
) -> bytes:
    if timeout is None or not hasattr(stream, "fileno"):
        data = stream.read(size)
        if data is None or len(data) != size:
            actual = 0 if data is None else len(data)
            raise ProtocolError(
                f"{label} truncated: expected {size} bytes, received {actual}"
            )
        return data

    data = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise ProtocolError(
                    f"{label} timed out while reading {size} bytes"
                )
            chunk = os.read(stream.fileno(), size - len(data))
            if not chunk:
                raise ProtocolError(
                    f"{label} truncated: expected {size} bytes, received {len(data)}"
                )
            data.extend(chunk)
    finally:
        selector.close()
    return bytes(data)


def encode_ai_frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def decode_ai_frame(
    stream: BinaryIO,
    *,
    timeout: float | None = None,
    label: str = "AI",
    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> bytes:
    size = struct.unpack(
        ">I", _read_exact(stream, 4, timeout=timeout, label=label)
    )[0]
    if size > max_frame_size:
        raise ProtocolError(f"{label} announced unreasonable frame length {size}")
    return _read_exact(stream, size, timeout=timeout, label=label)


def encode_logic_input(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def decode_logic_frame(
    stream: BinaryIO,
    *,
    timeout: float | None = None,
    label: str = "logic",
    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> tuple[int, bytes]:
    header = _read_exact(stream, 8, timeout=timeout, label=label)
    size, target = struct.unpack(">Ii", header)
    if size > max_frame_size:
        raise ProtocolError(f"{label} announced unreasonable frame length {size}")
    return target, _read_exact(stream, size, timeout=timeout, label=label)


def write_logic_input(stream: BinaryIO, payload: bytes) -> None:
    stream.write(encode_logic_input(payload))
    stream.flush()


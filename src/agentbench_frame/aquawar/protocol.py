"""Framing primitives for the historical AquaWar process protocol."""

from __future__ import annotations

import os
import selectors
import struct
import time
from typing import BinaryIO


DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024


class AquaWarProtocolError(RuntimeError):
    """Raised when a process violates or stalls the AquaWar protocol."""


def read_exact(stream: BinaryIO, size: int, timeout: float, label: str) -> bytes:
    data = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise AquaWarProtocolError(
                    f"{label} timed out while reading {size} bytes"
                )
            chunk = os.read(stream.fileno(), size - len(data))
            if not chunk:
                raise AquaWarProtocolError(
                    f"{label} exited while reading {size} bytes"
                )
            data.extend(chunk)
    finally:
        selector.close()
    return bytes(data)


def read_frame(
    stream: BinaryIO,
    timeout: float,
    label: str,
    *,
    has_target: bool = False,
    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> bytes:
    length = struct.unpack(">I", read_exact(stream, 4, timeout, label))[0]
    if has_target:
        read_exact(stream, 4, timeout, label)
    if length > max_frame_size:
        raise AquaWarProtocolError(
            f"{label} announced unreasonable frame length {length}"
        )
    return read_exact(stream, length, timeout, label)


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()

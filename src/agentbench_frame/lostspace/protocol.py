"""Framing primitives for the LostSpace Saiblo process protocol.

Wire framing (identical to AquaWar's Saiblo framing):

* judger -> logic: ``[4-byte big-endian length][json]`` (no target word)
* logic -> judger: ``[4-byte big-endian length][4-byte big-endian target][json]``
  (target ``-1`` means "to judger / broadcast")
* judger -> AI:   the logic pre-wraps each ``content`` with
  ``convert_byte_str_for_ai`` (4 ASCII digits + json); the harness forwards
  those bytes verbatim to the AI's stdin.
* AI -> judger:   ``[4-byte big-endian length][json]`` (binary, no target)

``read_exact`` is thread-based rather than ``selectors``-based so that it works
on Windows too: Windows ``select`` only handles sockets, not anonymous pipes,
whereas ``os.read`` on a blocking pipe fd works cross-platform.
"""

from __future__ import annotations

import os
import struct
import threading
from typing import BinaryIO


DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024


class LostSpaceProtocolError(RuntimeError):
    """Raised when a process violates or stalls the LostSpace protocol."""


def read_exact(stream: BinaryIO, size: int, timeout: float, label: str) -> bytes:
    """Read exactly ``size`` bytes from ``stream`` or raise within ``timeout``.

    Uses a daemon reader thread so a stalled peer cannot block forever and so
    the call works on Windows pipes (which ``selectors`` cannot watch).
    """
    fd = stream.fileno()
    result: dict = {}

    def worker() -> None:
        data = bytearray()
        try:
            while len(data) < size:
                chunk = os.read(fd, size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            result["data"] = bytes(data)
        except OSError as exc:  # closed fd / bad descriptor mid-read
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise LostSpaceProtocolError(
            f"{label} timed out while reading {size} bytes"
        )
    if "error" in result:
        raise result["error"]
    data = result.get("data", b"")
    if len(data) < size:
        raise LostSpaceProtocolError(
            f"{label} exited while reading {size} bytes"
        )
    return data


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
        raise LostSpaceProtocolError(
            f"{label} announced unreasonable frame length {length}"
        )
    return read_exact(stream, length, timeout, label)


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()

import os
import struct

import pytest

from agentbench_frame.doto.protocol import (
    DotoProtocolError,
    read_ai_frame,
    read_server_frame,
)


def pipe_with(payload: bytes):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    return os.fdopen(read_fd, "rb", buffering=0)


def test_server_frame_keeps_type_target_and_json():
    payload = b'{"frame":0,"faction":1}'
    raw = struct.pack(">iii", len(payload) + 8, 0, 1) + payload
    stream = pipe_with(raw)
    try:
        frame = read_server_frame(stream, 0.2, "server")
    finally:
        stream.close()

    assert (frame.message_type, frame.target, frame.payload) == (0, 1, payload)


def test_ai_frame_uses_length_including_no_header():
    payload = b'{"flag":0}'
    stream = pipe_with(struct.pack(">i", len(payload)) + payload)
    try:
        assert read_ai_frame(stream, 0.2, "ai") == payload
    finally:
        stream.close()


def test_negative_server_length_is_rejected():
    stream = pipe_with(struct.pack(">i", -1))
    try:
        with pytest.raises(DotoProtocolError, match="negative frame length"):
            read_server_frame(stream, 0.2, "server")
    finally:
        stream.close()

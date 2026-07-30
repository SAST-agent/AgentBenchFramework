import io
import struct

import pytest

from agentbench_frame.games.rollman.protocol import (
    ProtocolError,
    decode_ai_frame,
    decode_logic_frame,
    encode_ai_frame,
    encode_logic_input,
)


def test_ai_frame_is_four_byte_big_endian_length_plus_payload():
    encoded = encode_ai_frame(b'{"role":0,"action":"4"}')

    assert encoded[:4] == struct.pack(">I", 23)
    assert decode_ai_frame(io.BytesIO(encoded)) == b'{"role":0,"action":"4"}'


def test_logic_frame_has_unsigned_length_and_signed_target():
    payload = b'{"state":1}'
    encoded = struct.pack(">Ii", len(payload), -1) + payload

    target, decoded = decode_logic_frame(io.BytesIO(encoded))
    assert target == -1
    assert decoded == payload
    assert encode_logic_input(payload) == struct.pack(">I", len(payload)) + payload


def test_protocol_fails_closed_on_truncated_or_oversized_frames():
    with pytest.raises(ProtocolError, match="truncated"):
        decode_ai_frame(io.BytesIO(struct.pack(">I", 10) + b"short"))
    with pytest.raises(ProtocolError, match="unreasonable"):
        decode_ai_frame(io.BytesIO(struct.pack(">I", 100)), max_frame_size=20)


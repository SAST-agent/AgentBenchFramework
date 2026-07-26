import io
import struct

import pytest

from agentbench_frame.generals.protocol import (
    InvalidCommand,
    InvalidEncoding,
    MissingEndTurn,
    PacketTooLarge,
    encode_peer_commands,
    parse_command_packet,
    read_packet,
)


class FragmentedStream:
    def __init__(self, data: bytes, chunk: int):
        self._data = io.BytesIO(data)
        self._chunk = chunk

    def read(self, size: int = -1) -> bytes:
        return self._data.read(min(size, self._chunk))


def test_reads_fragmented_length_prefixed_packet():
    payload = b"1 7 7 4 9\n8\n"
    stream = FragmentedStream(struct.pack(">I", len(payload)) + payload, chunk=1)
    assert read_packet(stream, max_bytes=65536) == payload


def test_rejects_packet_larger_than_limit_before_reading_body():
    stream = io.BytesIO(struct.pack(">I", 65537))
    with pytest.raises(PacketTooLarge):
        read_packet(stream, max_bytes=65536)


def test_commands_require_exactly_one_terminal_end_turn():
    with pytest.raises(MissingEndTurn):
        parse_command_packet(b"1 7 7 4 9\n")
    assert parse_command_packet(b"1 7 7 4 9\n8\n") == ((1, 7, 7, 4, 9), (8,))
    with pytest.raises(InvalidCommand, match="after terminal"):
        parse_command_packet(b"8\n8\n")


def test_peer_commands_use_plain_newline_protocol():
    assert encode_peer_commands(((1, 7, 7, 4, 9), (8,))) == b"1 7 7 4 9\n8\n"


@pytest.mark.parametrize("payload", [b"", b"\n", b"abc\n8\n", b"1 2\n8\n"])
def test_rejects_malformed_commands(payload):
    with pytest.raises((MissingEndTurn, InvalidCommand)):
        parse_command_packet(payload)


def test_rejects_invalid_utf8():
    with pytest.raises(InvalidEncoding):
        parse_command_packet(b"\xff\n8\n")

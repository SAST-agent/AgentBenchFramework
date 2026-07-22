import io
import os
import struct
import unittest

from agentbench_frame.aquawar.protocol import (
    AquaWarProtocolError,
    read_exact,
    read_frame,
    write_frame,
)


class ProtocolTest(unittest.TestCase):
    def pipe_with(self, payload: bytes):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, payload)
        os.close(write_fd)
        return os.fdopen(read_fd, "rb", buffering=0)

    def test_write_frame_uses_big_endian_length(self):
        stream = io.BytesIO()

        write_frame(stream, b'{"ok":true}')

        self.assertEqual(stream.getvalue(), b"\x00\x00\x00\x0b" + b'{"ok":true}')

    def test_read_frame_discards_logic_target_word(self):
        payload = b'{"state":1}'
        stream = self.pipe_with(struct.pack(">Ii", len(payload), 7) + payload)
        self.addCleanup(stream.close)

        result = read_frame(stream, timeout=0.2, label="logic", has_target=True)

        self.assertEqual(result, payload)

    def test_read_frame_rejects_oversized_payload(self):
        stream = self.pipe_with(struct.pack(">I", 9))
        self.addCleanup(stream.close)

        with self.assertRaisesRegex(AquaWarProtocolError, "unreasonable frame length 9"):
            read_frame(stream, timeout=0.2, label="ai", max_frame_size=8)

    def test_read_exact_reports_early_eof(self):
        stream = self.pipe_with(b"ab")
        self.addCleanup(stream.close)

        with self.assertRaisesRegex(AquaWarProtocolError, "exited while reading 4 bytes"):
            read_exact(stream, 4, timeout=0.2, label="logic")

    def test_read_exact_times_out(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb", buffering=0)
        self.addCleanup(stream.close)
        self.addCleanup(os.close, write_fd)

        with self.assertRaisesRegex(AquaWarProtocolError, "timed out while reading 1 bytes"):
            read_exact(stream, 1, timeout=0.01, label="player 0")


if __name__ == "__main__":
    unittest.main()

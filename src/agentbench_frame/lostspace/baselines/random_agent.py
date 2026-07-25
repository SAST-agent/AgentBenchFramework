"""Random LostSpace baseline.

A minimal but protocol-correct AI client: on its own turn it attempts one
random adjacent move, reads the response, then ends the turn. Everything else
(other players' round-begin, off-round "see" notifications, action responses)
is read and ignored. Used as the default table filler and a weak reference
opponent.

Speaks the real LostSpace wire format:
  - receive: 4 ASCII digits (length) + JSON   (``convert_byte_str_for_ai``)
  - send:     4-byte big-endian length + JSON (``convert_to_bytes``)
"""
import json
import random
import struct
import sys


# Communication-format move deltas (0..6 grid, z unchanged).
_DELTAS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
           (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0)]


class RandomAgent:
    def __init__(self, seed=None):
        self.id = None
        self.pos = [0, 0, 1]
        self._rng = random.Random(seed)

    def _recv(self):
        prefix = sys.stdin.buffer.read(4)
        if len(prefix) < 4:
            raise SystemExit(0)
        size = int(prefix.decode("utf-8"))
        return json.loads(sys.stdin.buffer.read(size).decode("utf-8"))

    def _send(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        sys.stdout.buffer.write(struct.pack(">I", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

    def _random_move(self):
        x, y, z = self.pos
        dx, dy, _ = self._rng.choice(_DELTAS)
        self._send({"type": "action", "action": ["move", [x + dx, y + dy, z]]})
        self._recv()  # consume the move_response addressed to us

    def run(self):
        while True:
            msg = self._recv()
            msg_type = msg.get("type")
            if msg_type == "id":
                self.id = msg["id"]
                bx, by = msg["birth_pos"]
                self.pos = [bx, by, 1]
            elif msg_type == "roundbegin":
                if msg.get("inturn") == self.id:
                    self.pos = msg.get("pos", self.pos)
                    self._random_move()
                    self._send({"type": "finish"})
            # action responses / offround notifications: ignore


def main():
    RandomAgent().run()


if __name__ == "__main__":
    main()

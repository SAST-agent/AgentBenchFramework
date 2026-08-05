import json
import struct
import sys
from pathlib import Path


mode = Path(sys.argv[0]).stem


def read_message():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        return None
    size = struct.unpack(">i", raw)[0]
    return json.loads(sys.stdin.buffer.read(size))


read_message()
for message in iter(read_message, None):
    if message.get("frame") == -1:
        break
    if mode == "crash":
        raise SystemExit(9)
    x = -1
    y = -1
    if mode == "left":
        x, y = 9.9, 175.0
    elif mode == "right":
        x, y = 10.1, 175.0
    action = {
        "flag": 0,
        "move": [[x, y]] + [[-1, -1]] * 4,
        "shoot": [[-1, -1]] * 5,
        "meteor": [[-1, -1]] * 5,
        "flash": [False] * 5,
    }
    payload = json.dumps(action, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">i", len(payload)) + payload)
    sys.stdout.buffer.flush()

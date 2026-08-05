import json
import struct
import sys


def read_message():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        return None
    size = struct.unpack(">i", raw)[0]
    return json.loads(sys.stdin.buffer.read(size))


read_message()  # faction initialization
while True:
    message = read_message()
    if message is None or message.get("frame") == -1:
        break
    action = {
        "flag": 0,
        "move": [[-1, -1]] * 5,
        "shoot": [[-1, -1]] * 5,
        "meteor": [[-1, -1]] * 5,
        "flash": [False] * 5,
        "debug": "",
    }
    payload = json.dumps(action, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">i", len(payload)) + payload)
    sys.stdout.buffer.flush()

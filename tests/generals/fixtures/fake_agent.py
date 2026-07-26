import json
import os
import struct
import sys
import time


def send(payload: bytes) -> None:
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()


mode = sys.argv[1]
initial = json.loads(sys.stdin.buffer.readline())
if mode == "valid":
    send(b"8\n")
elif mode == "bad":
    send(b"1 2\n8\n")
elif mode == "stderr":
    sys.stderr.write("x" * 100_000)
    sys.stderr.flush()
    send(b"8\n")
elif mode == "exit":
    raise SystemExit(3)
elif mode == "hang":
    time.sleep(60)

import json
import os
import struct
import sys
import time


def send(payload: bytes) -> None:
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()


def read_peer():
    while True:
        line = sys.stdin.buffer.readline()
        if not line or line.strip() == b"8":
            return


mode = sys.argv[1]
initial = json.loads(sys.stdin.buffer.readline())
if mode == "valid":
    seat = initial["Player"]
    for _ in range(600):
        if seat == 1:
            read_peer()
        send(b"8\n")
        if seat == 0:
            read_peer()
elif mode == "bad":
    send(b"1 2\n8\n")
elif mode == "illegal":
    send(b"1 -1 -1 1 10\n8\n")
elif mode == "stderr":
    sys.stderr.write("x" * 100_000)
    sys.stderr.flush()
    send(b"8\n")
elif mode == "exit":
    raise SystemExit(3)
elif mode == "hang":
    time.sleep(60)

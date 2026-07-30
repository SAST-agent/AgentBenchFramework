#!/usr/bin/env python3
import json
import struct
import sys


def send_action() -> None:
    payload = json.dumps({"role": 1, "action": "0 0 0"}).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()


sys.stdin.readline()
first = True
for line in sys.stdin:
    if not line.strip():
        continue
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if first or "pacman_action" in value:
        send_action()
        first = False


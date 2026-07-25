#!/usr/bin/env python3
"""Minimal LostSpace AI client for tests.

Speaks the real LostSpace AI wire format:
  - receive: 4 ASCII digits (length) + JSON payload  (convert_byte_str_for_ai)
  - send:     4-byte big-endian length + JSON payload (convert_to_bytes)

Reads the id broadcast then the round-begin observation, replies with one
action, then stays alive so the harness must clean the process up.
"""
import argparse
import json
import os
import struct
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=float, default=0.0,
                    help="delay before sending the action (timeout test)")
parser.add_argument("--pid-file", help="write this process's pid to a file")
args = parser.parse_args()

if args.pid_file:
    with open(args.pid_file, "w") as output:
        output.write(str(os.getpid()))


def recv_ascii() -> dict:
    prefix = sys.stdin.buffer.read(4)
    if len(prefix) < 4:
        raise SystemExit(0)
    size = int(prefix.decode("utf-8"))
    payload = sys.stdin.buffer.read(size)
    return json.loads(payload.decode("utf-8"))


def send_binary(obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


recv_ascii()                       # {"type": "id", ...}
roundbegin = recv_ascii()          # {"type": "roundbegin", "inturn": 0, ...}
time.sleep(args.sleep)
send_binary({"type": "action", "action": ["move", [0, 0, roundbegin.get("state", 1)]]})

# Remain alive so the harness must clean us up after the logic finishes.
time.sleep(60)

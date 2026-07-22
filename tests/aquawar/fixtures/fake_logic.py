#!/usr/bin/env python3
import argparse
import json
import struct
import sys


def read_frame():
    size = struct.unpack(">I", sys.stdin.buffer.read(4))[0]
    return sys.stdin.buffer.read(size)


def write_logic_message(value):
    payload = value if isinstance(value, bytes) else json.dumps(value).encode()
    sys.stdout.buffer.write(struct.pack(">Ii", len(payload), -1))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("success", "malformed"), default="success")
args = parser.parse_args()

read_frame()
if args.mode == "malformed":
    write_logic_message(b"not-json")
    raise SystemExit(0)

observation = '{"turn":1}'
write_logic_message(
    {
        "state": 1,
        "player": [0],
        "content": [observation],
        "listen": [0],
    }
)
read_frame()
write_logic_message(
    {
        "state": -1,
        "end_info": json.dumps({"0": 100, "1": 0}),
    }
)

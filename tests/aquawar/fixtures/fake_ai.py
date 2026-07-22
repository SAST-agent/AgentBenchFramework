#!/usr/bin/env python3
import argparse
import os
import struct
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--pid-file")
args = parser.parse_args()

if args.pid_file:
    with open(args.pid_file, "w") as output:
        output.write(str(os.getpid()))

sys.stdin.buffer.read(len(b'{"turn":1}'))
time.sleep(args.sleep)
payload = b'{"action":"pass"}'
sys.stdout.buffer.write(struct.pack(">I", len(payload)))
sys.stdout.buffer.write(payload)
sys.stdout.buffer.flush()

# Remain alive so the harness must clean us up after the logic finishes.
time.sleep(60)

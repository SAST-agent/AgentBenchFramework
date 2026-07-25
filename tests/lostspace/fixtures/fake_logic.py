#!/usr/bin/env python3
"""Minimal 4-player LostSpace logic for tests.

Emits the Saiblo internal protocol (length + target(-1) + json) that the
harness reads with has_target=True:

  1. consume the init frame from the judger
  2. broadcast the id message to all 4 players (no listener)
  3. send a round-begin observation to all 4, player 0 listens/responds
  4. consume player 0's action
  5. emit end-of-game with a 4-player ranking (4=1st ... 1=4th)
"""
import argparse
import json
import struct
import sys


def read_frame() -> bytes:
    size = struct.unpack(">I", sys.stdin.buffer.read(4))[0]
    return sys.stdin.buffer.read(size)


def write_logic_message(value) -> None:
    payload = value if isinstance(value, bytes) else json.dumps(value).encode()
    sys.stdout.buffer.write(struct.pack(">Ii", len(payload), -1))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def ascii_wrap(obj) -> str:
    text = json.dumps(obj)
    return f"{len(text):04d}{text}"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=("success", "malformed", "notify"),
    default="success",
)
args = parser.parse_args()

read_frame()  # init from judger
if args.mode == "malformed":
    write_logic_message(b"not-json")
    raise SystemExit(0)

# 1) id broadcast: every player learns its id, nobody responds.
write_logic_message({
    "state": 0,
    "listen": [],
    "player": [0, 1, 2, 3],
    "content": [ascii_wrap({"type": "id", "id": i, "birth_pos": [0, 0]}) for i in range(4)],
})

# 2) round begin: all 4 see the observation, player 0 must act.
write_logic_message({
    "state": 1,
    "listen": [0],
    "player": [0, 1, 2, 3],
    "content": [ascii_wrap({"type": "roundbegin", "state": 1, "inturn": 0}) for _ in range(4)],
})

read_frame()  # consume player 0's routed action

if args.mode == "notify":
    # Reproduce the real-game notification frame: the in-turn player (0) is
    # both the listener and the recipient of an `other_death` status update.
    # The logic does NOT expect a reply here (it would continue its read loop
    # or end the game); a harness that reads a reply from player 0 deadlocks
    # until it times out.
    write_logic_message({
        "state": 1,
        "listen": [0],
        "player": [0],
        "content": [ascii_wrap({"type": "other_death", "playerid": 2})],
    })

# 3) end-of-game: 4-player ranking points (4=1st ... 1=4th).
write_logic_message({
    "state": -1,
    "end_info": json.dumps({"0": 4, "1": 3, "2": 2, "3": 1}),
})

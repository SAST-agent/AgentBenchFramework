#!/usr/bin/env python3
import json
import struct
import sys


def read_frame():
    size = struct.unpack(">I", sys.stdin.buffer.read(4))[0]
    return json.loads(sys.stdin.buffer.read(size))


def send(value):
    payload = json.dumps(value).encode()
    sys.stdout.buffer.write(struct.pack(">Ii", len(payload), -1) + payload)
    sys.stdout.buffer.flush()


init = read_frame()
replay_path = init["replay"]
initial = {
    "level": 1,
    "round": 0,
    "board_size": 3,
    "board": [[0, 0, 0], [0, 9, 0], [0, 0, 0]],
    "pacman_skill_status": [0, 0, 0, 0, 0],
    "pacman_coord": [1, 1],
    "ghosts_coord": [[1, 1], [1, 1], [1, 1]],
    "score": [0, 0],
    "beannumber": 0,
    "portal_available": False,
    "portal_coord": [1, 1],
}
with open(replay_path, "w", encoding="utf-8") as replay:
    replay.write(json.dumps(initial) + "\n")
send({"watch": json.dumps(initial) + "\n"})
send({"state": 1, "listen": [], "player": [0, 1], "content": ["0\n", "1\n"]})
send(
    {
        "state": 2,
        "listen": [],
        "player": [0, 1],
        "content": [json.dumps(initial) + "\n", json.dumps(initial) + "\n"],
    }
)
responses = {}
for state, player in ((3, 0), (4, 1)):
    send({"state": state, "listen": [player], "player": [], "content": []})
    responses[player] = read_frame()

round_frame = {
    "round": 1,
    "level": 1,
    "pacman_step_block": [[1, 1], [1, 1]],
    "pacman_coord": [1, 1],
    "pacman_skills": [0, 0, 0, 0, 0],
    "ghosts_step_block": [[[1, 1], [1, 1]]] * 3,
    "ghosts_coord": [[1, 1], [1, 1], [1, 1]],
    "score": [0, 0],
    "events": [3],
    "portal_available": False,
    "StopReason": None,
}
terminal = dict(round_frame, StopReason="time is up", events=[])
with open(replay_path, "a", encoding="utf-8") as replay:
    replay.write(json.dumps(round_frame) + "\n")
    replay.write(json.dumps(terminal) + "\n")
send({"watch": json.dumps(round_frame) + "\n"})
send(
    {
        "state": -1,
        "end_info": json.dumps({"0": 0, "1": 0}),
        "end_state": json.dumps(["OK", "OK"]),
    }
)


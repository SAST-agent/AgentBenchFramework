import json
import struct
import sys
import zipfile
from pathlib import Path


def send(message, message_type=0, target=-1):
    body = json.dumps(message, separators=(",", ":")).encode()
    if message_type == 2:
        packet = struct.pack(">ii", len(body) + 4, message_type) + body
    else:
        packet = struct.pack(">iii", len(body) + 8, message_type, target) + body
    sys.stdout.buffer.write(packet)
    sys.stdout.buffer.flush()


def receive():
    length, message_type, faction = struct.unpack(">iii", sys.stdin.buffer.read(12))
    return faction, json.loads(sys.stdin.buffer.read(length - 8))


for faction in (0, 1):
    send({"frame": 0, "map": 0, "faction": faction}, target=faction)

frame = {
    "frame": 1,
    "humans": "[]",
    "fireballs": "[]",
    "meteors": "[]",
    "balls": "[]",
    "scores": "[12.0, 7.0]",
    "bonus": "[]",
}
send(frame)
actions = [receive(), receive()]
assert {faction for faction, _ in actions} == {0, 1}
send({"frame": -1, "scores": "[12.0, 7.0]"}, message_type=2)

replay_path = Path(sys.argv[1])
replay_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(replay_path, "w") as archive:
    archive.writestr("replay.json", json.dumps([frame, {"frame": -1, "scores": "[12.0, 7.0]"}]))

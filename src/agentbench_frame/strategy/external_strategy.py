"""
ExternalStrategy: drives an external bot process over stdio.

Wire protocol (one request/reply per turn):
  - host -> process: 4-byte big-endian length prefix + UTF-8 JSON request
    {"player": int, "round": int, "coins": [...], "bases": [...],
     "towers": [[...]], "ants": [[...]], "weapon_cooldowns": [[...]],
     "active_effects": [[...]], "die_count": [...], "old_count": [...]}
  - process -> host: 4-byte big-endian length prefix + UTF-8 JSON reply
    [[op_type, arg0, arg1], ...]

Any process speaking this contract can be a strategy. A reference bot is
emitted alongside the population so external iteration is reproducible.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from typing import Any, List, Optional

from agentbench_frame.strategy.base import BaseStrategy, StrategyMeta

REFERENCE_BOT = '''import json, struct, sys


def read_msg(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    n = struct.unpack(">I", header)[0]
    return json.loads(stream.read(n).decode("utf-8"))


def write_msg(stream, obj):
    payload = json.dumps(obj).encode("utf-8")
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        req = read_msg(stdin)
        if req is None:
            return
        write_msg(stdout, [])


if __name__ == "__main__":
    main()
'''


def _encode_state(backend_state, player: int) -> dict:
    public = backend_state.to_public_round_state()
    return {
        "player": player,
        "round": backend_state.round_index,
        "coins": list(backend_state.coins),
        "bases": [
            {"player": b.player, "x": b.x, "y": b.y, "hp": b.hp,
             "generation_level": b.generation_level, "ant_level": b.ant_level}
            for b in backend_state.bases
        ],
        "towers": [list(t) for t in public.towers],
        "ants": [list(a) for a in public.ants],
        "weapon_cooldowns": [list(w) for w in (public.weapon_cooldowns or ())],
        "active_effects": [list(e) for e in (public.active_effects or [])],
        "die_count": list(backend_state.die_count),
        "old_count": list(backend_state.old_count),
    }


class ExternalStrategy(BaseStrategy):
    kind = "external"

    def __init__(self,
                 name: str = "ExternalStrategy",
                 command: Optional[List[str]] = None,
                 script_path: Optional[str] = None,
                 cwd: Optional[str] = None,
                 timeout: float = 5.0,
                 meta: Optional[StrategyMeta] = None,
                 **kwargs):
        super().__init__(name=name, meta=meta)
        self.command = command
        self.script_path = script_path
        self.cwd = cwd
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self.meta.kind = self.kind
        if command:
            self.meta.extra["command"] = list(command)
        if script_path:
            self.meta.extra["script"] = os.path.basename(script_path)

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            self._cleanup()
        cmd = self.command
        if cmd is None:
            script = self.script_path or self._resolve_script()
            cmd = [sys.executable, script]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self.cwd,
        )

    def _resolve_script(self) -> str:
        local = os.path.join(".", "bot.py")
        return local

    def _decide(self, backend_state, player: int) -> List[List[int]]:
        self._ensure_started()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return []
        req = _encode_state(backend_state, player)
        self._send(req)
        reply = self._recv()
        if reply is None:
            return []
        return self._parse_ops(reply)

    def _send(self, obj: dict):
        payload = json.dumps(obj).encode("utf-8")
        self._proc.stdin.write(struct.pack(">I", len(payload)))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _recv(self):
        stream = self._proc.stdout
        header = stream.read(4)
        if len(header) < 4:
            return None
        n = struct.unpack(">I", header)[0]
        data = stream.read(n)
        if len(data) < n:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_ops(reply) -> List[List[int]]:
        if not isinstance(reply, list):
            return []
        out: List[List[int]] = []
        for item in reply:
            if isinstance(item, list):
                try:
                    out.append([int(t) for t in item])
                except (TypeError, ValueError):
                    continue
        return out

    def _cleanup(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def reset(self):
        self._cleanup()

    def close(self):
        self._cleanup()

    def _save_artifacts(self, path: str):
        target = os.path.join(path, "bot.py")
        if self.script_path and os.path.exists(self.script_path):
            shutil.copyfile(self.script_path, target)
            self.meta.extra["script"] = "bot.py"
        elif self.command:
            self.meta.extra["command"] = list(self.command)
        else:
            with open(target, "w") as f:
                f.write(REFERENCE_BOT)
            self.meta.extra["script"] = "bot.py"

    def _load_artifacts(self, path: str):
        script = self.meta.extra.get("script")
        command = self.meta.extra.get("command")
        if script:
            self.script_path = os.path.join(path, script)
        if command:
            self.command = list(command)

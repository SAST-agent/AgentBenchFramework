#!/usr/bin/env python3
"""Run an editable Rollman ``ai.py`` against the pinned Python SDK."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any


def _load_policy(workspace: Path):
    source = workspace / "ai.py"
    if not source.is_file():
        raise FileNotFoundError(f"candidate workspace has no ai.py: {source}")
    spec = importlib.util.spec_from_file_location("agentbench_candidate_ai", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate policy: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = getattr(module, "ai_func", None)
    if not callable(policy):
        raise TypeError("candidate ai.py must define callable ai_func(game_state)")
    return policy


def _decision(value: Any) -> tuple[int, str | None]:
    memory_id = None
    if isinstance(value, dict):
        action = value.get("action")
        memory_id = value.get("memory_id")
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Rollman ai_func sequence must contain one action")
        action = value[0]
    else:
        action = value
    if not isinstance(action, int) or action not in range(5):
        raise ValueError("Rollman action must be an integer in 0..4")
    if memory_id is not None:
        memory_id = str(memory_id)
        if len(memory_id) > 200:
            raise ValueError("memory_id cannot exceed 200 characters")
    return action, memory_id


def _send(action: int, memory_id: str | None) -> None:
    message: dict[str, Any] = {"role": 0, "action": str(action)}
    if memory_id is not None:
        message["memory_id"] = memory_id
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--sdk-root", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    sdk_root = Path(args.sdk_root).resolve()
    if not (sdk_root / "core" / "GymEnvironment.py").is_file():
        parser.error(f"invalid pinned Pacman SDK root: {sdk_root}")
    sys.path.insert(0, str(sdk_root))
    sys.path.insert(0, str(workspace))
    policy = _load_policy(workspace)
    from core.GymEnvironment import PacmanEnv

    env = PacmanEnv()
    player_id = int(input())
    level_change = True
    while True:
        if level_change:
            env.ai_reset(json.loads(input()))
            level_change = False
        if player_id == 1:
            input()
        action, memory_id = _decision(policy(env.game_state()))
        _send(action, memory_id)
        if player_id == 0:
            input()
        feedback = json.loads(input())
        _, _, _, level_change, _ = env.step(
            int(feedback["pacman_action"]),
            [int(value) for value in feedback["ghosts_action"]],
        )


if __name__ == "__main__":
    raise SystemExit(main())

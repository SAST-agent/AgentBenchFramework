#!/usr/bin/env python3
"""Probe one candidate on a fixed sequence of visible Rollman states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agentbench_frame.eval.measurement import canonical_state_id


def _load_policy(workspace: Path):
    source = workspace / "ai.py"
    spec = importlib.util.spec_from_file_location("agentbench_probe_ai", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate policy: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = getattr(module, "ai_func", None)
    if not callable(policy):
        raise TypeError("candidate ai.py must define ai_func(game_state)")
    return policy


def _decision(value: Any) -> tuple[int, str | None]:
    if isinstance(value, dict):
        action = value.get("action")
        memory_id = value.get("memory_id")
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        action = value[0]
        memory_id = None
    else:
        action = value
        memory_id = None
    if not isinstance(action, int) or action not in range(5):
        raise ValueError("probed Rollman action must be an integer in 0..4")
    return action, None if memory_id is None else str(memory_id)


def probe_states(
    workspace: str | Path,
    sdk_root: str | Path,
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate = Path(workspace).resolve()
    sdk = Path(sdk_root).resolve()
    sys.path.insert(0, str(sdk))
    sys.path.insert(0, str(candidate))
    from core.GymEnvironment import PacmanEnv
    from core.gamedata import GameState

    policy = _load_policy(candidate)
    decisions = []
    for index, state in enumerate(states):
        environment = PacmanEnv()
        environment.ai_reset(dict(state))
        score = state["score"]
        game_state = GameState(
            space_info=environment.game_state().space_info,
            level=int(state["level"]),
            round=int(state["round"]),
            board_size=int(state["board_size"]),
            board=np.asarray(state["board"], dtype=int),
            pacman_skill_status=[
                int(value) for value in state["pacman_skill_status"]
            ],
            pacman_pos=np.asarray(state["pacman_coord"], dtype=int),
            ghosts_pos=[
                np.asarray(value, dtype=int) for value in state["ghosts_coord"]
            ],
            pacman_score=int(score[0]),
            ghosts_score=int(score[1]),
            beannumber=int(state["beannumber"]),
            portal_available=bool(state["portal_available"]),
            portal_coord=np.asarray(state["portal_coord"], dtype=int),
        )
        action, memory_id = _decision(policy(game_state))
        decisions.append(
            {
                "reference_index": index,
                "state_id": canonical_state_id(state),
                "action": action,
                "memory_id": memory_id,
            }
        )
    return decisions


def run_probe_episode(
    *,
    workspace: str | Path,
    sdk_root: str | Path,
    states: Sequence[Mapping[str, Any]],
    artifact_path: str | Path,
    timeout_s: float = 30.0,
) -> tuple[dict[str, Any], ...]:
    output = Path(artifact_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_path = output.with_suffix(".states.json")
    input_path.write_text(
        json.dumps(list(states), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
        if name in os.environ
    }
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--workspace",
            str(Path(workspace).resolve()),
            "--sdk-root",
            str(Path(sdk_root).resolve()),
            "--states",
            str(input_path),
            "--output",
            str(output),
        ),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"policy probe failed: {(completed.stderr or completed.stdout)[-4000:]}"
        )
    value = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("policy probe output must be a list")
    return tuple(dict(item) for item in value)


def load_trace_decisions(path: str | Path) -> tuple[dict[str, Any], ...]:
    decisions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("type") == "action" and value.get("player") == 0:
                decisions.append(dict(value["decision"]))
    if not decisions:
        raise ValueError(f"trace contains no Rollman decisions: {path}")
    return tuple(decisions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))
    result = probe_states(args.workspace, args.sdk_root, states)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

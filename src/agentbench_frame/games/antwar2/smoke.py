"""Candidate-side AntWar2 import and public-state legality smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from agentbench_frame.hl.game_profile import SmokeResult


_MARKER = "AGENTBENCH_SMOKE_RESULT="
_PROGRAM = r'''
import json
from ai import AI
from SDK.backend.state import create_python_backend_state

counts = {}
for player in (0, 1):
    agent = AI()
    agent.on_match_start(player, 7)
    state = create_python_backend_state(seed=7)
    operations = agent.choose_operations(state, player)
    if not isinstance(operations, list):
        raise TypeError("AI.choose_operations must return a list")
    accepted = []
    for operation in operations:
        if not state.can_apply_operation(player, operation, accepted):
            raise ValueError(f"role P{player} emitted an illegal operation: {operation}")
        accepted.append(operation)
    counts[f"P{player}"] = len(accepted)
print("AGENTBENCH_SMOKE_RESULT=" + json.dumps({"counts": counts}, sort_keys=True))
'''


def verify_candidate_smoke(
    workspace: str | Path,
    *,
    timeout_s: float = 30.0,
) -> SmokeResult:
    candidate = Path(workspace).resolve()
    source = candidate / "ai.py"
    for required in (source, candidate / "main.py", candidate / "SDK/__init__.py"):
        if not required.is_file():
            return SmokeResult(
                status="failed",
                error=f"candidate package is missing {required.relative_to(candidate)}",
                artifacts={},
            )
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")
        if name in os.environ
    }
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            (sys.executable, "-c", _PROGRAM),
            cwd=candidate,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SmokeResult(status="failed", error=f"smoke failed: {exc}", artifacts={})
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(_MARKER):
            payload = line.removeprefix(_MARKER)
    if completed.returncode != 0 or payload is None:
        diagnostic = " ".join(
            (completed.stderr or completed.stdout or "missing smoke result").split()
        )[-4000:]
        return SmokeResult(
            status="failed",
            error=f"candidate smoke rejected: {diagnostic}",
            artifacts={"returncode": completed.returncode},
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        return SmokeResult(
            status="failed",
            error=f"candidate smoke result is invalid JSON: {exc}",
            artifacts={"returncode": completed.returncode},
        )
    return SmokeResult(
        status="complete",
        error=None,
        artifacts={
            "roles": ["P0", "P1"],
            "accepted_operation_counts": value["counts"],
            "policy_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "returncode": completed.returncode,
        },
    )

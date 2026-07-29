"""Round-counter state for HL auto-naming.

The HL loop assigns each round a unique name ``hl-v<YY-MM-DD>-round<n>``. The
round number ``n`` is **global monotonic** across runs and persists in
``<hl_root>/hl_state.json``. It cannot be derived from ``events.jsonl`` (that
file is truncated on every CLI run) and is not reliably derivable from scanning
dirs alone (deletions would create gaps). So: read the state file; if it is
missing, recover ``n`` from the highest ``hl-v*-round<N>/`` dir under ``hl_root``
before incrementing — so a deleted state file cannot reset the counter to a
colliding value.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentbench_frame.hl.naming import ROUND_NAME_RE

STATE_FILENAME = "hl_state.json"


def _state_path(hl_root) -> Path:
    return Path(hl_root) / STATE_FILENAME


def _read_last_round(hl_root) -> int | None:
    p = _state_path(hl_root)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    n = d.get("last_round")
    return n if isinstance(n, int) and n >= 0 else None


def _recover_max_round(hl_root) -> int:
    """Highest round number found among existing ``hl-v*-round<N>/`` dirs.

    Used only when the state file is absent, so deleting it does not reset the
    counter to 1 and collide with a prior round.
    """
    root = Path(hl_root)
    if not root.is_dir():
        return 0
    best = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = ROUND_NAME_RE.match(d.name)
        if m:
            best = max(best, int(m.group("n")))
    return best


def next_round(hl_root) -> int:
    """Atomically read, increment, and persist the round counter.

    Returns the new round number (1 on a fresh root). Creates ``hl_root`` and
    the state file if missing. Recovery: if no state file exists, seeds from
    the max existing round dir before incrementing.
    """
    root = Path(hl_root)
    root.mkdir(parents=True, exist_ok=True)
    current = _read_last_round(hl_root)
    if current is None:
        current = _recover_max_round(hl_root)
    n = current + 1
    _state_path(hl_root).write_text(
        json.dumps({"schema": 1, "last_round": n}), encoding="utf-8")
    return n

"""Pure parser of a ``run_match`` trace JSONL into a ``ReferenceStateSet``.

This module implements plan §2 of the ``recorded-reference-set`` change. It
is a PURE PARSER: it reads a match trace (the JSONL written by
``lostspace/match.py``) and emits ``ReferenceSample``s for every Alive
on-turn ``roundbegin`` decision point the candidate (at ``seat``) reached
during the recorded roll. It does NOT touch ``match.py`` / ``ladder.py`` /
``evaluator.py`` / ``probe.py`` / ``controller.py`` — the recorder is a
post-hoc transform on a saved trace.

Decision point (plan §2.1 / design D2/D3):
    A seat-N judger→AI frame whose ``content.type == "roundbegin"`` AND
    ``int(content.inturn) == 0`` AND ``int(content.status) == STATUS_ALIVE``
    (0). Frames with status in {DIED, ESCAPED, SKIP, ERROR, WAIT_FOR_ESCAPE}
    are NOT decision points (consistent with
    ``distribution.enumerate_legal_actions``).

Transcript prefix (plan §2.1):
    The ordered tuple of ALL seat-N ``type:"observation"`` frames from the
    first one (the ``id`` frame) through the decision-point ``roundbegin``
    (inclusive) — on-turn AND off-turn frames both go in (off-turn
    ``see``/``getkey``/``interprops_status_update`` notifications build the
    candidate's map; they must be replayed).

Sample fields (plan §2.1):
    ``observation`` = the full roundbegin content dict (the candidate reads
    top-level ``inturn/status/state/hp/keys/tools/others``).
    ``legal_actions`` = the logic-merged ``get_legal_actions()`` projection
    (``attack``/``move``/``detect``/``interprops``).
    ``inventory`` = derived from ``content.tools`` — traps count is
    ``made - used`` (``tools[name] = [made, used]``); ``Kit``/``Transport``
    are bare ints.
    ``status`` / ``seat`` / ``opponent`` come from the frame / call args.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agentbench_frame.hl.distribution import (
    STATUS_ALIVE,
    STATUS_DIED,
    STATUS_ESCAPED,
    STATUS_ERROR,
    STATUS_SKIP,
    STATUS_WAIT_FOR_ESCAPE,
)
from agentbench_frame.hl.reference import ReferenceSample, ReferenceStateSet

__all__ = ["record_reference_states", "main"]

_log = logging.getLogger(__name__)

# Statuses that are NOT decision points — mirrors enumerate_legal_actions.
_NON_DECISION_STATUSES = frozenset(
    {STATUS_DIED, STATUS_ESCAPED, STATUS_SKIP, STATUS_ERROR, STATUS_WAIT_FOR_ESCAPE}
)


def _is_decision_point(content: Dict[str, Any]) -> bool:
    """True iff ``content`` is a seat-N on-turn Alive roundbegin frame."""
    if not isinstance(content, dict):
        return False
    if content.get("type") != "roundbegin":
        return False
    try:
        inturn = int(content.get("inturn", -1))
        status = int(content.get("status", -1))
    except (TypeError, ValueError):
        return False
    return inturn == 0 and status == STATUS_ALIVE


def _derive_inventory(tools: Any) -> Dict[str, int]:
    """Project ``tools`` into the inventory dict the candidate's
    ``enumerate_legal_actions`` expects.

    The roundbegin ``tools`` field shape (see ``reference_seed.py:50-53``,
    ``player.py:227``):
        ``{"LandMine":[made,used], "Sticky":[made,used], "Kit":int,
        "Transport":int}``

    Traps count = ``made - used``; ``Kit`` / ``Transport`` are bare ints.
    Missing fields default to 0. Non-dict input → empty inventory.
    """
    inv: Dict[str, int] = {}
    if not isinstance(tools, dict):
        return inv
    for name in ("LandMine", "Sticky"):
        val = tools.get(name)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                made, used = int(val[0]), int(val[1])
                inv[name] = made - used
            except (TypeError, ValueError):
                inv[name] = 0
        elif isinstance(val, (int, float)):
            # Defensive: a flat int (already-counted) shape — use as-is.
            inv[name] = int(val)
        else:
            inv[name] = 0
    for name in ("Kit", "Transport"):
        val = tools.get(name)
        if isinstance(val, (int, float)):
            inv[name] = int(val)
        elif isinstance(val, (list, tuple)) and len(val) >= 1:
            try:
                inv[name] = int(val[0])
            except (TypeError, ValueError):
                inv[name] = 0
        else:
            inv[name] = 0
    return inv


def record_reference_states(
    trace_path: Path, *, spec_id: str, opponent: str, seat: int = 0,
) -> ReferenceStateSet:
    """Parse ``trace_path`` (JSONL) into a ``ReferenceStateSet``.

    Pure parser — no side effects beyond reading the file. Emits one
    ``ReferenceSample`` per Alive on-turn ``roundbegin`` decision point the
    seat reached. An empty trace (no such frames) yields an empty
    ``ReferenceStateSet`` (no raise): a stub reference policy that never got
    a turn is a legitimate, if useless, result.
    """
    # Collect, in file order, the seat's full judger→AI frame stream (each
    # reduced to its `content` dict). Entries of type "action" / "ai_error"
    # are skipped — the candidate regenerates its own actions; only judger→AI
    # frames are replayed.
    stream: List[Dict[str, Any]] = []
    raw_lines = Path(trace_path).read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            _log.warning("skipping non-JSON trace line: %r", line[:120])
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "observation":
            continue
        try:
            player = int(entry.get("player", -1))
        except (TypeError, ValueError):
            continue
        if player != int(seat):
            continue
        content = entry.get("content")
        if not isinstance(content, dict):
            # Malformed trace — skip defensively, never raise.
            _log.warning("skipping observation with non-dict content: %r",
                         entry)
            continue
        stream.append(content)

    samples: List[ReferenceSample] = []
    for k, content in enumerate(stream):
        if not _is_decision_point(content):
            continue
        # status re-check: decision point requires STATUS_ALIVE, which
        # _is_decision_point already verified. Skip the redundant check
        # against _NON_DECISION_STATUSES for clarity (Alive is the only
        # remaining case).
        transcript = tuple(stream[0:k + 1])
        legal_actions = {
            "attack": content.get("attack", []),
            "move": content.get("move", []),
            "detect": content.get("detect", False),
            "interprops": content.get("interprops", []),
        }
        inventory = _derive_inventory(content.get("tools", {}))
        try:
            status = int(content.get("status", 0))
        except (TypeError, ValueError):
            status = STATUS_ALIVE
        samples.append(ReferenceSample(
            observation=dict(content),
            legal_actions=legal_actions,
            inventory=inventory,
            status=status,
            seat=int(seat),
            opponent=opponent,
            transcript=transcript,
        ))

    if not samples:
        _log.warning(
            "no Alive on-turn roundbegin decision points for seat=%s in %s",
            seat, trace_path,
        )

    return ReferenceStateSet(spec_id=spec_id, samples=tuple(samples))


def main(argv: List[str] | None = None) -> int:
    """CLI: parse a trace JSONL into a saved ``ReferenceStateSet`` JSON.

    Usage::

        python -m agentbench_frame.hl.reference_recorder \
            --trace <trace.jsonl> --spec-id <id> --opponent <name> \
            [--seat 0] --out nu.json
    """
    parser = argparse.ArgumentParser(
        prog="agentbench_frame.hl.reference_recorder",
        description="Parse a LostSpace match trace JSONL into a recorded "
                    "reference state set (ν) for HL policy KL.",
    )
    parser.add_argument("--trace", type=Path, required=True,
                        help="Path to the match trace JSONL.")
    parser.add_argument("--spec-id", required=True,
                        help="Benchmark spec_id to pin the reference set to.")
    parser.add_argument("--opponent", required=True,
                        help="Opponent name recorded for each sample.")
    parser.add_argument("--seat", type=int, default=0,
                        help="Seat index whose decision points to capture "
                             "(default 0).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the saved ReferenceStateSet JSON.")
    args = parser.parse_args(argv)

    rss = record_reference_states(
        args.trace, spec_id=args.spec_id, opponent=args.opponent,
        seat=args.seat,
    )
    rss.save(args.out)
    print(f"wrote {len(rss.samples)} reference samples to {args.out} "
          f"(spec_id={rss.spec_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

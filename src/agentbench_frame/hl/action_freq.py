"""Per-version action-frequency statistics + KL between consecutive versions.

The user-facing measurement channel (in addition to the reference-probe
first-primitive KL): after every evaluation, aggregate the evaluated version's
**full** real-match action stream — every primitive it emitted across all its
matches — into a frequency distribution over a canonical action vocabulary, and
report the KL between consecutive versions' distributions.

Why this channel exists (and why the first-primitive probe can't replace it):
the coding agent's ``play()`` top branch stabilizes after one strategy flip
(``use_tool("Kit")`` → ``interact("KeyMachine")`` → ...) and the winning
improvements land in **mid-game** logic (BFS key-targeting weights, escape
timing, box-loot order). Those edits leave the first emitted primitive
identical, so first-primitive KL reads 0 even on a freshly recorded ν. The
action-frequency distribution sees the mid-game changes directly: a reweighted
targeting heuristic shifts the ``move``/``interact``/``tool`` frequency mix.

Canonicalization: coordinate-parameterized actions (``move [x,y,z]``,
``detect [x,y,z]``, ``attack [x,y,z] id``) collapse to their macro token, so
the vocabulary stays small and the distribution meaningful; prop/tool arguments
are kept (``("interact", "KeyMachine")`` vs ``("interact", "Box")`` vs
``("tool", "Kit")``).

KL uses the same epsilon channel as the reference-probe measurement:
``(1-eps) * empirical_freq + eps * U(vocab)``, so a one-sided action flip
(e.g. the old version never took ``("tool", "Kit")``) stays finite.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

#: Action names whose parameters are positional/coordinate-dependent and carry
#: no decision information — collapse to the bare macro token.
_MACRO_ONLY = frozenset({"move", "detect", "attack"})
#: Action names that carry a prop/tool/trap name argument worth keeping.
_ARG_ACTIONS = frozenset({"interact", "tool", "use_tool", "trap", "use_trap"})


def canonical_action(content: Any) -> Tuple[Any, ...]:
    """Canonical token for one AI→judger action frame (``trace`` content).

    ``content`` is the wire frame the harness records for an action
    (``match.py``): ``{"type": "action", "action": ["interact", "Box"]}`` or
    ``{"type": "finish"}``. Returns a stable, hashable token; malformed input
    yields a distinguishable ``("<malformed>",)``-style token (never a crash —
    the frequency table is best-effort).
    """
    if not isinstance(content, dict):
        return ("<malformed>",)
    if content.get("type") == "finish":
        return ("finish",)
    prim = content.get("action")
    if not isinstance(prim, (list, tuple)) or not prim:
        return ("<empty>",)
    name = prim[0]
    if name in _MACRO_ONLY:
        return (str(name),)
    if name in _ARG_ACTIONS:
        # use_tool/use_trap are the same actions under a different wire name
        norm = {"use_tool": "tool", "use_trap": "trap"}.get(name, name)
        if norm == "interact":
            # keep the prop + any argument (Materials→tool, EscapeCapsule→flag)
            args = tuple(str(x) for x in prim[1:3])
            return (str(norm), *args) if args else (str(norm),)
        arg = prim[1] if len(prim) > 1 else None
        return (str(norm), str(arg))
    # fallback: freeze the non-coordinate parts of an unknown action name
    return tuple(str(x) for x in prim if not isinstance(x, (list, tuple)))


def count_actions(trace_path: Path, *, seat: int = 0) -> Counter:
    """Count the seat's canonicalized actions in one ``.trace.jsonl``.

    Pure parser of the same trace format ``reference_recorder`` consumes
    (``match.py`` frame stream). Entries of type ``"action"`` for ``seat`` are
    counted; everything else (observations, ai_error) is ignored. A malformed
    or missing file yields an empty counter (best-effort, never raises).
    """
    c: Counter = Counter()
    try:
        lines = Path(trace_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return c
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(e, dict) or e.get("type") != "action":
            continue
        try:
            player = int(e.get("player", -1))
        except (TypeError, ValueError):
            continue
        if player != seat:
            continue
        c[canonical_action(e.get("content"))] += 1
    return c


def action_freq_kl(counter_new: Counter, counter_old: Counter, *,
                   epsilon: float = 0.1) -> float:
    """KL(p_new || p_old) over the union action vocabulary, epsilon-smoothed.

    p(t) = (1-eps) * count(t)/total + eps * (1/|vocab|), same channel as the
    reference-probe measurement. The smoothing keeps KL finite when the two
    versions' supports differ (a one-sided action flip), and KL=0 iff the two
    empirical distributions are identical. An empty vocabulary (no actions
    measured) yields 0.0.
    """
    vocab = sorted(set(counter_new) | set(counter_old))
    if not vocab:
        return 0.0
    n = len(vocab)
    total_new = sum(counter_new.values()) or 1
    total_old = sum(counter_old.values()) or 1
    u = epsilon / n

    def dist(c: Counter, total: int) -> Dict[Any, float]:
        return {t: (1.0 - epsilon) * (c.get(t, 0) / total) + u for t in vocab}

    p = dist(counter_new, total_new)
    q = dist(counter_old, total_old)
    return sum(pv * math.log(pv / q[t]) for t, pv in p.items() if pv > 0)

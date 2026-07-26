"""
Canonical action space + measurement channel for HL policy KL.

Grounded in ``Player.get_legal_actions()`` (``player.py:230-262``) and
``GameController.solve()`` (``GameController.py:185-255``); see
``lostspace/GAME_RULES.md``.

Design decisions (see plan.md §distribution):
- **Granularity = primitive action.** One decision point = one wire primitive
  the agent emits (``move``/``attack``/``interact``/``trap``/``tool``/
  ``detect``/``finish``). A small round is a *sequence* of primitives ending
  in ``finish``. Primitive level keeps ``A(s)`` finite and enumerable so the
  epsilon-smoothed uniform is meaningful; macro level would be combinatorial
  and ~empty.
- **A(s)** is enumerated from ``get_legal_actions()`` output + inventory +
  player status. ``finish`` is always legal for a live player.
- **Measurement channel**: if the agent exposes ``get_action_distribution``,
  use it directly; else (deterministic HL) the controller observes the emitted
  primitive ``a*`` and uses ``(1-eps)*onehot(a*) + eps*U(A(s))`` — the *same*
  channel for RL and HL. ``eps`` keeps KL finite (q never exactly 0 on A(s)).
- **Documented projection**: ``tool(Transport)`` targets are enumerated over
  the edge-neighbor set (same as ``move``); this is a v1 projection, not the
  full Transport target space. ``detect`` targets are exactly the edge
  neighbors (correct, ``player.py:208``).
- **Documented limitation**: primitive-level A(s) does not capture
  inter-primitive correlations within a round. Two agents with identical
  per-primitive distributions but different round sequencing show KL≈0.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---- constants (mirror config.py; source-commented so drift is visible) ----

# config.py:3
TOOL_LIST: Tuple[str, ...] = ("LandMine", "Sticky", "Transport", "Kit")
# config.py:4
TRAP_LIST: Tuple[str, ...] = ("LandMine", "Sticky")
# config.py:2
PROP_LIST: Tuple[str, ...] = ("Materials", "EscapeCapsule", "KeyMachine", "Box")
# config.py:29 — 8 movement directions, first 4 are cardinals
DIRECTION_SEQ: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
    (1, -1, 0), (-1, 1, 0), (1, 1, 0), (-1, -1, 0),
)

# PlayerStatus (config.py:37-43)
STATUS_ALIVE = 0
STATUS_DIED = 1
STATUS_ESCAPED = 2
STATUS_SKIP = 3
STATUS_WAIT_FOR_ESCAPE = 4
STATUS_ERROR = 5

#: The canonical "end turn" primitive, always legal for a live player.
FINISH: Tuple[str, ...] = ("finish",)

ActionToken = Tuple[Any, ...]


@dataclass(frozen=True)
class LegalActionSet:
    """The complete normalized legal action set A(s) at one decision point.

    ``tokens`` is a canonical, sorted tuple of action tokens. ``state_id`` is
    a stable hash of the token set, used for occupancy aggregation. An empty
    ``tokens`` means there is no decision point at this state (e.g. the
    player is dead/skipped) — the trace records no entry for that round.
    """
    tokens: Tuple[ActionToken, ...]
    state_id: str = ""

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __contains__(self, token: ActionToken) -> bool:
        return token in self.tokens


def _canonical(token: ActionToken) -> ActionToken:
    """Freeze a token for hashing/sets."""
    return tuple(token)


def _state_id(tokens: Tuple[ActionToken, ...]) -> str:
    h = hashlib.sha1(json.dumps([list(t) for t in tokens]).encode()).hexdigest()
    return h[:16]


def enumerate_legal_actions(
    legal: Dict[str, Any],
    *,
    status: int,
    inventory: Dict[str, int],
) -> LegalActionSet:
    """Build A(s) from a ``get_legal_actions()``-style dict + player state.

    Args:
        legal: dict with keys ``attack`` (list of player ids), ``move`` (len-8
            bool list), ``detect`` (bool), ``interprops`` (list of prop names).
            For non-Alive statuses this may be empty.
        status: ``PlayerStatus`` value (0..5).
        inventory: carry counts keyed by tool name
            (``LandMine``/``Sticky``/``Transport``/``Kit``).

    Returns:
        A ``LegalActionSet``. Empty iff there is no decision point at this
        state (status Died/Escaped/Skip/Error).
    """
    # No decision point: round auto-ends, the agent is not queried.
    if status in (STATUS_DIED, STATUS_ESCAPED, STATUS_SKIP, STATUS_ERROR):
        return LegalActionSet(tokens=(), state_id="")

    tokens: List[ActionToken] = [FINISH]

    if status == STATUS_WAIT_FOR_ESCAPE:
        # solve() (GameController.py:190-199) only allows escape-capsule
        # interact in this state, plus finish.
        tokens.append(_canonical(("interact", "EscapeCapsule")))
        tokens = sorted(set(tokens))
        return LegalActionSet(tokens=tuple(tokens), state_id=_state_id(tuple(tokens)))

    # status == Alive
    move_mask = legal.get("move") or [False] * 8
    for d, ok in enumerate(move_mask):
        if ok:
            tokens.append(_canonical(("move", d)))

    for tid in legal.get("attack", []) or []:
        tokens.append(_canonical(("attack", int(tid))))

    interprops = set(legal.get("interprops", []) or [])
    if "Box" in interprops:
        tokens.append(_canonical(("interact", "Box")))
    if "Materials" in interprops:
        for tool in TOOL_LIST:
            tokens.append(_canonical(("interact", "Materials", tool)))
    if "KeyMachine" in interprops:
        tokens.append(_canonical(("interact", "KeyMachine")))
    if "EscapeCapsule" in interprops:
        tokens.append(_canonical(("interact", "EscapeCapsule")))

    inv = {k: int(v or 0) for k, v in inventory.items()}
    for trap in TRAP_LIST:
        if inv.get(trap, 0) > 0:
            tokens.append(_canonical(("trap", trap)))
    if inv.get("Kit", 0) > 0:
        tokens.append(_canonical(("tool", "Kit")))
    # v1 projection: Transport targets = edge-neighbors (the move set).
    if inv.get("Transport", 0) > 0:
        for d, ok in enumerate(move_mask):
            if ok:
                # neighbor tile = pos + direction; canonicalized as a token
                # param. The controller fills actual coords; here we key by
                # direction index so the token is stable across versions.
                tokens.append(_canonical(("tool", "Transport", ("dir", d))))

    if legal.get("detect"):
        for d, ok in enumerate(move_mask):
            if ok:
                tokens.append(_canonical(("detect", ("dir", d))))

    tokens = sorted(set(tokens))
    return LegalActionSet(tokens=tuple(tokens), state_id=_state_id(tuple(tokens)))


def epsilon_smoothed_distribution(
    chosen: Optional[ActionToken],
    legal: LegalActionSet,
    epsilon: float,
    *,
    return_flag: bool = False,
):
    """The shared measurement channel: ``(1-eps)*onehot(chosen) + eps*U(A(s))``.

    Same channel for RL and HL. If ``chosen`` is None (no distribution and no
    observed action — e.g. an agent that declined), returns uniform U(A(s)).

    If ``chosen`` is not in A(s) (an illegal / out-of-support emission), the
    measurement distribution collapses to uniform U(A(s)) and the
    ``out_of_support`` flag is set — this is a measurement anomaly the
    controller records (it does not coerce the action into A(s)).

    Args:
        chosen: the action the agent emitted (or None).
        legal: A(s) for this decision point. Must be non-empty.
        epsilon: smoothing mass in [0, 1].
        return_flag: if True, also return the out_of_support flag.

    Returns:
        ``dict[token -> prob]``, or ``(dict, bool)`` if ``return_flag``.
    """
    n = len(legal)
    if n == 0:
        dist: Dict[ActionToken, float] = {}
        return (dist, False) if return_flag else dist

    uniform = 1.0 / n
    dist = {t: uniform for t in legal.tokens}

    out_of_support = False
    if chosen is not None:
        chosen = _canonical(chosen)
        if chosen in dist:
            dist[chosen] += (1.0 - epsilon)
            # rescale others? No: the epsilon channel puts eps on the rest
            # uniformly already; the chosen gets the (1-eps) bonus. Total:
            # chosen = eps/n + (1-eps); others = eps/n. Sums to 1 only if
            # the uniform base was eps-weighted, not 1-weighted. Fix below.
            # --- correct formula below (recompute cleanly) ---
        else:
            out_of_support = True

    if out_of_support or chosen is None:
        # uniform U(A(s)) — no (1-eps) mass placed.
        u = 1.0 / n
        dist = {t: u for t in legal.tokens}
    else:
        u = epsilon / n
        dist = {t: u for t in legal.tokens}
        dist[chosen] += (1.0 - epsilon)

    return (dist, out_of_support) if return_flag else dist


def policy_kl(p: Dict[ActionToken, float], q: Dict[ActionToken, float]) -> float:
    """KL(p || q) in nats, over p's support.

    Per the measurement contract (doc §4): the newer policy is p (π_k),
    the older is q (π_{k-1}). With epsilon smoothing, q > 0 on all of A(s),
    so KL is finite. Returns +inf if p puts mass where q is exactly 0.
    """
    kl = 0.0
    for a, pi in p.items():
        if pi <= 0:
            continue
        qi = q.get(a, 0.0)
        if qi <= 0:
            return float("inf")
        kl += pi * math.log(pi / qi)
    return kl


def local_policy_kl_trace(
    chosen_new: Sequence[Optional[ActionToken]],
    chosen_old: Sequence[Optional[ActionToken]],
    legal_sets: Sequence[LegalActionSet],
    epsilon: float,
) -> List[float]:
    """Per-decision policy KL between two versions, over their shared decision
    points.

    Only decision points that actually occurred (non-empty A(s)) contribute an
    entry. If the two versions' traces have different lengths or decision-point
    structure, the shorter common prefix is used and the mismatch is recorded by
    the caller (occupancy_shift captures the rest). Here we zip elementwise.

    Returns the raw trace ``[KL_0, KL_1, ...]`` (doc §15: store raw traces,
    never pre-aggregated means).
    """
    trace: List[float] = []
    for a_new, a_old, las in zip(chosen_new, chosen_old, legal_sets):
        if len(las) == 0:
            continue  # no decision point -> no trace entry
        p = epsilon_smoothed_distribution(a_new, las, epsilon)
        q = epsilon_smoothed_distribution(a_old, las, epsilon)
        trace.append(policy_kl(p, q))
    return trace

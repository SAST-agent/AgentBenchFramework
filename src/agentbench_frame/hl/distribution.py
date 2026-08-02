"""
Canonical action space + measurement channel for HL policy KL.

Grounded in ``Player.get_legal_actions()`` (``player.py:230-262``) and
``GameController.solve()`` (``GameController.py:185-255``); see
``lostspace/GAME_RULES.md`` and the Q3 contract in ``decision_space.py``
(macro set / SUPPORT / termination / observation / mask live there — this
module is the primitive-level measurement channel that consumes it).

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

# Re-export the Q3 contract symbols from decision_space.py (single source).
# Existing callers import these from ``distribution`` — keep that working.
from agentbench_frame.hl.decision_space import (  # noqa: F401  (re-export)
    STATUS_ALIVE,
    STATUS_DIED,
    STATUS_ESCAPED,
    STATUS_ERROR,
    STATUS_SKIP,
    STATUS_WAIT_FOR_ESCAPE,
    SUPPORT,
    MACRO_ACTIONS,
    ActionMask,
    Termination,
    compute_mask,
    macro_id_of,
    termination,
)

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
    """Freeze a token for hashing/sets.

    Recursively freezes nested lists into tuples, so a token like
    ``("attack", [1, 2])`` becomes ``("attack", (1, 2))`` and is hashable.
    The coding agent's emitted actions can carry list-valued fields
    (e.g. attack target ids), and without this the policy-KL measurement
    crashes with ``TypeError: unhashable type: 'list'`` when it tries
    ``chosen in dist``.
    """
    return _freeze(token)


def _freeze(obj):
    """Recursively freeze lists/tuples of lists into tuples of tuples."""
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


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
    if termination(status).is_terminal:
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


@dataclass(frozen=True)
class PolicyKLPoint:
    """One per-sample policy-KL measurement, with its honesty status.

    ``kl`` is only populated when ``status == "ok"`` (both versions emitted an
    in-support primitive). Missing/unresponsive emissions and out-of-support
    emissions are recorded explicitly with ``kl=None`` and a ``reason`` — never
    silently folded into a false 0.0 (doc §4 honesty: missing stays missing).
    """

    kl: Optional[float]
    status: str  # "ok" | "no_emission" | "out_of_support"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"kl": self.kl, "status": self.status, "reason": self.reason}


def ok_kl_values(trace: Sequence["PolicyKLPoint"]) -> List[float]:
    """The real (status == "ok") KL values in a trace.

    Missing/out-of-support points carry ``kl=None`` and are excluded — they must
    never enter a mean or a changed-count denominator.
    """
    return [p.kl for p in trace if p.status == "ok" and p.kl is not None]


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
) -> List[PolicyKLPoint]:
    """Per-decision policy KL between two versions, as a structured trace.

    Each entry carries ``{kl, status, reason}`` (doc §15: store raw traces,
    never pre-aggregated means). The statuses are the honesty contract:

    - ``ok``: both versions emitted an in-support primitive; ``kl`` is the real
      value.
    - ``no_emission``: a version returned ``None`` (unresponsive); ``kl=None``,
      ``reason="version_new_unresponsive"`` / ``version_old_unresponsive"``.
    - ``out_of_support``: an emitted primitive was not in A(s); ``kl=None``,
      ``reason="chosen_not_in_support(version_new|old)"``.

    Only decision points that actually occurred (non-empty A(s)) contribute an
    entry. A ``no_emission`` point is KEPT (not dropped) so ``n_missing`` stays
    visible; if the two versions' decision-point structure differs, the shorter
    common prefix is used and the mismatch is recorded by the caller
    (``occupancy_shift``). Here we zip elementwise.
    """
    points: List[PolicyKLPoint] = []
    for a_new, a_old, las in zip(chosen_new, chosen_old, legal_sets):
        if len(las) == 0:
            continue  # no decision point -> no trace entry
        if a_new is None or a_old is None:
            missing = "new" if a_new is None else "old"
            points.append(PolicyKLPoint(
                kl=None, status="no_emission",
                reason=f"version_{missing}_unresponsive",
            ))
            continue
        p, oos_new = epsilon_smoothed_distribution(
            a_new, las, epsilon, return_flag=True)
        q, oos_old = epsilon_smoothed_distribution(
            a_old, las, epsilon, return_flag=True)
        if oos_new or oos_old:
            which = "new" if oos_new else "old"
            points.append(PolicyKLPoint(
                kl=None, status="out_of_support",
                reason=f"chosen_not_in_support(version_{which})",
            ))
            continue
        points.append(PolicyKLPoint(kl=policy_kl(p, q), status="ok"))
    return points

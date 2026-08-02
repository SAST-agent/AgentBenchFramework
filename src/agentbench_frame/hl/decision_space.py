"""The Q3 contract for the LostSpace HL measurement.

Defines the observation, the legal macro-action set, the action mask, the
termination conditions, and the action-support set used for information-gain
computation (mirrors ``snakego/decision_space.py``'s opening declaration on
``origin/zhongkaiyu``).

Honest downgrade (doc Fix-B / diagnosis §1.Q3 gap 2): LostSpace's fine-grained
action space is combinatorial (move directions × attack targets × interact
props × tool targets × trap kinds). This file therefore declares the **macro
projection** — the top-level ``play()`` macro set (``move/attack/interact/
trap/tool/detect/finish``) — as the cross-version-stable ``SUPPORT``, and
explicitly does NOT claim it is the full primitive space. The primitive-level
``A(s)`` enumeration (``distribution.enumerate_legal_actions``) stays the
measurement channel; ``compute_mask`` here is the macro-level view derived from
the *same* source.

Single-source of status/termination: the ``STATUS_*`` constants and the
terminal-state split that used to live inline in ``distribution.py`` are
centralized here; ``distribution.py`` imports them back and re-exports for
backward compatibility (``probe.py`` / ``controller.py`` /
``reference_recorder.py`` / tests import them from ``distribution``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# ---- player status (mirror config.py:37-43; source-commented so drift is
# visible, same as the old copy in distribution.py) ----
STATUS_ALIVE = 0
STATUS_DIED = 1
STATUS_ESCAPED = 2
STATUS_SKIP = 3
STATUS_WAIT_FOR_ESCAPE = 4
STATUS_ERROR = 5

#: The canonical macro set — the top-level ``play()`` macros (matches the
#: primitive token ``[0]`` names in ``distribution.enumerate_legal_actions``).
#: **Macro projection, NOT the full primitive space**: per-macro parameters
#: (direction, attack target, Transport target, trap kind) are the
#: fine-grained space enumerated by ``enumerate_legal_actions``; this contract
#: only fixes the macro layer, which is small enough to compare across versions.
MACRO_ACTIONS: Dict[int, str] = {
    1: "move",
    2: "attack",
    3: "interact",
    4: "trap",
    5: "tool",
    6: "detect",
    7: "finish",
}

#: The action-support set for IG computation. Fixed ordered macro ids,
#: cross-version stable, and — by construction — the same source as the mask
#: keys in :func:`compute_mask`. ``SUPPORT = tuple(MACRO_ACTIONS)``.
#:
#: Per-iteration IG ≡ the mean of the primitive-level
#: ``local_policy_kl_trace`` ok-values over ν (the ``ig`` field on
#: ``policy_kl`` events) — a strict KL over this support set, never a
#: substitute metric. When no ok sample exists, KL is genuinely unavailable
#: and the event records ``kl_missing_reason`` instead of a fabricated 0.
SUPPORT: Tuple[int, ...] = tuple(MACRO_ACTIONS)


def macro_id_of(token) -> Optional[int]:
    """Map a primitive action token to its macro id (token ``[0]`` name).

    Returns ``None`` for an unknown macro name (a token outside the declared
    ``MACRO_ACTIONS`` set — an out-of-support emission).
    """
    if not isinstance(token, (tuple, list)) or not token:
        return None
    return _NAME_TO_ID.get(token[0])


_NAME_TO_ID: Dict[Any, int] = {name: mid for mid, name in MACRO_ACTIONS.items()}


@dataclass(frozen=True)
class Termination:
    """The terminal/restricted classification of a player state."""

    is_terminal: bool
    status: int
    reason: str


def termination(status: int) -> Termination:
    """Centralize the status 分流 that decides whether there is a decision point.

    Mirrors ``distribution.enumerate_legal_actions``: Died/Escaped/Skip/Error
    are terminal (no decision point, the round auto-ends), WaitForEscape is a
    restricted decision state (escape-capsule + finish only), Alive is live.
    """
    if status in (STATUS_DIED, STATUS_ESCAPED, STATUS_SKIP, STATUS_ERROR):
        reason = {STATUS_DIED: "died", STATUS_ESCAPED: "escaped",
                  STATUS_SKIP: "skipped", STATUS_ERROR: "error"}[status]
        return Termination(is_terminal=True, status=status, reason=reason)
    if status == STATUS_WAIT_FOR_ESCAPE:
        return Termination(
            is_terminal=False, status=status,
            reason="waiting_for_escape: restricted to escape-capsule + finish",
        )
    return Termination(is_terminal=False, status=STATUS_ALIVE, reason="alive")


@dataclass(frozen=True)
class ActionMask:
    """Macro-level legality over ``SUPPORT``, with a reason per macro.

    ``mask`` is keyed exactly by ``SUPPORT`` (same source); a macro is legal
    iff at least one of its primitives appears in the decision point's A(s).
    """

    mask: Dict[int, bool]
    reasons: Dict[int, str]

    def __post_init__(self):
        object.__setattr__(self, "mask", dict(self.mask))
        object.__setattr__(self, "reasons", dict(self.reasons))


def compute_mask(tokens: Sequence) -> ActionMask:
    """Build the macro-level ``ActionMask`` from a decision point's A(s) tokens.

    ``tokens`` is the canonical primitive token tuple produced by
    ``distribution.enumerate_legal_actions`` (``LegalActionSet.tokens``). A
    macro is legal iff any of its primitives is present — so the mask and
    ``SUPPORT`` derive from the *same* enumeration source.
    """
    present: Dict[int, bool] = {}
    for t in tokens:
        mid = macro_id_of(t)
        if mid is not None:
            present[mid] = True
    mask = {mid: present.get(mid, False) for mid in SUPPORT}
    reasons = {
        mid: (MACRO_ACTIONS[mid] if present.get(mid, False)
              else f"{MACRO_ACTIONS[mid]} not in A(s)")
        for mid in SUPPORT
    }
    return ActionMask(mask=mask, reasons=reasons)


@dataclass(frozen=True)
class Observation:
    """Declared observation contract: the frozen ``roundbegin`` frame fields.

    The runtime measurement keeps the raw ``roundbegin`` dict (see
    ``reference.RecordedReferenceSample.observation`` and
    ``reference_recorder.record_reference_states``); this dataclass *declares*
    the stable subset of fields that constitute the observation for the
    contract's purposes. ``from_roundbegin`` reads known keys with declared
    defaults and keeps the full raw content for anything the contract does not
    enumerate.
    """

    type: str
    inturn: int
    status: int
    state: Any
    hp: Any
    keys: Any
    tools: Any
    others: Any
    raw: Mapping[str, Any]

    @classmethod
    def from_roundbegin(cls, content: Mapping[str, Any]) -> "Observation":
        return cls(
            type=content.get("type", ""),
            inturn=int(content.get("inturn", 0) or 0),
            status=int(content.get("status", STATUS_ALIVE) or STATUS_ALIVE),
            state=content.get("state"),
            hp=content.get("hp"),
            keys=content.get("keys"),
            tools=content.get("tools"),
            others=content.get("others"),
            raw=content,
        )

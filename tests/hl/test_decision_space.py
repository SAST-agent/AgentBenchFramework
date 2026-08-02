"""Tests for hl/decision_space.py — the Q3 contract file.

Covers the doc's Fix-B: MACRO_ACTIONS / SUPPORT (same source as the mask),
termination centralization, macro projection, and the distribution.py
re-export compatibility (probe / controller / reference_recorder import
STATUS_* and enumerate_legal_actions from distribution).
"""
import pytest

from agentbench_frame.hl import decision_space as ds
from agentbench_frame.hl.decision_space import (
    MACRO_ACTIONS,
    SUPPORT,
    STATUS_ALIVE,
    STATUS_DIED,
    STATUS_ESCAPED,
    STATUS_ERROR,
    STATUS_SKIP,
    STATUS_WAIT_FOR_ESCAPE,
    ActionMask,
    Observation,
    Termination,
    compute_mask,
    macro_id_of,
    termination,
)


def test_support_is_macro_action_keys_same_source():
    """SUPPORT is the fixed ordered macro-id set, same source as MACRO_ACTIONS."""
    assert SUPPORT == tuple(MACRO_ACTIONS)
    assert SUPPORT == tuple(sorted(MACRO_ACTIONS))
    assert len(SUPPORT) >= 7  # move/attack/interact/trap/tool/detect/finish


def test_macro_actions_declares_the_macro_projection():
    """The contract names the top-level play() macros, not the fine-grained
    primitive space (params like dir/attack-id are enumerated elsewhere)."""
    assert MACRO_ACTIONS[1] == "move"
    assert MACRO_ACTIONS[7] == "finish"
    assert set(MACRO_ACTIONS.values()) >= {
        "move", "attack", "interact", "trap", "tool", "detect", "finish",
    }


def test_macro_id_of_round_trips_all_macros():
    for mid, name in MACRO_ACTIONS.items():
        assert macro_id_of((name,)) == mid
        # a parameterized primitive still maps to its macro
        assert macro_id_of((name, ("dir", 3))) == mid
    assert macro_id_of(("teleport",)) is None
    assert macro_id_of(()) is None


def test_compute_mask_same_source_as_support():
    """A macro is legal iff at least one of its tokens is present in A(s)."""
    tokens = (
        ("finish",),
        ("move", 1),
        ("attack", 2),
        ("interact", "Box"),
        ("trap", "LandMine"),
        ("tool", "Kit"),
        ("detect", ("dir", 0)),
    )
    mask = compute_mask(tokens)
    assert isinstance(mask, ActionMask)
    for mid in SUPPORT:
        want = any(macro_id_of(t) == mid for t in tokens)
        assert mask.mask[mid] is want, f"macro {mid} ({MACRO_ACTIONS[mid]})"
    # macro 7 (finish) legal; every mask key is a SUPPORT member
    assert set(mask.mask) == set(SUPPORT)


def test_compute_mask_empty_tokens():
    mask = compute_mask(())
    assert set(mask.mask) == set(SUPPORT)
    assert not any(mask.mask.values())


def test_termination_covers_all_statuses():
    for s, want_terminal in (
        (STATUS_DIED, True),
        (STATUS_ESCAPED, True),
        (STATUS_SKIP, True),
        (STATUS_ERROR, True),
    ):
        t = termination(s)
        assert isinstance(t, Termination)
        assert t.is_terminal is want_terminal, f"status {s}"
        assert t.reason
    t = termination(STATUS_ALIVE)
    assert not t.is_terminal
    t = termination(STATUS_WAIT_FOR_ESCAPE)
    assert not t.is_terminal
    assert "restrict" in t.reason  # escape-capsule-only decision state


def test_observation_from_roundbegin_contract():
    content = {
        "type": "roundbegin", "inturn": 0, "status": 0, "state": "Alive",
        "hp": 10, "keys": 0, "tools": {"Kit": 1}, "others": [],
        "move": [True] * 8,
    }
    obs = Observation.from_roundbegin(content)
    assert obs.status == STATUS_ALIVE
    assert obs.raw is content  # runtime keeps the full dict
    # missing fields degrade to declared defaults, not crashes
    obs2 = Observation.from_roundbegin({"type": "roundbegin"})
    assert obs2.status == STATUS_ALIVE


def test_distribution_reexports_status_and_enumerator():
    """distribution.py keeps re-exporting the moved symbols so probe /
    controller / reference_recorder / tests import unchanged."""
    from agentbench_frame.hl import distribution
    assert distribution.STATUS_ALIVE is STATUS_ALIVE
    assert distribution.STATUS_WAIT_FOR_ESCAPE is STATUS_WAIT_FOR_ESCAPE
    assert distribution.SUPPORT == SUPPORT
    assert distribution.enumerate_legal_actions is not None
    # the terminal branch now delegates to the contract
    from agentbench_frame.hl.distribution import enumerate_legal_actions
    las = enumerate_legal_actions(
        {"attack": [], "move": [False] * 8, "detect": False, "interprops": []},
        status=STATUS_DIED, inventory={},
    )
    assert len(las) == 0  # dead player: no decision point

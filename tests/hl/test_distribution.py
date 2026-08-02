"""Tests for hl/distribution.py — the canonical action space + measurement channel.

Grounded in Player.get_legal_actions() (player.py:230-262) and
GameController.solve() (GameController.py:185-255). See lostspace/GAME_RULES.md.
"""
import math

import pytest

from agentbench_frame.hl.distribution import (
    enumerate_legal_actions,
    epsilon_smoothed_distribution,
    policy_kl,
    local_policy_kl_trace,
    PolicyKLPoint,
    ok_kl_values,
    LegalActionSet,
    FINISH,
)


def _legal(attack=(), move=(), detect=False, interprops=()):
    return {
        "attack": list(attack),
        "move": [i in move for i in range(8)],
        "detect": detect,
        "interprops": list(interprops),
    }


def test_finish_always_present_for_alive():
    las = enumerate_legal_actions(
        _legal(attack=[1], move=[2], interprops=["Box"]),
        status=0,  # Alive
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
    )
    assert isinstance(las, LegalActionSet)
    assert FINISH in las.tokens
    assert ("move", 2) in las.tokens
    assert ("attack", 1) in las.tokens
    assert ("interact", "Box") in las.tokens


def test_move_mask_respected():
    las = enumerate_legal_actions(
        _legal(move=[0, 7]),
        status=0, inventory={},
    )
    move_dirs = [t[1] for t in las.tokens if t[0] == "move"]
    assert sorted(move_dirs) == [0, 7]


def test_trap_and_tool_require_inventory():
    las_empty = enumerate_legal_actions(
        _legal(), status=0,
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
    )
    assert not any(t[0] == "trap" for t in las_empty.tokens)
    assert not any(t[0] == "tool" for t in las_empty.tokens)

    las_full = enumerate_legal_actions(
        _legal(), status=0,
        inventory={"LandMine": 2, "Sticky": 0, "Transport": 1, "Kit": 1},
    )
    assert ("trap", "LandMine") in las_full.tokens
    assert ("trap", "Sticky") not in las_full.tokens
    assert ("tool", "Kit") in las_full.tokens
    # Transport targets enumerated over edge-neighbors (the move set, which is
    # empty here) -> no transport targets when no neighbors reachable.
    assert not any(t[0] == "tool" and t[1] == "Transport" for t in las_full.tokens)


def test_materials_enumerates_all_tools():
    las = enumerate_legal_actions(
        _legal(interprops=["Materials"]), status=0, inventory={},
    )
    mats = [t for t in las.tokens if t[0] == "interact" and t[1] == "Materials"]
    assert {t[2] for t in mats} == {"LandMine", "Sticky", "Transport", "Kit"}


def test_detect_targets_are_edge_neighbors():
    las = enumerate_legal_actions(
        _legal(move=[2, 3], detect=True), status=0, inventory={},
    )
    detect_targets = [t for t in las.tokens if t[0] == "detect"]
    # detect targets mirror the move-reachable neighbor set
    assert len(detect_targets) == 2


def test_wait_for_escape_only_escape_abort_and_finish():
    las = enumerate_legal_actions(
        _legal(), status=4,  # WaitForEscape
        inventory={},
    )
    # While waiting only the ABORT variant is legal (interactive_props.py:113)
    assert set(las.tokens) == {("interact", "EscapeCapsule", False), FINISH}


def test_alive_escape_capsule_only_start_variant():
    """Alive + capsule on tile: only the START (True) variant is legal.
    A False (abort) emission here must be out_of_support — that is the exact
    bug the rules doc caused (REPLAY_SKILL said False starts the escape)."""
    las = enumerate_legal_actions(
        _legal(interprops=["EscapeCapsule"]), status=0, inventory={},
    )
    assert ("interact", "EscapeCapsule", True) in las.tokens
    assert ("interact", "EscapeCapsule", False) not in las.tokens
    assert ("interact", "EscapeCapsule") not in las.tokens  # bare never emitted


def test_dead_status_yields_empty_no_decision_point():
    for status in (1, 2, 3, 5):  # Died, Escaped, Skip, Error
        las = enumerate_legal_actions(_legal(), status=status, inventory={})
        assert len(las) == 0  # no decision point -> no trace entry


def test_tokens_are_canonical_and_sorted():
    las = enumerate_legal_actions(
        _legal(attack=[1, 0], move=[2], interprops=["Box", "KeyMachine"]),
        status=0, inventory={},
    )
    assert las.tokens == tuple(sorted(las.tokens))
    # canonical state id is stable
    assert las.state_id == las.state_id


def test_epsilon_distribution_sums_to_one():
    las = enumerate_legal_actions(
        _legal(attack=[1], move=[2]), status=0, inventory={},
    )
    dist = epsilon_smoothed_distribution(
        chosen=("move", 2), legal=las, epsilon=0.1,
    )
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
    n = len(las)
    # chosen gets (1-eps) + eps/n; others get eps/n
    assert math.isclose(dist[("move", 2)], 0.9 + 0.1 / n, abs_tol=1e-9)
    assert math.isclose(dist[("attack", 1)], 0.1 / n, abs_tol=1e-9)


def test_epsilon_distribution_chosen_none_is_uniform():
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    dist = epsilon_smoothed_distribution(chosen=None, legal=las, epsilon=0.2)
    n = len(las)
    for v in dist.values():
        assert math.isclose(v, 1.0 / n, abs_tol=1e-9)


def test_epsilon_distribution_out_of_support_flagged_uniform():
    """An emitted action outside A(s) is an anomaly -> uniform + flag."""
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    dist, out_of_support = epsilon_smoothed_distribution(
        chosen=("move", 5), legal=las, epsilon=0.1, return_flag=True,
    )
    assert out_of_support is True
    n = len(las)
    for v in dist.values():
        assert math.isclose(v, 1.0 / n, abs_tol=1e-9)  # uniform


def test_epsilon_distribution_chosen_with_nested_list_is_hashable():
    """A coding agent may emit an action carrying a list-valued field
    (e.g. ``("attack", [1, 2])``). _canonical must recursively freeze it so
    ``chosen in dist`` doesn't raise TypeError: unhashable type: 'list'."""
    las = enumerate_legal_actions(_legal(attack=[1, 2]), status=0, inventory={})
    # chosen carries a nested list; must not crash, must be treated as
    # out_of_support (canonicalized form unlikely to match the token set).
    dist, out_of_support = epsilon_smoothed_distribution(
        chosen=("attack", [1, 2]), legal=las, epsilon=0.1, return_flag=True,
    )
    # No exception; distribution is well-formed and sums to ~1.
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9)





def test_policy_kl_identical_is_zero():
    las = enumerate_legal_actions(_legal(attack=[1], move=[2]), status=0, inventory={})
    p = epsilon_smoothed_distribution(("move", 2), las, 0.1)
    assert math.isclose(policy_kl(p, p), 0.0, abs_tol=1e-12)


def test_policy_kl_with_epsilon_is_finite():
    """Deterministic HL: same A(s), different chosen -> KL finite under epsilon."""
    las = enumerate_legal_actions(_legal(attack=[1], move=[2, 3]), status=0, inventory={})
    p = epsilon_smoothed_distribution(("move", 2), las, 0.1)  # v_k
    q = epsilon_smoothed_distribution(("move", 3), las, 0.1)  # v_{k-1}
    kl = policy_kl(p, q)
    assert kl > 0
    assert math.isfinite(kl)


def test_policy_kl_is_not_symmetric_in_general():
    """KL is not symmetric in general. A symmetric-permutation case (two
    onehot+epsilon dists over the same support) *is* symmetric by structure,
    so to demonstrate asymmetry we use supports of different size: the older
    version had a larger A(s) (more legal actions) than the newer."""
    las_small = enumerate_legal_actions(
        _legal(attack=[1], move=[2]), status=0, inventory={},
    )
    las_big = enumerate_legal_actions(
        _legal(attack=[1], move=[2, 3]), status=0, inventory={},
    )
    # newer version sees the small A(s), older saw the big A(s). The chosen
    # action is the same, but the smoothing denominators differ -> asymmetric.
    p = epsilon_smoothed_distribution(("move", 2), las_small, 0.1)
    q = epsilon_smoothed_distribution(("move", 2), las_big, 0.1)
    # supports are not identical -> KL is genuinely asymmetric
    assert not math.isclose(policy_kl(p, q), policy_kl(q, p), abs_tol=1e-9)


def test_local_policy_kl_trace_per_decision():
    """Two versions emit different primitives at 3 decision points; trace has 3 KLs."""
    las = enumerate_legal_actions(_legal(attack=[1], move=[2, 3]), status=0, inventory={})
    eps = 0.1
    # version k chose move2, move3, attack1 ; version k-1 chose move3, move3, move2
    chosen_new = [("move", 2), ("move", 3), ("attack", 1)]
    chosen_old = [("move", 3), ("move", 3), ("move", 2)]
    trace = local_policy_kl_trace(chosen_new, chosen_old, [las, las, las], eps)
    assert len(trace) == 3
    # point 1 differs -> KL > 0 ; point 2 identical -> KL = 0 ; point 3 differs -> KL > 0
    assert all(p.status == "ok" for p in trace)
    assert trace[0].kl > 0
    assert math.isclose(trace[1].kl, 0.0, abs_tol=1e-12)
    assert trace[2].kl > 0


def test_local_policy_kl_trace_handles_no_decision_point():
    """Empty A(s) at a round -> no entry in the trace (no decision point)."""
    empty = enumerate_legal_actions(_legal(), status=1, inventory={})  # Died
    assert len(empty) == 0
    # An empty round contributes nothing to the trace.
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    trace = local_policy_kl_trace(
        [("attack", 1)], [("attack", 1)], [las], 0.1,
    )
    assert trace == [PolicyKLPoint(kl=0.0, status="ok")]


def test_local_policy_kl_trace_no_emission_not_folded_to_zero():
    """Fix-A: a version that returned None (unresponsive) -> status
    ``no_emission`` with kl=None, NEVER a false 0.0. This is what lets
    events.jsonl distinguish "both chose the same action" from "both silent"."""
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    # both versions silent at the decision point -> no_emission, not 0.0
    trace = local_policy_kl_trace([None], [None], [las], 0.1)
    assert len(trace) == 1
    p = trace[0]
    assert p.status == "no_emission"
    assert p.kl is None
    assert p.reason == "version_new_unresponsive"
    # old silent, new emitted -> still no_emission (old is the unresponsive one)
    trace2 = local_policy_kl_trace([("attack", 1)], [None], [las], 0.1)
    assert trace2[0].status == "no_emission"
    assert trace2[0].kl is None
    assert trace2[0].reason == "version_old_unresponsive"


def test_local_policy_kl_trace_out_of_support_recorded():
    """Fix-A: an emission outside A(s) -> status ``out_of_support`` with
    kl=None and the offending version named, instead of a coerced uniform."""
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    # attack 99 is not a legal target id
    trace = local_policy_kl_trace([("attack", 99)], [("attack", 1)], [las], 0.1)
    assert len(trace) == 1
    assert trace[0].status == "out_of_support"
    assert trace[0].kl is None
    assert trace[0].reason == "chosen_not_in_support(version_new)"


def test_ok_kl_values_excludes_missing():
    """Fix-A: ``ok_kl_values`` returns only real KL values; a no_emission
    point never enters a mean or a changed-count denominator."""
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    trace = local_policy_kl_trace(
        [("attack", 1), ("attack", 1), None],
        [("attack", 1), ("attack", 1), None],
        [las, las, las], 0.1,
    )
    ok = ok_kl_values(trace)
    assert len(ok) == 2  # first two ok; third no_emission excluded
    assert all(v == 0.0 for v in ok)


def test_normalize_emitted_maps_move_coordinate_to_direction():
    """A wire-format ('move', [x,y,z]) target that matches pos + DIRECTION_SEQ
    maps to the A(s) ('move', direction_index) token."""
    from agentbench_frame.hl.distribution import normalize_emitted
    # pos [0,0,1]; DIRECTION_SEQ[0] == (0,1,0) -> target [0,1,1]
    assert normalize_emitted(("move", [0, 1, 1]), pos=[0, 0, 1]) == ("move", 0)
    assert normalize_emitted(("detect", [0, 1, 1]), pos=[0, 0, 1]) == \
        ("detect", ("dir", 0))
    # unmatchable target stays verbatim (will be out-of-support, honestly)
    assert normalize_emitted(("move", [5, 5, 5]), pos=[0, 0, 1]) == \
        ("move", [5, 5, 5])


def test_normalize_emitted_attack_drops_coordinate():
    from agentbench_frame.hl.distribution import normalize_emitted
    assert normalize_emitted(("attack", [1, 1, 1], 2), pos=[0, 0, 1]) == \
        ("attack", 2)
    # parameterless primitives are unchanged
    assert normalize_emitted(("interact", "Box"), pos=[0, 0, 1]) == \
        ("interact", "Box")
    assert normalize_emitted(None, pos=[0, 0, 1]) is None


def test_normalize_emitted_interact_token_shapes():
    from agentbench_frame.hl.distribution import normalize_emitted
    # Escape-capsule flag is SEMANTIC and kept: True=start, False=abort. The
    # seed client sends the capsule arg as int (1/0); map to bool so a true
    # start still matches A(s), while a False-abort stays distinct.
    assert normalize_emitted(("interact", "EscapeCapsule", True),
                             pos=[0, 0, 1]) == ("interact", "EscapeCapsule", True)
    assert normalize_emitted(("interact", "EscapeCapsule", False),
                             pos=[0, 0, 1]) == ("interact", "EscapeCapsule", False)
    assert normalize_emitted(("interact", "EscapeCapsule", 1),
                             pos=[0, 0, 1]) == ("interact", "EscapeCapsule", True)
    assert normalize_emitted(("interact", "EscapeCapsule", 0),
                             pos=[0, 0, 1]) == ("interact", "EscapeCapsule", False)
    # Box-first sends a tool arg ("Key"); A(s) has bare Box.
    assert normalize_emitted(("interact", "Box", "Key"),
                             pos=[0, 0, 1]) == ("interact", "Box")
    # Materials keeps the tool arg (A(s) has ("interact","Materials","Kit")).
    assert normalize_emitted(("interact", "Materials", "Kit"),
                             pos=[0, 0, 1]) == ("interact", "Materials", "Kit")
    # 2-arg interacts pass through unchanged.
    assert normalize_emitted(("interact", "KeyMachine"),
                             pos=[0, 0, 1]) == ("interact", "KeyMachine")


def test_escape_false_while_alive_is_out_of_support():
    """A False (abort) emission while Alive must register as out_of_support in
    the measurement channel — never ratified as a valid in-support escape."""
    las = enumerate_legal_actions(
        _legal(interprops=["EscapeCapsule"]), status=0, inventory={},
    )
    dist, oos = epsilon_smoothed_distribution(
        chosen=("interact", "EscapeCapsule", False), legal=las, epsilon=0.1,
        return_flag=True)
    assert oos is True
    assert dist == {t: 1.0 / len(las) for t in las.tokens}
    # the START emission is in-support
    dist2, oos2 = epsilon_smoothed_distribution(
        chosen=("interact", "EscapeCapsule", True), legal=las, epsilon=0.1,
        return_flag=True)
    assert oos2 is False
    assert dist2[("interact", "EscapeCapsule", True)] > 0.9


def test_tracked_pos_from_transcript_uses_id_frame():
    from agentbench_frame.hl.distribution import tracked_pos_from_transcript
    # id birth_pos is 2D; the seeded candidate appends the z-layer (spawn).
    assert tracked_pos_from_transcript(
        [{"type": "id", "id": 0, "birth_pos": [0, 0]},
         {"type": "roundbegin", "pos": [3, 2, 1]}]) == [0, 0, 1]
    # 3D birth_pos passes through (first 3 elements).
    assert tracked_pos_from_transcript(
        [{"type": "id", "birth_pos": [1, 4, 0]}]) == [1, 4, 0]
    # no id frame -> None (caller falls back to the sample obs pos).
    assert tracked_pos_from_transcript([{"type": "roundbegin"}]) is None
    assert tracked_pos_from_transcript(None) is None

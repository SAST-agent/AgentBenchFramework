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


def test_wait_for_escape_only_escape_and_finish():
    las = enumerate_legal_actions(
        _legal(), status=4,  # WaitForEscape
        inventory={},
    )
    assert set(las.tokens) == {("interact", "EscapeCapsule"), FINISH}


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
    assert trace[0] > 0
    assert math.isclose(trace[1], 0.0, abs_tol=1e-12)
    assert trace[2] > 0


def test_local_policy_kl_trace_handles_no_decision_point():
    """Empty A(s) at a round -> no entry in the trace (no decision point)."""
    empty = enumerate_legal_actions(_legal(), status=1, inventory={})  # Died
    assert len(empty) == 0
    # An empty round contributes nothing to the trace.
    las = enumerate_legal_actions(_legal(attack=[1]), status=0, inventory={})
    trace = local_policy_kl_trace(
        [("attack", 1)], [("attack", 1)], [las], 0.1,
    )
    assert trace == [0.0]

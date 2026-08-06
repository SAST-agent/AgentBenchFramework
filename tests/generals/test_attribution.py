import pytest


def _complete_cells():
    return {
        "A": {"s1-p0": 0.0, "s2-p1": 1.0},
        "B": {"s1-p0": 1.0, "s2-p1": 1.0},
        "C": {"s1-p0": 0.0, "s2-p1": 0.0},
        "D": {"s1-p0": 1.0, "s2-p1": 0.0},
    }


def test_factorial_effects_use_paired_case_values():
    from agentbench_frame.generals.attribution import compute_paired_attribution

    report = compute_paired_attribution(
        {"win": _complete_cells()},
        bootstrap_seed=9049,
        bootstrap_replicates=100,
    )
    effect = report.metrics["win"]

    assert effect.large_stack == pytest.approx(0.5)
    assert effect.contact == pytest.approx(-0.5)
    assert effect.interaction == pytest.approx(0.0)
    assert effect.complete_pair_count == 2
    assert effect.case_effects["s1-p0"].large_stack == 1.0
    assert effect.case_effects["s2-p1"].contact == -1.0


def test_missing_case_stays_missing_instead_of_becoming_zero():
    from agentbench_frame.generals.attribution import compute_paired_attribution

    cells = _complete_cells()
    cells["B"]["s1-p0"] = None

    effect = compute_paired_attribution(
        {"army": cells},
        bootstrap_replicates=100,
    ).metrics["army"]

    assert effect.complete_pair_count == 1
    assert effect.case_effects["s1-p0"].large_stack is None
    assert effect.large_stack == 0.0


def test_paired_bootstrap_is_deterministic_and_local():
    from agentbench_frame.generals.attribution import compute_paired_attribution

    first = compute_paired_attribution(
        {"win": _complete_cells()},
        bootstrap_seed=9049,
        bootstrap_replicates=250,
    )
    second = compute_paired_attribution(
        {"win": _complete_cells()},
        bootstrap_seed=9049,
        bootstrap_replicates=250,
    )

    assert first == second
    assert first.metrics["win"].intervals["large_stack"] == (0.0, 1.0)


def test_factorial_effects_reject_mismatched_cells_or_pairs():
    from agentbench_frame.generals.attribution import compute_paired_attribution

    with pytest.raises(ValueError, match="cells A, B, C, and D"):
        compute_paired_attribution({"win": {"A": {}, "B": {}, "C": {}}})

    cells = _complete_cells()
    del cells["D"]["s2-p1"]
    with pytest.raises(ValueError, match="identical pair IDs"):
        compute_paired_attribution({"win": cells})


def test_factorial_effects_reject_bool_and_nonpositive_bootstrap():
    from agentbench_frame.generals.attribution import compute_paired_attribution

    cells = _complete_cells()
    cells["A"]["s1-p0"] = True
    with pytest.raises(ValueError, match="numeric or missing"):
        compute_paired_attribution({"win": cells})
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        compute_paired_attribution(
            {"win": _complete_cells()},
            bootstrap_replicates=0,
        )

import pytest

from agentbench_frame.games.rollman.measurement import (
    ROLLMAN_ACTION_SUPPORT,
    decision_trace_metrics,
)


def test_rollman_measurement_uses_exact_five_direction_support():
    assert ROLLMAN_ACTION_SUPPORT.action_ids == ("0", "1", "2", "3", "4")


def test_deterministic_trace_reports_finite_kl_and_occupancy_separately():
    old = [
        {"state_id": "s1", "action": 0},
        {"state_id": "s2", "action": 1},
    ]
    new = [
        {"state_id": "s1", "action": 4},
        {"state_id": "s2", "action": 1},
    ]

    result = decision_trace_metrics(new, old, epsilon=0.05)

    assert result["local_policy_kl_trace"][0] > 0
    assert result["local_policy_kl_trace"][1] == pytest.approx(0)
    assert result["mean_local_policy_kl"] > 0
    assert result["occupancy_shift"] == pytest.approx(0)


def test_trace_measurement_requires_the_same_explicit_reference_contexts():
    with pytest.raises(ValueError, match="reference contexts"):
        decision_trace_metrics(
            [{"state_id": "new-only", "action": 0}],
            [{"state_id": "old-only", "action": 0}],
            epsilon=0.05,
        )


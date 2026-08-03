import math


def _case(*steps, terminal_support=None):
    return {
        "state_id": "replay-a:0:P0",
        "steps": list(steps),
        "terminal_support": terminal_support or [[0, -1, -1], [11, 4, 5]],
    }


def _step(selected, support=None):
    return {
        "selected": selected,
        "support": support or [[0, -1, -1], [11, 4, 5]],
    }


def test_identical_deterministic_atomic_policies_have_zero_kl():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    probe = {"cases": [_case(_step([11, 4, 5]))]}

    result = compare_probe_outputs(probe, probe, epsilon=0.05)

    assert result.status == "complete"
    assert result.decision_count == 1
    assert result.changed_action_count == 0
    assert result.details["mean_kl_nats_per_decision"] == 0.0


def test_different_atomic_actions_have_positive_epsilon_smoothed_kl():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    parent = {"cases": [_case(_step([0, -1, -1]))]}
    candidate = {"cases": [_case(_step([11, 4, 5]))]}

    result = compare_probe_outputs(parent, candidate, epsilon=0.05)

    assert result.changed_action_count == 1
    assert result.decision_count == 1
    assert math.isfinite(result.details["mean_kl_nats_per_decision"])
    assert result.details["mean_kl_nats_per_decision"] > 0


def test_extra_bundle_atom_is_compared_against_hold_at_same_frozen_state():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    parent = {
        "cases": [
            _case(
                _step([11, 4, 5]),
                terminal_support=[[0, -1, -1], [31, -1, -1]],
            )
        ]
    }
    candidate = {
        "cases": [
            _case(
                _step([11, 4, 5]),
                _step([31, -1, -1], [[0, -1, -1], [31, -1, -1]]),
                terminal_support=[[0, -1, -1]],
            )
        ]
    }

    result = compare_probe_outputs(parent, candidate, epsilon=0.05)

    assert result.decision_count == 2
    assert result.changed_action_count == 1
    assert result.details["state_count"] == 1

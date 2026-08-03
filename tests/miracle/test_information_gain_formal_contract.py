import math
from types import SimpleNamespace

import pytest

from agentbench_frame.eval.measurement import ActionCandidate, ActionSupport
from agentbench_frame.games.miracle import decision_kl_v1 as decision_kl


MAIN_EPSILON = 0.01


def _observation():
    return {
        "round": 1,
        "camp": 0,
        "map": {"units": [], "barracks": [-1] * 4, "miracles": [30, 30]},
        "players": [[[], 20, 20, [], []], [[], 0, 0, [], []]],
    }


def _identity(support):
    return {
        "schema_version": support.schema_version,
        "support_id": support.support_id,
        "action_ids": list(support.action_ids),
    }


def _evidence(old, new, *, step=1):
    observation = _observation()
    support = decision_kl.build_trusted_action_support(observation)
    first, second = support.action_ids
    aliases = {"a": first, "b": second}
    return decision_kl.DecisionKLEvidence(
        decision_step=step,
        state_before=observation,
        support_identity=_identity(support),
        old_distribution={aliases[key]: value for key, value in old.items()},
        new_distribution={aliases[key]: value for key, value in new.items()},
    )


def _smoothed(values, epsilon=MAIN_EPSILON):
    uniform = 1.0 / len(values)
    return [(1.0 - epsilon) * value + epsilon * uniform for value in values]


def _new_old_kl(new, old, epsilon=MAIN_EPSILON):
    new_smoothed = _smoothed(new, epsilon)
    old_smoothed = _smoothed(old, epsilon)
    return math.fsum(
        new_probability * math.log(new_probability / old_probability)
        for new_probability, old_probability in zip(
            new_smoothed, old_smoothed, strict=True
        )
    )


def test_miracle_local_kl_uses_new_old_direction_and_fixed_symmetric_smoothing():
    old = {"a": 0.8, "b": 0.2}
    new = {"a": 0.55, "b": 0.45}
    summary = decision_kl.compute_trajectory_kl([_evidence(old, new)])

    expected = _new_old_kl([0.55, 0.45], [0.8, 0.2])
    reverse = _new_old_kl([0.8, 0.2], [0.55, 0.45])
    assert summary.status == "complete"
    assert summary.trajectory_kl == pytest.approx(expected)
    assert summary.trajectory_kl != pytest.approx(reverse)
    assert summary.direction == "new||old"
    assert summary.epsilon == MAIN_EPSILON
    assert summary.smoothing == "symmetric_epsilon_uniform_full_support"
    assert summary.log_base == "e"


def test_deterministic_hl_unchanged_is_zero_and_changed_is_finite_positive():
    unchanged = decision_kl.compute_trajectory_kl(
        [_evidence({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0})]
    )
    changed = decision_kl.compute_trajectory_kl(
        [_evidence({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})]
    )

    assert unchanged.status == "complete"
    assert unchanged.trajectory_kl == pytest.approx(0.0)
    assert changed.status == "complete"
    assert changed.trajectory_kl == pytest.approx(
        _new_old_kl([0.0, 1.0], [1.0, 0.0])
    )
    assert math.isfinite(changed.trajectory_kl)
    assert changed.trajectory_kl > 0.0


def test_episode_primary_information_gain_is_trace_mean_and_sum_is_distinct():
    evidence = [
        _evidence({"a": 0.8, "b": 0.2}, {"a": 0.6, "b": 0.4}, step=1),
        _evidence({"a": 0.3, "b": 0.7}, {"a": 0.5, "b": 0.5}, step=2),
    ]
    summary = decision_kl.compute_trajectory_kl(evidence)

    assert summary.information_gain == pytest.approx(math.fsum(summary.trace) / 2)
    assert summary.sum_local_kl == pytest.approx(math.fsum(summary.trace))
    assert summary.information_gain_unit == "nats / decision"
    assert summary.sum_local_kl_unit == "nats / episode"
    assert summary.aggregation == "arithmetic_mean"


def test_formal_runtime_config_has_no_noncanonical_epsilon_entry():
    from agentbench_frame.eval.trajectory_kl import TrajectoryKLConfig

    config = TrajectoryKLConfig.for_policy_information_gain(
        "old", "new", metadata={"sensitivity_epsilon": 0.05}
    )
    assert config.epsilon == MAIN_EPSILON
    assert config.metadata["sensitivity_epsilon"] == 0.05
    with pytest.raises(TypeError):
        TrajectoryKLConfig.for_policy_information_gain(
            "old", "new", epsilon=0.05
        )


def test_generic_runtime_epsilon_is_separate_from_formal_information_gain():
    from agentbench_frame.eval.information_gain import (
        FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
        GENERIC_TRAJECTORY_KL_PROFILE,
    )
    from agentbench_frame.eval.trajectory_kl import TrajectoryKLConfig

    generic = TrajectoryKLConfig("old", "new", 0.05)
    assert generic.measurement_profile == GENERIC_TRAJECTORY_KL_PROFILE
    assert not generic.is_formal_policy_information_gain

    formal = TrajectoryKLConfig.for_policy_information_gain("old", "new")
    assert formal.epsilon == MAIN_EPSILON
    assert formal.measurement_profile == FORMAL_POLICY_INFORMATION_GAIN_PROFILE
    assert formal.is_formal_policy_information_gain
    with pytest.raises(TypeError):
        TrajectoryKLConfig.for_policy_information_gain(
            "old", "new", epsilon=0.05
        )


def test_sensitivity_is_reproducible_without_changing_main_identity():
    from agentbench_frame.eval import information_gain

    sensitivity = information_gain.policy_kl_sensitivity(
        [0.0, 1.0],
        [1.0, 0.0],
    )
    repeated = information_gain.policy_kl_sensitivity(
        [0.0, 1.0],
        [1.0, 0.0],
    )

    assert sensitivity == repeated
    assert tuple(sensitivity) == (0.001, 0.01, 0.05)
    assert sensitivity[MAIN_EPSILON] == pytest.approx(
        information_gain.policy_kl(
            [0.0, 1.0], [1.0, 0.0], epsilon=MAIN_EPSILON
        )
    )
    assert information_gain.MAIN_POLICY_KL_EPSILON == MAIN_EPSILON


def test_generic_probability_boundary_rejects_bool_and_string_coercion():
    from agentbench_frame.eval.information_gain import (
        policy_kl,
        validate_policy_distribution,
    )

    support = ActionSupport(
        [ActionCandidate("a", "a"), ActionCandidate("b", "b")],
        "test-support-v1",
    )
    with pytest.raises(TypeError, match="probabil"):
        validate_policy_distribution({"a": True, "b": 0.0}, support)
    with pytest.raises(TypeError, match="probabil"):
        policy_kl(["1", "0"], [1.0, 0.0], epsilon=MAIN_EPSILON)


def test_episode_trace_rejects_incomplete_policy_support_instead_of_zero_filling():
    from agentbench_frame.eval.information_gain import episode_policy_kl_trace

    with pytest.raises(ValueError, match="exactly match legal actions"):
        episode_policy_kl_trace(
            lambda _context: {"a": 1.0},
            lambda _context: {"a": 0.5, "b": 0.5},
            contexts=["state"],
            legal_actions=lambda _context: ["a", "b"],
            epsilon=MAIN_EPSILON,
        )


def test_support_size_one_and_action_id_reordering_are_well_defined():
    from agentbench_frame.eval.information_gain import epsilon_regularize, policy_kl

    assert policy_kl([1.0], [1.0], epsilon=MAIN_EPSILON) == pytest.approx(0.0)
    assert epsilon_regularize([1.0, 0.0, 0.0], MAIN_EPSILON) == pytest.approx(
        [1.0 - MAIN_EPSILON + MAIN_EPSILON / 3, MAIN_EPSILON / 3, MAIN_EPSILON / 3]
    )
    summary = decision_kl.compute_trajectory_kl(
        [_evidence({"b": 0.2, "a": 0.8}, {"b": 0.4, "a": 0.6})]
    )
    assert summary.status == "complete"
    assert summary.trajectory_kl == pytest.approx(
        _new_old_kl([0.6, 0.4], [0.8, 0.2])
    )


def test_iteration_validator_recomputes_local_kl_instead_of_trusting_upload():
    from agentbench_frame.games.miracle import iteration_protocol as protocol

    forged_local = 0.5
    payload = {
        "episode": 1,
        "version_before": "old",
        "version_after": "new",
            "epsilon": MAIN_EPSILON,
            "measurement_profile": "24_miracle_policy_information_gain_v2",
        "status": "complete",
        "measurement_status": "complete",
        "direction": "new||old",
        "log_base": "e",
        "rollout_source": "new_policy",
        "estimand": "epsilon_regularized_local_kl_sum_under_new_policy_occupancy",
        "decision_steps": 1,
        "trace": [forged_local],
        "trajectory_kl_episode": forged_local,
        "mean_local_policy_kl": forged_local,
        "errors": [],
        "decisions": [
            {
                "decision_step": 1,
                "context_ref": "context",
                "action_schema_version": "actions-v1",
                "support_id": "support",
                "legal_action_ids": ["a", "b"],
                "selected_action_id": "a",
                "new_distribution": {"a": 1.0, "b": 0.0},
                "old_distribution": {"a": 1.0, "b": 0.0},
                "new_probabilities": [1.0, 0.0],
                "old_probabilities": [1.0, 0.0],
                "local_policy_kl": forged_local,
                "errors": [],
            }
        ],
    }
    plan = SimpleNamespace(
        baseline_strategy_version="old", candidate_strategy_version="new"
    )

    with pytest.raises(protocol.IterationPreflightError, match="local policy KL"):
        protocol._validate_trajectory_kl(payload, plan)

    payload.update(
        status="incomplete",
        measurement_status="incomplete",
        errors=["later decision evidence missing"],
        trajectory_kl_episode=None,
        mean_local_policy_kl=None,
        information_gain=None,
        local_policy_kl_sum=None,
    )
    with pytest.raises(protocol.IterationPreflightError, match="local policy KL"):
        protocol._validate_trajectory_kl(payload, plan)


def test_occupancy_shift_remains_separate_from_episode_information_gain():
    from agentbench_frame.eval.information_gain import occupancy_shift

    summary = decision_kl.compute_trajectory_kl(
        [_evidence({"a": 0.8, "b": 0.2}, {"a": 0.6, "b": 0.4})]
    )
    shift = occupancy_shift(["new-state"], ["old-state"])

    assert shift > 0.0
    assert "occupancy" not in summary.to_dict()
    assert summary.information_gain == pytest.approx(summary.trace[0])

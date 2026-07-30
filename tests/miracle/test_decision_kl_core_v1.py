import inspect
import json
import math

import pytest

from agentbench_frame.eval.measurement import ActionCandidate, ActionSupport
from agentbench_frame.games.miracle import decision_kl_v1 as kl
from agentbench_frame.games.miracle.research_protocol import (
    IncompleteActionSupportError,
)


def empty_observation(*, artifact=None):
    artifacts = [] if artifact is None else [artifact]
    return {
        "round": 1,
        "camp": 0,
        "map": {"units": [], "barracks": [-1] * 4, "miracles": [30, 30]},
        "players": [[artifacts, 20, 20, [], []], [[], 0, 0, [], []]],
    }


def identity(support):
    return {
        "schema_version": support.schema_version,
        "support_id": support.support_id,
        "action_ids": list(support.action_ids),
    }


def two_action_support():
    return ActionSupport(
        (
            ActionCandidate("a", {"operation_type": "endround"}),
            ActionCandidate("b", {"operation_type": "surrender"}),
        ),
        "test-actions-v1",
    )


def local(old, new, *, support=None, supplied_identity=None, step=1):
    support = support or two_action_support()
    return kl.compute_local_kl(
        old,
        new,
        support,
        identity(support) if supplied_identity is None else supplied_identity,
        decision_step=step,
    )


def test_trusted_support_wraps_canonical_sorted_unique_atomic_commands():
    first = kl.build_trusted_action_support({"camp": 0})
    second = kl.build_trusted_action_support({"camp": 0})
    assert len(first.actions) == 840
    assert first.action_ids == second.action_ids == tuple(sorted(first.action_ids))
    assert len(first.action_ids) == len(set(first.action_ids))
    operations = {candidate.action["operation_type"] for candidate in first.actions}
    assert operations == {"init"}
    assert any(
        "Inferno" in candidate.action["operation_parameters"]["creatures"]
        for candidate in first.actions
    )


def test_runtime_support_exposes_only_public_atomic_families_and_no_continue():
    support = kl.build_trusted_action_support(empty_observation())
    operations = [candidate.action["operation_type"] for candidate in support.actions]
    assert set(operations) == {"endround", "surrender"}
    assert not ({"forbid", "select", "startround", "continue"} & set(operations))


def test_wind_blessing_fails_closed_instead_of_claiming_finite_support():
    wind = [7, 3, 0, 1, 0, 0, 0, [-1, -1, -1]]
    with pytest.raises(IncompleteActionSupportError, match="WindBlessing"):
        kl.build_trusted_action_support(empty_observation(artifact=wind))


def test_local_kl_is_strict_unsmoothed_old_new_natural_log():
    record = local({"a": 0.8, "b": 0.2}, {"a": 0.5, "b": 0.5})
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    reverse = local({"a": 0.5, "b": 0.5}, {"a": 0.8, "b": 0.2})
    assert record.status == "complete"
    assert record.local_kl == pytest.approx(expected)
    assert record.local_kl != pytest.approx(reverse.local_kl)
    assert record.direction == "old||new"
    assert record.smoothing == "none"
    assert record.log_base == "e"


def test_zero_probability_rules_are_structured_and_json_safe():
    zero_old = local({"a": 0.0, "b": 1.0}, {"a": 0.5, "b": 0.5})
    assert zero_old.local_kl == pytest.approx(math.log(2.0))
    infinite = local({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
    assert infinite.status == "threshold_failed"
    assert infinite.local_kl is None
    assert infinite.reason == "old_positive_new_zero"
    json.dumps(infinite.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid", [True, False, "0.5", -0.1, math.nan, math.inf])
def test_distribution_rejects_weak_or_nonstandard_numbers(invalid):
    with pytest.raises((TypeError, ValueError)):
        local({"a": invalid, "b": 1.0}, {"a": 0.5, "b": 0.5})


@pytest.mark.parametrize(
    "distribution",
    [
        {"a": 0.5},
        {"a": 0.5, "b": 0.4, "c": 0.1},
        {"a": 0.4, "b": 0.4},
    ],
)
def test_distribution_is_not_zero_filled_normalized_or_extended(distribution):
    with pytest.raises(ValueError):
        local(distribution, {"a": 0.5, "b": 0.5})


def test_probability_sum_uses_absolute_tolerance_1e_9():
    support = two_action_support()
    accepted = {"a": 0.5, "b": 0.5 + 0.9e-9}
    assert kl.validate_distribution(accepted, support)["b"] == accepted["b"]
    with pytest.raises(ValueError, match="sum"):
        kl.validate_distribution({"a": 0.5, "b": 0.5 + 1.1e-9}, support)


@pytest.mark.parametrize("mutation", ["missing", "extra", "schema", "support", "order"])
def test_support_identity_is_exact_and_not_self_reported(mutation):
    support = two_action_support()
    supplied = identity(support)
    if mutation == "missing":
        supplied.pop("support_id")
    elif mutation == "extra":
        supplied["extra"] = "x"
    elif mutation == "schema":
        supplied["schema_version"] = "forged"
    elif mutation == "support":
        supplied["support_id"] = "0" * 64
    else:
        supplied["action_ids"].reverse()
    with pytest.raises(ValueError, match="identity"):
        local(
            {"a": 0.5, "b": 0.5},
            {"a": 0.5, "b": 0.5},
            support=support,
            supplied_identity=supplied,
        )


def finite_record(value, step):
    support = two_action_support()
    return kl.DecisionKLRecord(
        decision_step=step,
        schema_version=support.schema_version,
        support_id=support.support_id,
        action_ids=support.action_ids,
        status="complete",
        local_kl=value,
    )


def trajectory_evidence(
    target_kl=0.0,
    *,
    step=1,
    state_before=None,
    supplied_identity=None,
    old_distribution=None,
    new_distribution=None,
):
    state_before = empty_observation() if state_before is None else state_before
    support = kl.build_trusted_action_support(state_before)
    first, second = support.action_ids
    if old_distribution is None:
        old_distribution = {first: 0.5, second: 0.5}
    if new_distribution is None:
        first_probability = (
            0.5
            if target_kl == 0.0
            else (1 - math.sqrt(1 - math.exp(-2 * target_kl))) / 2
        )
        new_distribution = {
            first: first_probability,
            second: 1 - first_probability,
        }
    return kl.DecisionKLEvidence(
        decision_step=step,
        state_before=state_before,
        support_identity=(
            identity(support) if supplied_identity is None else supplied_identity
        ),
        old_distribution=old_distribution,
        new_distribution=new_distribution,
    )


def test_trajectory_rejects_publicly_constructed_output_record():
    forged = finite_record(0.0, 1)
    with pytest.raises(TypeError, match="evidence"):
        kl.compute_trajectory_kl([forged])


def test_trajectory_evidence_schema_has_no_self_reported_scalar_input():
    support = kl.build_trusted_action_support(empty_observation())
    with pytest.raises(TypeError):
        kl.DecisionKLEvidence(
            decision_step=1,
            state_before=empty_observation(),
            support_identity=identity(support),
            old_distribution={action_id: 1 / len(support.action_ids) for action_id in support.action_ids},
            new_distribution={action_id: 1 / len(support.action_ids) for action_id in support.action_ids},
            local_kl=0.0,
        )
    payload = {
        "decision_step": 1,
        "state_before": empty_observation(),
        "support_identity": identity(support),
        "old_distribution": {
            action_id: 1 / len(support.action_ids)
            for action_id in support.action_ids
        },
        "new_distribution": {
            action_id: 1 / len(support.action_ids)
            for action_id in support.action_ids
        },
    }
    for forbidden in ("local_kl", "reported_local_kl", "decision_change_rate"):
        with pytest.raises(TypeError, match="evidence"):
            kl.compute_trajectory_kl([{**payload, forbidden: 0.0}])


def test_trajectory_recomputes_from_distributions_and_ignores_no_uploaded_kl():
    baseline = kl.compute_trajectory_kl([trajectory_evidence(0.002)])
    changed = kl.compute_trajectory_kl([trajectory_evidence(0.008)])
    assert baseline.trajectory_kl == pytest.approx(0.002)
    assert changed.trajectory_kl == pytest.approx(0.008)
    assert changed.trajectory_kl != baseline.trajectory_kl


def test_trajectory_rejects_support_identity_not_regenerated_from_state():
    state = empty_observation()
    support = kl.build_trusted_action_support(state)
    forged_identity = identity(support)
    forged_identity["support_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        kl.compute_trajectory_kl(
            [trajectory_evidence(state_before=state, supplied_identity=forged_identity)]
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_trajectory_rejects_incomplete_or_extended_distributions(mutation):
    state = empty_observation()
    support = kl.build_trusted_action_support(state)
    first, second = support.action_ids
    old = {first: 0.5, second: 0.5}
    if mutation == "missing":
        old.pop(second)
    else:
        old["forged-action"] = 0.0
    with pytest.raises(ValueError, match="action IDs"):
        kl.compute_trajectory_kl(
            [trajectory_evidence(state_before=state, old_distribution=old)]
        )


def test_empty_trajectory_is_incomplete_and_finite_trace_uses_arithmetic_mean():
    empty = kl.compute_trajectory_kl([])
    assert empty.status == "incomplete"
    assert empty.trajectory_kl is None
    summary = kl.compute_trajectory_kl(
        [trajectory_evidence(0.002, step=1), trajectory_evidence(0.008, step=2)]
    )
    assert summary.status == "complete"
    assert summary.trajectory_kl == pytest.approx(0.005)
    assert summary.sum_local_kl == pytest.approx(0.01)
    assert summary.unit == "nats / decision"
    assert summary.rollout_source == "new_policy"


@pytest.mark.parametrize(
    ("value", "status", "passed"),
    [(0.009999999, "complete", True), (0.01, "complete", True), (0.010000001, "threshold_failed", False)],
)
def test_threshold_is_exact_without_hidden_tolerance(value, status, passed):
    summary = kl.compute_trajectory_kl([trajectory_evidence(value)])
    assert summary.status == status
    assert summary.threshold_passed is passed
    assert summary.acceptance_threshold == 0.01


def test_incomplete_or_infinite_decision_cannot_produce_partial_scalar():
    state = empty_observation()
    support = kl.build_trusted_action_support(state)
    first, second = support.action_ids
    failed = trajectory_evidence(
        step=2,
        state_before=state,
        old_distribution={first: 1.0, second: 0.0},
        new_distribution={first: 0.0, second: 1.0},
    )
    summary = kl.compute_trajectory_kl([trajectory_evidence(0.0, step=1), failed])
    assert summary.status == "threshold_failed"
    assert summary.trajectory_kl is None
    assert summary.trace == (0.0, None)
    json.dumps(summary.to_dict(), allow_nan=False)


def test_decision_steps_must_be_strict_continuous_integers():
    for invalid in (True, 0, -1, 1.0, "1"):
        with pytest.raises(ValueError, match="decision_step"):
            trajectory_evidence(0.0, step=invalid)
    for evidence in (
        [trajectory_evidence(0.0, step=1), trajectory_evidence(0.0, step=1)],
        [trajectory_evidence(0.0, step=2)],
    ):
        with pytest.raises(ValueError, match="decision_step"):
            kl.compute_trajectory_kl(evidence)


def test_core_api_has_no_policy_environment_or_mutable_factory_inputs():
    for function in (
        kl.build_trusted_action_support,
        kl.validate_distribution,
        kl.compute_local_kl,
        kl.compute_trajectory_kl,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {
            "policy", "agent", "judge", "provider", "runner", "session", "factory"
        }

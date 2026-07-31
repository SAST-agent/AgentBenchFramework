"""Pure Decision Space and trusted trajectory-KL calculator for 24_miracle.

This module never runs a policy, environment, Judge, Provider, or session.  It
accepts only a visible observation, captured support identity, and captured
old/new probability distributions.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentbench_frame.eval.measurement import ActionSupport
from agentbench_frame.games.miracle.research_protocol import (
    IncompleteActionSupportError,
    build_action_support,
    enumerate_legal_commands,
)


ACCEPTANCE_THRESHOLD = 0.01
DIRECTION = "old||new"
ROLLOUT_SOURCE = "new_policy"
SMOOTHING = "none"
UNIT = "nats / decision"
_IDENTITY_KEYS = {"schema_version", "support_id", "action_ids"}
_DISTRIBUTION_SUM_TOLERANCE = 1e-9
_STRICT_MASS_ROUNDOFF = 8 * math.ulp(1.0)
_INCOMPLETE_REASONS = frozenset({
    "action_support_not_finitely_enumerable",
    "old_distribution_action_ids_mismatch",
    "new_distribution_action_ids_mismatch",
    "old_distribution_probability_type",
    "new_distribution_probability_type",
    "old_distribution_probability_non_finite",
    "new_distribution_probability_non_finite",
    "old_distribution_probability_negative",
    "new_distribution_probability_negative",
    "old_distribution_probability_sum_mismatch",
    "new_distribution_probability_sum_mismatch",
    "old_distribution_mass_not_strict",
    "new_distribution_mass_not_strict",
    "local_kl_negative_beyond_roundoff",
})


class _DistributionValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _passes_acceptance_threshold(value: float) -> bool:
    return value <= ACCEPTANCE_THRESHOLD


def _strict_step(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("decision_step must be a positive strict integer")
    return value


def _strict_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be an int or float, not bool")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class DecisionKLEvidence:
    """Raw decision evidence from which trajectory KL must be recomputed."""

    decision_step: int
    state_before: Mapping[str, Any]
    support_identity: Mapping[str, Any]
    old_distribution: Mapping[str, int | float]
    new_distribution: Mapping[str, int | float]

    def __post_init__(self) -> None:
        _strict_step(self.decision_step)
        if not isinstance(self.state_before, Mapping):
            raise TypeError("state_before must be an observation object")
        if not isinstance(self.support_identity, Mapping):
            raise TypeError("support_identity must be an object")
        if not isinstance(self.old_distribution, Mapping):
            raise TypeError("old_distribution must be an action-ID mapping")
        if not isinstance(self.new_distribution, Mapping):
            raise TypeError("new_distribution must be an action-ID mapping")


@dataclass(frozen=True)
class DecisionKLRecord:
    """One trusted local-KL result bound to a canonical ActionSupport."""

    decision_step: int
    schema_version: str
    support_id: str
    action_ids: tuple[str, ...]
    status: str
    local_kl: float | None
    reason: str | None = None
    direction: str = DIRECTION
    smoothing: str = SMOOTHING
    log_base: str = "e"

    def __post_init__(self) -> None:
        _strict_step(self.decision_step)
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.support_id, str) or not self.support_id:
            raise ValueError("support_id must be a non-empty string")
        action_ids = tuple(self.action_ids)
        if (
            not action_ids
            or any(not isinstance(item, str) or not item for item in action_ids)
            or len(action_ids) != len(set(action_ids))
        ):
            raise ValueError("action_ids must be non-empty, unique strings")
        object.__setattr__(self, "action_ids", action_ids)
        if self.status not in {"complete", "incomplete", "threshold_failed"}:
            raise ValueError("local KL status is invalid")
        if self.status == "complete":
            object.__setattr__(
                self,
                "local_kl",
                _strict_nonnegative_number(self.local_kl, "local_kl"),
            )
            if self.reason is not None:
                raise ValueError("complete local KL cannot have a failure reason")
        elif self.local_kl is not None:
            raise ValueError("failed or incomplete local KL must not expose a scalar")
        if self.status == "incomplete" and self.reason not in _INCOMPLETE_REASONS:
            raise ValueError("incomplete local KL reason is invalid")
        if self.direction != DIRECTION or self.smoothing != SMOOTHING:
            raise ValueError("local KL contract identity is invalid")
        if self.log_base != "e":
            raise ValueError("local KL must use the natural logarithm")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_step": self.decision_step,
            "schema_version": self.schema_version,
            "support_id": self.support_id,
            "action_ids": list(self.action_ids),
            "status": self.status,
            "local_kl": self.local_kl,
            "reason": self.reason,
            "direction": self.direction,
            "smoothing": self.smoothing,
            "log_base": self.log_base,
        }


@dataclass(frozen=True)
class TrajectoryKLSummary:
    """Arithmetic-mean trajectory value plus diagnostic-only aggregates."""

    status: str
    trajectory_kl: float | None
    trace: tuple[float | None, ...]
    threshold_passed: bool | None
    reason: str | None
    sum_local_kl: float | None
    max_local_kl: float | None
    p50_local_kl: float | None
    p95_local_kl: float | None
    acceptance_threshold: float = ACCEPTANCE_THRESHOLD
    direction: str = DIRECTION
    smoothing: str = SMOOTHING
    log_base: str = "e"
    unit: str = UNIT
    rollout_source: str = ROLLOUT_SOURCE
    aggregation: str = "arithmetic_mean"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trajectory_kl": self.trajectory_kl,
            "trace": list(self.trace),
            "threshold_passed": self.threshold_passed,
            "reason": self.reason,
            "sum_local_kl": self.sum_local_kl,
            "max_local_kl": self.max_local_kl,
            "p50_local_kl": self.p50_local_kl,
            "p95_local_kl": self.p95_local_kl,
            "acceptance_threshold": self.acceptance_threshold,
            "direction": self.direction,
            "smoothing": self.smoothing,
            "log_base": self.log_base,
            "unit": self.unit,
            "rollout_source": self.rollout_source,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class _IncompleteDecisionRecord:
    decision_step: int
    reason: str
    status: str = "incomplete"
    local_kl: None = None

    def __post_init__(self) -> None:
        _strict_step(self.decision_step)
        if self.reason not in _INCOMPLETE_REASONS:
            raise ValueError("incomplete decision reason is invalid")


def build_trusted_action_support(observation: Mapping[str, Any]) -> ActionSupport:
    """Derive complete canonical support from a trusted visible observation."""

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be an object")
    legal = enumerate_legal_commands(copy.deepcopy(dict(observation)))
    return build_action_support(legal)


def trusted_support_identity(support: ActionSupport) -> dict[str, Any]:
    if not isinstance(support, ActionSupport):
        raise TypeError("support must be an ActionSupport")
    return {
        "schema_version": support.schema_version,
        "support_id": support.support_id,
        "action_ids": list(support.action_ids),
    }


def _validate_support_identity(
    supplied: Mapping[str, Any], support: ActionSupport
) -> None:
    if not isinstance(supplied, Mapping) or set(supplied) != _IDENTITY_KEYS:
        raise ValueError("support identity must contain exactly the frozen fields")
    action_ids = supplied.get("action_ids")
    if not isinstance(action_ids, (list, tuple)):
        raise ValueError("support identity action_ids must be an ordered sequence")
    actual = (
        supplied.get("schema_version"),
        supplied.get("support_id"),
        tuple(action_ids),
    )
    trusted = (support.schema_version, support.support_id, support.action_ids)
    if actual != trusted:
        raise ValueError("support identity does not match trusted ActionSupport")


def validate_distribution(
    distribution: Mapping[str, int | float], support: ActionSupport
) -> dict[str, float]:
    """Validate an exact distribution without normalization or zero filling."""

    if not isinstance(support, ActionSupport):
        raise TypeError("support must be an ActionSupport")
    if not isinstance(distribution, Mapping):
        raise TypeError("distribution must be an action-ID mapping")
    if set(distribution) != set(support.action_ids):
        raise _DistributionValidationError(
            "action_ids_mismatch",
            "distribution action IDs must exactly match ActionSupport",
        )
    validated: dict[str, float] = {}
    for action_id in support.action_ids:
        value = distribution[action_id]
        label = f"probability[{action_id!r}]"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _DistributionValidationError(
                "probability_type",
                f"{label} must be an int or float, not bool",
            )
        number = float(value)
        if not math.isfinite(number):
            raise _DistributionValidationError(
                "probability_non_finite",
                f"{label} must be finite",
            )
        if number < 0.0:
            raise _DistributionValidationError(
                "probability_negative",
                f"{label} must be non-negative",
            )
        validated[action_id] = number
    if not math.isclose(
        math.fsum(validated.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_DISTRIBUTION_SUM_TOLERANCE,
    ):
        raise _DistributionValidationError(
            "probability_sum_mismatch",
            "distribution probability sum must equal 1 within 1e-9",
        )
    return validated


def _incomplete_local_record(
    step: int, support: ActionSupport, reason: str
) -> DecisionKLRecord:
    return DecisionKLRecord(
        step,
        support.schema_version,
        support.support_id,
        support.action_ids,
        "incomplete",
        None,
        reason,
    )


def _compute_local_kl(
    old_distribution: Mapping[str, int | float],
    new_distribution: Mapping[str, int | float],
    support: ActionSupport,
    support_identity: Mapping[str, Any],
    *,
    decision_step: int,
) -> DecisionKLRecord:
    """Compute strict unsmoothed local ``D_KL(old || new)``."""

    step = _strict_step(decision_step)
    _validate_support_identity(support_identity, support)
    try:
        old = validate_distribution(old_distribution, support)
    except _DistributionValidationError as exc:
        return _incomplete_local_record(
            step, support, f"old_distribution_{exc.code}"
        )
    try:
        new = validate_distribution(new_distribution, support)
    except _DistributionValidationError as exc:
        return _incomplete_local_record(
            step, support, f"new_distribution_{exc.code}"
        )
    for action_id in support.action_ids:
        old_probability = old[action_id]
        new_probability = new[action_id]
        if old_probability > 0.0 and new_probability == 0.0:
            return DecisionKLRecord(
                step,
                support.schema_version,
                support.support_id,
                support.action_ids,
                "threshold_failed",
                None,
                "old_positive_new_zero",
            )
    old_mass = math.fsum(old.values())
    new_mass = math.fsum(new.values())
    if abs(old_mass - 1.0) > _STRICT_MASS_ROUNDOFF:
        return _incomplete_local_record(
            step, support, "old_distribution_mass_not_strict"
        )
    if abs(new_mass - 1.0) > _STRICT_MASS_ROUNDOFF:
        return _incomplete_local_record(
            step, support, "new_distribution_mass_not_strict"
        )
    terms = [
        old_probability
        * (math.log(old_probability) - math.log(new[action_id]))
        for action_id in support.action_ids
        if (old_probability := old[action_id]) > 0.0
    ]
    value = math.fsum(terms)
    negative_roundoff = 8 * math.ulp(1.0) * max(1, len(terms))
    if value < 0.0:
        if abs(value) <= negative_roundoff:
            value = 0.0
        else:
            return _incomplete_local_record(
                step, support, "local_kl_negative_beyond_roundoff"
            )
    return DecisionKLRecord(
        step,
        support.schema_version,
        support.support_id,
        support.action_ids,
        "complete",
        value,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _missing_summary(
    status: str,
    trace: tuple[float | None, ...],
    reason: str,
    threshold_passed: bool | None,
) -> TrajectoryKLSummary:
    return TrajectoryKLSummary(
        status,
        None,
        trace,
        threshold_passed,
        reason,
        None,
        None,
        None,
        None,
    )


def compute_trajectory_kl(
    evidence: Sequence[DecisionKLEvidence],
) -> TrajectoryKLSummary:
    """Recompute and aggregate trusted local KL from raw decision evidence."""

    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence")
    if not evidence:
        return _missing_summary("incomplete", (), "empty_trajectory", None)
    records: list[DecisionKLRecord | _IncompleteDecisionRecord] = []
    for expected_step, item in enumerate(evidence, start=1):
        if type(item) is not DecisionKLEvidence:
            raise TypeError("trajectory inputs must be trusted decision evidence")
        if item.decision_step != expected_step:
            raise ValueError("decision_step must be strict, unique, and continuous")
        try:
            support = build_trusted_action_support(item.state_before)
        except IncompleteActionSupportError:
            records.append(_IncompleteDecisionRecord(
                item.decision_step,
                "action_support_not_finitely_enumerable",
            ))
            continue
        records.append(
            _compute_local_kl(
                item.old_distribution,
                item.new_distribution,
                support,
                item.support_identity,
                decision_step=item.decision_step,
            )
        )
    trace = tuple(record.local_kl for record in records)
    failed = next(
        (record for record in records if record.status == "threshold_failed"), None
    )
    if failed is not None:
        return _missing_summary(
            "threshold_failed", trace, failed.reason or "threshold_failed", False
        )
    incomplete = next(
        (record for record in records if record.status == "incomplete"), None
    )
    if incomplete is not None:
        return _missing_summary(
            "incomplete", trace, incomplete.reason or "incomplete_decision", None
        )
    values = [record.local_kl for record in records]
    if any(value is None for value in values):
        raise ValueError("complete local KL records must contain scalars")
    finite = [float(value) for value in values if value is not None]
    total = math.fsum(finite)
    mean = total / len(finite)
    passed = _passes_acceptance_threshold(mean)
    return TrajectoryKLSummary(
        "complete" if passed else "threshold_failed",
        mean,
        tuple(finite),
        passed,
        None if passed else "trajectory_kl_above_threshold",
        total,
        max(finite),
        _percentile(finite, 0.50),
        _percentile(finite, 0.95),
    )


__all__ = [
    "ACCEPTANCE_THRESHOLD",
    "DIRECTION",
    "ROLLOUT_SOURCE",
    "SMOOTHING",
    "UNIT",
    "DecisionKLEvidence",
    "DecisionKLRecord",
    "TrajectoryKLSummary",
    "build_trusted_action_support",
    "trusted_support_identity",
    "validate_distribution",
    "compute_trajectory_kl",
]

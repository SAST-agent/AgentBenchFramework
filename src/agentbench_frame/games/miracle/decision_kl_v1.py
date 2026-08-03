"""Pure Decision Space and trusted trajectory-KL calculator for 24_miracle.

This module never runs a policy, environment, Judge, Provider, or session.  It
accepts only a visible observation, captured support identity, and captured
old/new probability distributions.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agentbench_frame.eval.information_gain import (
    FORMAL_POLICY_INFORMATION_GAIN_UNIT,
    FORMAL_POLICY_KL_DIRECTION,
    FORMAL_POLICY_KL_ROLLOUT_SOURCE,
    FORMAL_POLICY_KL_SMOOTHING,
    FORMAL_POLICY_KL_SUM_UNIT,
    MAIN_POLICY_KL_EPSILON,
    formal_policy_kl,
)
from agentbench_frame.eval.measurement import ActionSupport
from agentbench_frame.games.miracle.research_protocol import (
    IncompleteActionSupportError,
    build_action_support,
    enumerate_legal_commands,
)


ACCEPTANCE_THRESHOLD = None
DIRECTION = FORMAL_POLICY_KL_DIRECTION
ROLLOUT_SOURCE = FORMAL_POLICY_KL_ROLLOUT_SOURCE
EPSILON = MAIN_POLICY_KL_EPSILON
SMOOTHING = FORMAL_POLICY_KL_SMOOTHING
UNIT = FORMAL_POLICY_INFORMATION_GAIN_UNIT
SUM_UNIT = FORMAL_POLICY_KL_SUM_UNIT
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


def _strict_step(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("decision_step must be a positive strict integer")
    return value


def _strict_nonnegative_number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an int or float, not bool")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _freeze_json_value(
    value: Any,
    label: str,
    active_containers: set[int] | None = None,
    *,
    reject_non_finite: bool = True,
) -> Any:
    """Copy JSON-shaped input into mappings and sequences that cannot mutate."""

    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{label} contains an invalid Unicode scalar") from exc
        return value
    if type(value) is float:
        if reject_non_finite and not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite floats")
        return value
    if active_containers is None:
        active_containers = set()
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{label} contains a cyclic container")
        active_containers.add(container_id)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{label} contains an unsupported mapping key")
                try:
                    key.encode("utf-8", errors="strict")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        f"{label} contains an invalid Unicode mapping key"
                    ) from exc
                frozen[key] = _freeze_json_value(
                    item,
                    f"{label}.{key}",
                    active_containers,
                    reject_non_finite=reject_non_finite,
                )
            return MappingProxyType(frozen)
        finally:
            active_containers.remove(container_id)
    if type(value) in {list, tuple}:
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{label} contains a cyclic container")
        active_containers.add(container_id)
        try:
            return tuple(
                _freeze_json_value(
                    item,
                    f"{label}[{index}]",
                    active_containers,
                    reject_non_finite=reject_non_finite,
                )
                for index, item in enumerate(value)
            )
        finally:
            active_containers.remove(container_id)
    raise TypeError(f"{label} contains an unsupported value type")


def _thaw_json_value(value: Any) -> Any:
    """Return a detached mutable JSON-shaped value for the rule enumerator."""

    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DecisionKLEvidence:
    """Synthetic inputs from which mechanism-only KL must be recomputed.

    This structure carries no issuer-bound policy or occupancy provenance.
    """

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
        for field_name in (
            "state_before",
            "support_identity",
            "old_distribution",
            "new_distribution",
        ):
            object.__setattr__(
                self,
                field_name,
                _freeze_json_value(
                    getattr(self, field_name),
                    field_name,
                    reject_non_finite=field_name
                    not in {"old_distribution", "new_distribution"},
                ),
            )


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
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
    epsilon: float = EPSILON
    smoothing: str = SMOOTHING
    log_base: str = "e"

    def __new__(cls, *_args: Any, **_kwargs: Any):
        raise TypeError("DecisionKLRecord can only be issued by the KL calculator")

    def __post_init__(self) -> None:
        _strict_step(self.decision_step)
        if type(self.schema_version) is not str or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if type(self.support_id) is not str or not self.support_id:
            raise ValueError("support_id must be a non-empty string")
        action_ids = tuple(self.action_ids)
        if (
            not action_ids
            or any(type(item) is not str or not item for item in action_ids)
            or len(action_ids) != len(set(action_ids))
        ):
            raise ValueError("action_ids must be non-empty, unique strings")
        object.__setattr__(self, "action_ids", action_ids)
        if (
            type(self.status) is not str
            or self.status not in {"complete", "incomplete"}
        ):
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
            raise ValueError("incomplete local KL must not expose a scalar")
        if self.reason is not None and type(self.reason) is not str:
            raise ValueError("local KL reason must be an exact string or null")
        if self.status == "incomplete" and self.reason not in _INCOMPLETE_REASONS:
            raise ValueError("incomplete local KL reason is invalid")
        if (
            type(self.direction) is not str
            or type(self.smoothing) is not str
            or self.direction != DIRECTION
            or type(self.epsilon) is not float
            or self.epsilon != EPSILON
            or self.smoothing != SMOOTHING
        ):
            raise ValueError("local KL contract identity is invalid")
        if type(self.log_base) is not str or self.log_base != "e":
            raise ValueError("local KL must use the natural logarithm")

    def to_dict(self) -> dict[str, Any]:
        if not _is_issued_decision_record(self):
            raise ValueError("issued KL decision record is required")
        return {
            "decision_step": self.decision_step,
            "schema_version": self.schema_version,
            "support_id": self.support_id,
            "action_ids": list(self.action_ids),
            "status": self.status,
            "local_kl": self.local_kl,
            "reason": self.reason,
            "direction": self.direction,
            "epsilon": self.epsilon,
            "smoothing": self.smoothing,
            "log_base": self.log_base,
        }


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class _IncompleteDecisionRecord:
    decision_step: int
    reason: str
    status: str = "incomplete"
    local_kl: None = None
    schema_version: None = None
    support_id: None = None
    action_ids: tuple[()] = ()
    direction: str = DIRECTION
    epsilon: float = EPSILON
    smoothing: str = SMOOTHING
    log_base: str = "e"

    def __new__(cls, *_args: Any, **_kwargs: Any):
        raise TypeError("incomplete KL records can only be issued by the calculator")

    def __post_init__(self) -> None:
        _strict_step(self.decision_step)
        if type(self.reason) is not str or self.reason not in _INCOMPLETE_REASONS:
            raise ValueError("incomplete decision reason is invalid")
        if (
            self.status != "incomplete"
            or self.local_kl is not None
            or self.schema_version is not None
            or self.support_id is not None
            or self.action_ids != ()
        ):
            raise ValueError("unavailable support identity must remain unknown")
        if (
            type(self.status) is not str
            or type(self.direction) is not str
            or type(self.epsilon) is not float
            or type(self.smoothing) is not str
            or type(self.log_base) is not str
            or self.direction != DIRECTION
            or self.epsilon != EPSILON
            or self.smoothing != SMOOTHING
            or self.log_base != "e"
        ):
            raise ValueError("incomplete local KL contract identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        if not _is_issued_decision_record(self):
            raise ValueError("issued KL decision record is required")
        return {
            "decision_step": self.decision_step,
            "schema_version": self.schema_version,
            "support_id": self.support_id,
            "action_ids": list(self.action_ids),
            "status": self.status,
            "local_kl": self.local_kl,
            "reason": self.reason,
            "direction": self.direction,
            "epsilon": self.epsilon,
            "smoothing": self.smoothing,
            "log_base": self.log_base,
        }


def _decision_record_snapshot(
    record: DecisionKLRecord | _IncompleteDecisionRecord,
) -> tuple[Any, ...]:
    record.__post_init__()
    return (
        record.decision_step,
        record.schema_version,
        record.support_id,
        record.action_ids,
        record.status,
        record.local_kl,
        record.reason,
        record.direction,
        record.epsilon,
        record.smoothing,
        record.log_base,
    )


def _build_decision_record_authority():
    registry: weakref.WeakKeyDictionary[
        DecisionKLRecord | _IncompleteDecisionRecord, tuple[Any, ...]
    ] = weakref.WeakKeyDictionary()

    def issue_supported(
        step: int,
        support: ActionSupport,
        status: str,
        local_kl: float | None,
        reason: str | None = None,
    ) -> DecisionKLRecord:
        record = object.__new__(DecisionKLRecord)
        values = {
            "decision_step": step,
            "schema_version": support.schema_version,
            "support_id": support.support_id,
            "action_ids": support.action_ids,
            "status": status,
            "local_kl": local_kl,
            "reason": reason,
            "direction": DIRECTION,
            "epsilon": EPSILON,
            "smoothing": SMOOTHING,
            "log_base": "e",
        }
        for field_name, value in values.items():
            object.__setattr__(record, field_name, value)
        snapshot = _decision_record_snapshot(record)
        registry[record] = snapshot
        return record

    def issue_without_support(step: int, reason: str) -> _IncompleteDecisionRecord:
        record = object.__new__(_IncompleteDecisionRecord)
        values = {
            "decision_step": step,
            "reason": reason,
            "status": "incomplete",
            "local_kl": None,
            "schema_version": None,
            "support_id": None,
            "action_ids": (),
            "direction": DIRECTION,
            "epsilon": EPSILON,
            "smoothing": SMOOTHING,
            "log_base": "e",
        }
        for field_name, value in values.items():
            object.__setattr__(record, field_name, value)
        snapshot = _decision_record_snapshot(record)
        registry[record] = snapshot
        return record

    def is_issued(value: Any) -> bool:
        if type(value) not in {DecisionKLRecord, _IncompleteDecisionRecord}:
            return False
        try:
            registered = registry.get(value)
            return (
                registered is not None
                and _decision_record_snapshot(value) == registered
            )
        except (AttributeError, TypeError, ValueError):
            return False

    return issue_supported, issue_without_support, is_issued


(
    _issue_supported_decision_record,
    _issue_unsupported_decision_record,
    _is_issued_decision_record,
) = _build_decision_record_authority()
del _build_decision_record_authority


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _exact_scientific_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(
            _exact_scientific_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _derived_trajectory_fields(
    records: tuple[DecisionKLRecord | _IncompleteDecisionRecord, ...],
) -> dict[str, Any]:
    trace = tuple(record.local_kl for record in records)
    missing = {
        "trajectory_kl": None,
        "sum_local_kl": None,
        "max_local_kl": None,
        "p50_local_kl": None,
        "p95_local_kl": None,
    }
    if not records:
        return {
            "status": "incomplete",
            "trace": trace,
            "threshold_passed": None,
            "reason": "empty_trajectory",
            **missing,
        }
    incomplete = next(
        (record for record in records if record.status == "incomplete"),
        None,
    )
    if incomplete is not None:
        return {
            "status": "incomplete",
            "trace": trace,
            "threshold_passed": None,
            "reason": incomplete.reason or "incomplete_decision",
            **missing,
        }
    values = tuple(
        _strict_nonnegative_number(record.local_kl, "decision record local_kl")
        for record in records
    )
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("trajectory aggregates must remain finite") from exc
    mean = total / len(values)
    if not math.isfinite(total) or not math.isfinite(mean):
        raise ValueError("trajectory aggregates must remain finite")
    return {
        "status": "complete",
        "trajectory_kl": mean,
        "trace": values,
        "threshold_passed": None,
        "reason": None,
        "sum_local_kl": total,
        "max_local_kl": max(values),
        "p50_local_kl": _percentile(values, 0.50),
        "p95_local_kl": _percentile(values, 0.95),
    }


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class TrajectoryKLSummary:
    """Fake-only arithmetic-mean result plus diagnostic-only aggregates."""

    status: str
    trajectory_kl: float | None
    trace: tuple[float | None, ...]
    decision_records: tuple[DecisionKLRecord | _IncompleteDecisionRecord, ...]
    threshold_passed: bool | None
    reason: str | None
    sum_local_kl: float | None
    max_local_kl: float | None
    p50_local_kl: float | None
    p95_local_kl: float | None
    acceptance_threshold: None = ACCEPTANCE_THRESHOLD
    direction: str = DIRECTION
    epsilon: float = EPSILON
    smoothing: str = SMOOTHING
    log_base: str = "e"
    unit: str = UNIT
    evidence_scope: str = "synthetic_fake_only"
    authoritative_readiness: bool = False
    rollout_source_contract: str = ROLLOUT_SOURCE
    verified_rollout_source: None = None
    policy_binding_verified: bool = False
    aggregation: str = "arithmetic_mean"
    information_gain_unit: str = UNIT
    sum_local_kl_unit: str = SUM_UNIT

    def __new__(cls, *_args: Any, **_kwargs: Any):
        raise TypeError("TrajectoryKLSummary can only be issued by the KL calculator")

    def __post_init__(self) -> None:
        records = tuple(self.decision_records)
        if any(
            not _is_issued_decision_record(record)
            for record in records
        ):
            raise TypeError("decision_records must contain issued local records")
        if tuple(record.decision_step for record in records) != tuple(
            range(1, len(records) + 1)
        ):
            raise ValueError("decision_records must be strict, ordered, and continuous")
        contract = {
            "acceptance_threshold": ACCEPTANCE_THRESHOLD,
            "direction": DIRECTION,
            "epsilon": EPSILON,
            "smoothing": SMOOTHING,
            "log_base": "e",
            "unit": UNIT,
            "evidence_scope": "synthetic_fake_only",
            "authoritative_readiness": False,
            "rollout_source_contract": ROLLOUT_SOURCE,
            "verified_rollout_source": None,
            "policy_binding_verified": False,
            "aggregation": "arithmetic_mean",
            "information_gain_unit": UNIT,
            "sum_local_kl_unit": SUM_UNIT,
        }
        for field_name, expected in contract.items():
            if not _exact_scientific_value(getattr(self, field_name), expected):
                raise ValueError(
                    f"trajectory KL {field_name} contract identity is invalid"
                )
        derived = _derived_trajectory_fields(records)
        for field_name, expected in derived.items():
            actual = tuple(self.trace) if field_name == "trace" else getattr(
                self, field_name
            )
            if not _exact_scientific_value(actual, expected):
                raise ValueError(
                    f"trajectory KL {field_name} must match decision_records"
                )
        object.__setattr__(self, "trace", derived["trace"])
        object.__setattr__(self, "decision_records", records)

    def to_dict(self) -> dict[str, Any]:
        if not _is_issued_trajectory_summary(self):
            raise ValueError("issued trajectory KL summary is required")
        return {
            "status": self.status,
            "trajectory_kl": self.trajectory_kl,
            "information_gain": self.information_gain,
            "trace": list(self.trace),
            "decision_records": [record.to_dict() for record in self.decision_records],
            "threshold_passed": self.threshold_passed,
            "reason": self.reason,
            "sum_local_kl": self.sum_local_kl,
            "max_local_kl": self.max_local_kl,
            "p50_local_kl": self.p50_local_kl,
            "p95_local_kl": self.p95_local_kl,
            "acceptance_threshold": self.acceptance_threshold,
            "direction": self.direction,
            "epsilon": self.epsilon,
            "smoothing": self.smoothing,
            "log_base": self.log_base,
            "unit": self.unit,
            "evidence_scope": self.evidence_scope,
            "authoritative_readiness": self.authoritative_readiness,
            "rollout_source_contract": self.rollout_source_contract,
            "verified_rollout_source": self.verified_rollout_source,
            "policy_binding_verified": self.policy_binding_verified,
            "aggregation": self.aggregation,
            "information_gain_unit": self.information_gain_unit,
            "sum_local_kl_unit": self.sum_local_kl_unit,
        }

    @property
    def information_gain(self) -> float | None:
        return self.trajectory_kl


def _trajectory_summary_snapshot(summary: TrajectoryKLSummary) -> tuple[Any, ...]:
    summary.__post_init__()
    return (
        summary.status,
        summary.trajectory_kl,
        summary.trace,
        summary.decision_records,
        summary.threshold_passed,
        summary.reason,
        summary.sum_local_kl,
        summary.max_local_kl,
        summary.p50_local_kl,
        summary.p95_local_kl,
        summary.acceptance_threshold,
        summary.direction,
        summary.epsilon,
        summary.smoothing,
        summary.log_base,
        summary.unit,
        summary.evidence_scope,
        summary.authoritative_readiness,
        summary.rollout_source_contract,
        summary.verified_rollout_source,
        summary.policy_binding_verified,
        summary.aggregation,
        summary.information_gain_unit,
        summary.sum_local_kl_unit,
    )


def _build_trajectory_summary_authority():
    registry: weakref.WeakKeyDictionary[TrajectoryKLSummary, tuple[Any, ...]] = (
        weakref.WeakKeyDictionary()
    )

    def issue(
        status: str,
        trajectory_kl: float | None,
        trace: tuple[float | None, ...],
        decision_records: tuple[
            DecisionKLRecord | _IncompleteDecisionRecord, ...
        ],
        threshold_passed: bool | None,
        reason: str | None,
        sum_local_kl: float | None,
        max_local_kl: float | None,
        p50_local_kl: float | None,
        p95_local_kl: float | None,
    ) -> TrajectoryKLSummary:
        summary = object.__new__(TrajectoryKLSummary)
        values = {
            "status": status,
            "trajectory_kl": trajectory_kl,
            "trace": trace,
            "decision_records": decision_records,
            "threshold_passed": threshold_passed,
            "reason": reason,
            "sum_local_kl": sum_local_kl,
            "max_local_kl": max_local_kl,
            "p50_local_kl": p50_local_kl,
            "p95_local_kl": p95_local_kl,
            "acceptance_threshold": ACCEPTANCE_THRESHOLD,
            "direction": DIRECTION,
            "epsilon": EPSILON,
            "smoothing": SMOOTHING,
            "log_base": "e",
            "unit": UNIT,
            "evidence_scope": "synthetic_fake_only",
            "authoritative_readiness": False,
            "rollout_source_contract": ROLLOUT_SOURCE,
            "verified_rollout_source": None,
            "policy_binding_verified": False,
            "aggregation": "arithmetic_mean",
            "information_gain_unit": UNIT,
            "sum_local_kl_unit": SUM_UNIT,
        }
        for field_name, value in values.items():
            object.__setattr__(summary, field_name, value)
        snapshot = _trajectory_summary_snapshot(summary)
        registry[summary] = snapshot
        return summary

    def is_issued(value: Any) -> bool:
        if type(value) is not TrajectoryKLSummary:
            return False
        try:
            registered = registry.get(value)
            return (
                registered is not None
                and _trajectory_summary_snapshot(value) == registered
            )
        except (AttributeError, TypeError, ValueError):
            return False

    return issue, is_issued


(
    _issue_trajectory_summary,
    _is_issued_trajectory_summary,
) = _build_trajectory_summary_authority()
del _build_trajectory_summary_authority


def build_trusted_action_support(observation: Mapping[str, Any]) -> ActionSupport:
    """Derive complete canonical support from a trusted visible observation."""

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be an object")
    frozen_observation = _freeze_json_value(observation, "observation")
    legal = enumerate_legal_commands(_thaw_json_value(frozen_observation))
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
        if type(value) not in (int, float):
            raise _DistributionValidationError(
                "probability_type",
                f"{label} must be an int or float, not bool",
            )
        if type(value) is int:
            if value < 0:
                raise _DistributionValidationError(
                    "probability_negative",
                    f"{label} must be non-negative",
                )
            if value > 1:
                raise _DistributionValidationError(
                    "probability_sum_mismatch",
                    "distribution probability sum must equal 1 within 1e-9",
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
        if number > 1.0 + _DISTRIBUTION_SUM_TOLERANCE:
            raise _DistributionValidationError(
                "probability_sum_mismatch",
                "distribution probability sum must equal 1 within 1e-9",
            )
        validated[action_id] = number
    try:
        total = math.fsum(validated.values())
    except OverflowError as exc:
        raise _DistributionValidationError(
            "probability_sum_mismatch",
            "distribution probability sum must equal 1 within 1e-9",
        ) from exc
    if not math.isclose(
        total,
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
    return _issue_supported_decision_record(
        step, support, "incomplete", None, reason
    )


def _compute_local_kl(
    old_distribution: Mapping[str, int | float],
    new_distribution: Mapping[str, int | float],
    support: ActionSupport,
    support_identity: Mapping[str, Any],
    *,
    decision_step: int,
) -> DecisionKLRecord:
    """Compute epsilon-regularized local ``D_KL(new || old)``."""

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
    value = formal_policy_kl(
        [new[action_id] for action_id in support.action_ids],
        [old[action_id] for action_id in support.action_ids],
    )
    negative_roundoff = 8 * math.ulp(1.0) * max(1, len(support.action_ids))
    if value < 0.0:
        if abs(value) <= negative_roundoff:
            value = 0.0
        else:
            return _incomplete_local_record(
                step, support, "local_kl_negative_beyond_roundoff"
            )
    return _issue_supported_decision_record(step, support, "complete", value)


def _missing_summary(
    status: str,
    trace: tuple[float | None, ...],
    records: tuple[DecisionKLRecord | _IncompleteDecisionRecord, ...],
    reason: str,
    threshold_passed: bool | None,
) -> TrajectoryKLSummary:
    return _issue_trajectory_summary(
        status,
        None,
        trace,
        records,
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
    """Recompute fake-only local KL without asserting policy provenance."""

    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence")
    frozen_evidence = tuple(evidence)
    if not frozen_evidence:
        return _missing_summary("incomplete", (), (), "empty_trajectory", None)
    records: list[DecisionKLRecord | _IncompleteDecisionRecord] = []
    for expected_step, item in enumerate(frozen_evidence, start=1):
        if type(item) is not DecisionKLEvidence:
            raise TypeError("trajectory inputs must be trusted decision evidence")
        item = DecisionKLEvidence(
            decision_step=item.decision_step,
            state_before=item.state_before,
            support_identity=item.support_identity,
            old_distribution=item.old_distribution,
            new_distribution=item.new_distribution,
        )
        if item.decision_step != expected_step:
            raise ValueError("decision_step must be strict, unique, and continuous")
        try:
            support = build_trusted_action_support(item.state_before)
        except IncompleteActionSupportError:
            records.append(
                _issue_unsupported_decision_record(
                    item.decision_step,
                    "action_support_not_finitely_enumerable",
                )
            )
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
    frozen_records = tuple(records)
    incomplete = next(
        (record for record in records if record.status == "incomplete"), None
    )
    if incomplete is not None:
        return _missing_summary(
            "incomplete",
            trace,
            frozen_records,
            incomplete.reason or "incomplete_decision",
            None,
        )
    values = [record.local_kl for record in records]
    if any(value is None for value in values):
        raise ValueError("complete local KL records must contain scalars")
    finite = [float(value) for value in values if value is not None]
    total = math.fsum(finite)
    mean = total / len(finite)
    return _issue_trajectory_summary(
        "complete",
        mean,
        tuple(finite),
        frozen_records,
        None,
        None,
        total,
        max(finite),
        _percentile(finite, 0.50),
        _percentile(finite, 0.95),
    )


__all__ = [
    "ACCEPTANCE_THRESHOLD",
    "DIRECTION",
    "EPSILON",
    "ROLLOUT_SOURCE",
    "SMOOTHING",
    "UNIT",
    "SUM_UNIT",
    "DecisionKLEvidence",
    "DecisionKLRecord",
    "TrajectoryKLSummary",
    "build_trusted_action_support",
    "trusted_support_identity",
    "validate_distribution",
    "compute_trajectory_kl",
]

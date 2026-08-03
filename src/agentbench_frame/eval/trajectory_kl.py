"""Online trajectory-KL measurement on the new policy's actual rollout."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

from agentbench_frame.eval.information_gain import (
    FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION,
    FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND,
    FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
    FORMAL_POLICY_INFORMATION_GAIN_UNIT,
    FORMAL_POLICY_KL_DIRECTION,
    FORMAL_POLICY_KL_LOG_BASE,
    FORMAL_POLICY_KL_ROLLOUT_SOURCE,
    FORMAL_POLICY_KL_SUM_ESTIMAND,
    FORMAL_POLICY_KL_SUM_UNIT,
    GENERIC_TRAJECTORY_KL_PROFILE,
    MAIN_POLICY_KL_EPSILON,
    episode_information_gain_from_trace,
    formal_policy_kl,
    policy_kl,
    trajectory_kl_from_trace,
    validate_policy_distribution,
)
from agentbench_frame.eval.measurement import (
    ActionSupport,
    PolicyDecision,
    canonical_state_id,
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return copy.deepcopy(value)


def _strict_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_nonnegative(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an int or float, not bool")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if type(left) not in {int, float} or type(right) not in {int, float}:
        return False

    left_number = float(left)
    right_number = float(right)
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and math.isclose(
            left_number,
            right_number,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
@dataclass(frozen=True)
class TrajectoryKLConfig:
    """Immutable identities and measurement-channel parameters."""

    version_before: str
    version_after: str
    epsilon: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    measurement_profile: str = GENERIC_TRAJECTORY_KL_PROFILE

    def __post_init__(self) -> None:
        _strict_text(self.version_before, "version_before")
        _strict_text(self.version_after, "version_after")
        if type(self.epsilon) not in {int, float}:
            raise TypeError("epsilon must be an int or float, not bool")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.measurement_profile not in {
            GENERIC_TRAJECTORY_KL_PROFILE,
            FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
        }:
            raise ValueError("measurement_profile is not supported")
        if (
            self.measurement_profile == FORMAL_POLICY_INFORMATION_GAIN_PROFILE
            and epsilon != MAIN_POLICY_KL_EPSILON
        ):
            raise ValueError("formal policy information gain epsilon must be exactly 0.01")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    @classmethod
    def for_policy_information_gain(
        cls,
        version_before: str,
        version_after: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TrajectoryKLConfig":
        """Build the fixed formal profile without an epsilon override."""

        return cls(
            version_before=version_before,
            version_after=version_after,
            epsilon=MAIN_POLICY_KL_EPSILON,
            metadata={} if metadata is None else metadata,
            measurement_profile=FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
        )

    @property
    def is_formal_policy_information_gain(self) -> bool:
        return self.measurement_profile == FORMAL_POLICY_INFORMATION_GAIN_PROFILE


@dataclass(frozen=True)
class TrajectoryKLDecisionRecord:
    """First-hand policy data and local KL for one target-agent decision."""

    decision_step: int
    context_ref: str
    action_schema_version: str
    support_id: str
    legal_action_ids: tuple[str, ...]
    selected_action_id: str
    new_distribution: Mapping[str, float]
    old_distribution: Optional[Mapping[str, float]]
    new_probabilities: Optional[tuple[float, ...]]
    old_probabilities: Optional[tuple[float, ...]]
    local_policy_kl: Optional[float]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.decision_step) is not int or self.decision_step <= 0:
            raise ValueError("decision_step must be a positive integer")
        for label in (
            "context_ref",
            "action_schema_version",
            "support_id",
            "selected_action_id",
        ):
            _strict_text(getattr(self, label), label)
        legal = tuple(self.legal_action_ids)
        if (
            not legal
            or any(not isinstance(item, str) or not item for item in legal)
            or len(set(legal)) != len(legal)
        ):
            raise ValueError("legal_action_ids must be non-empty and unique")
        if self.selected_action_id not in legal:
            raise ValueError("selected_action_id must belong to legal_action_ids")
        if not isinstance(self.new_distribution, Mapping):
            raise TypeError("new_distribution must be a mapping")
        if self.old_distribution is not None and not isinstance(
            self.old_distribution, Mapping
        ):
            raise TypeError("old_distribution must be a mapping or None")
        for label in ("new_probabilities", "old_probabilities"):
            values = getattr(self, label)
            if values is not None:
                values = tuple(
                    _strict_nonnegative(item, f"{label}[{index}]")
                    for index, item in enumerate(values)
                )
                object.__setattr__(self, label, values)
        if self.local_policy_kl is not None:
            object.__setattr__(
                self,
                "local_policy_kl",
                _strict_nonnegative(self.local_policy_kl, "local_policy_kl"),
            )
        errors = tuple(self.errors)
        if any(not isinstance(item, str) for item in errors):
            raise TypeError("decision errors must be strings")
        object.__setattr__(self, "legal_action_ids", legal)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(
            self, "new_distribution", _freeze_value(self.new_distribution)
        )
        object.__setattr__(
            self,
            "old_distribution",
            None
            if self.old_distribution is None
            else _freeze_value(self.old_distribution),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_step": self.decision_step,
            "context_ref": self.context_ref,
            "action_schema_version": self.action_schema_version,
            "support_id": self.support_id,
            "legal_action_ids": list(self.legal_action_ids),
            "selected_action_id": self.selected_action_id,
            "new_distribution": dict(self.new_distribution),
            "old_distribution": (
                dict(self.old_distribution)
                if self.old_distribution is not None
                else None
            ),
            "new_probabilities": (
                list(self.new_probabilities)
                if self.new_probabilities is not None
                else None
            ),
            "old_probabilities": (
                list(self.old_probabilities)
                if self.old_probabilities is not None
                else None
            ),
            "local_policy_kl": self.local_policy_kl,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class TrajectoryKLEpisodeResult:
    """Complete or explicitly incomplete trajectory-KL measurement."""

    episode: int
    version_before: str
    version_after: str
    epsilon: float
    status: str
    decisions: tuple[TrajectoryKLDecisionRecord, ...]
    trace: tuple[Optional[float], ...]
    trajectory_kl_episode: Optional[float]
    mean_local_policy_kl: Optional[float]
    errors: tuple[str, ...]
    metadata: Mapping[str, Any]
    measurement_profile: str = GENERIC_TRAJECTORY_KL_PROFILE
    direction: str = FORMAL_POLICY_KL_DIRECTION
    log_base: str = FORMAL_POLICY_KL_LOG_BASE
    rollout_source: str = FORMAL_POLICY_KL_ROLLOUT_SOURCE
    estimand: str = FORMAL_POLICY_KL_SUM_ESTIMAND
    information_gain_estimand: str = FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND
    aggregation: str = FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION
    information_gain_unit: str = FORMAL_POLICY_INFORMATION_GAIN_UNIT
    local_policy_kl_sum_unit: str = FORMAL_POLICY_KL_SUM_UNIT

    def __post_init__(self) -> None:
        if type(self.episode) is not int or self.episode <= 0:
            raise ValueError("episode must be a positive integer")
        _strict_text(self.version_before, "version_before")
        _strict_text(self.version_after, "version_after")
        if type(self.epsilon) not in {int, float}:
            raise TypeError("epsilon must be an int or float, not bool")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.measurement_profile not in {
            GENERIC_TRAJECTORY_KL_PROFILE,
            FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
        }:
            raise ValueError("measurement profile is not supported")
        if self.measurement_profile == FORMAL_POLICY_INFORMATION_GAIN_PROFILE:
            if epsilon != MAIN_POLICY_KL_EPSILON:
                raise ValueError("formal policy information gain epsilon must be 0.01")
            identity = {
                "direction": FORMAL_POLICY_KL_DIRECTION,
                "log_base": FORMAL_POLICY_KL_LOG_BASE,
                "rollout_source": FORMAL_POLICY_KL_ROLLOUT_SOURCE,
                "estimand": FORMAL_POLICY_KL_SUM_ESTIMAND,
                "information_gain_estimand": FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND,
                "aggregation": FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION,
                "information_gain_unit": FORMAL_POLICY_INFORMATION_GAIN_UNIT,
                "local_policy_kl_sum_unit": FORMAL_POLICY_KL_SUM_UNIT,
            }
            for label, expected in identity.items():
                if getattr(self, label) != expected:
                    raise ValueError(f"formal trajectory KL {label} identity mismatch")
        if self.status not in {"complete", "incomplete"}:
            raise ValueError("status must be complete or incomplete")
        decisions = tuple(self.decisions)
        if any(type(item) is not TrajectoryKLDecisionRecord for item in decisions):
            raise TypeError("decisions must contain TrajectoryKLDecisionRecord values")
        if tuple(item.decision_step for item in decisions) != tuple(
            range(1, len(decisions) + 1)
        ):
            raise ValueError("decision steps must be strict and continuous")
        trace = tuple(self.trace)
        if len(trace) != len(decisions):
            raise ValueError("trace must align one-to-one with decisions")
        for index, (value, decision) in enumerate(
            zip(trace, decisions, strict=True), start=1
        ):
            if value is not None:
                value = _strict_nonnegative(value, f"trace[{index}]")
            if not _same_number(value, decision.local_policy_kl):
                raise ValueError("trace must be derived from decision records")
        errors = tuple(self.errors)
        if any(not isinstance(item, str) for item in errors):
            raise TypeError("episode errors must be strings")
        if self.status == "complete":
            if not decisions or errors:
                raise ValueError("complete trajectory KL requires decisions and no errors")
            recomputed: list[float] = []
            for decision in decisions:
                if decision.errors or decision.old_distribution is None:
                    raise ValueError("complete decision cannot contain errors or missing policy")
                if (
                    decision.new_probabilities is None
                    or decision.old_probabilities is None
                    or decision.local_policy_kl is None
                ):
                    raise ValueError("complete decision requires probability evidence")
                legal = decision.legal_action_ids
                if (
                    set(decision.new_distribution) != set(legal)
                    or set(decision.old_distribution) != set(legal)
                ):
                    raise ValueError("complete decision distribution support mismatch")
                new = [decision.new_distribution[action_id] for action_id in legal]
                old = [decision.old_distribution[action_id] for action_id in legal]
                if any(
                    not _same_number(left, right)
                    for left, right in zip(
                        decision.new_probabilities, new, strict=True
                    )
                ) or any(
                    not _same_number(left, right)
                    for left, right in zip(
                        decision.old_probabilities, old, strict=True
                    )
                ):
                    raise ValueError("probability vectors disagree with distributions")
                local = (
                    formal_policy_kl(new, old)
                    if self.measurement_profile
                    == FORMAL_POLICY_INFORMATION_GAIN_PROFILE
                    else policy_kl(new, old, epsilon=epsilon)
                )
                if not math.isfinite(local) or not _same_number(
                    local, decision.local_policy_kl
                ):
                    raise ValueError("local policy KL disagrees with distributions")
                recomputed.append(local)
            total = trajectory_kl_from_trace(recomputed)
            mean = episode_information_gain_from_trace(recomputed)
            if not _same_number(self.trajectory_kl_episode, total):
                raise ValueError("trajectory KL aggregate must match decision records")
            if not _same_number(self.mean_local_policy_kl, mean):
                raise ValueError("mean local policy KL must match decision records")
        else:
            if self.trajectory_kl_episode is not None or self.mean_local_policy_kl is not None:
                raise ValueError("incomplete trajectory KL cannot expose aggregates")
            if not errors and not any(item.errors for item in decisions):
                raise ValueError("incomplete trajectory KL requires a structured error")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    @property
    def decision_steps(self) -> int:
        return len(self.decisions)

    @property
    def information_gain(self) -> Optional[float]:
        if self.measurement_profile != FORMAL_POLICY_INFORMATION_GAIN_PROFILE:
            return None
        return self.mean_local_policy_kl

    @property
    def local_policy_kl_sum(self) -> Optional[float]:
        return self.trajectory_kl_episode

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "epsilon": self.epsilon,
            "status": self.status,
            "measurement_status": self.status,
            "measurement_profile": self.measurement_profile,
            "direction": self.direction,
            "log_base": self.log_base,
            "rollout_source": self.rollout_source,
            "estimand": self.estimand,
            "information_gain_estimand": self.information_gain_estimand,
            "aggregation": self.aggregation,
            "decision_steps": self.decision_steps,
            "trace": list(self.trace),
            "trajectory_kl_episode": self.trajectory_kl_episode,
            "mean_local_policy_kl": self.mean_local_policy_kl,
            "information_gain": self.information_gain,
            "local_policy_kl_sum": self.local_policy_kl_sum,
            "information_gain_unit": self.information_gain_unit,
            "local_policy_kl_sum_unit": self.local_policy_kl_sum_unit,
            "errors": list(self.errors),
            "metadata": _thaw_value(self.metadata),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def trajectory_kl_result_from_payload(
    payload: Mapping[str, Any],
    *,
    require_formal: bool = False,
) -> TrajectoryKLEpisodeResult:
    """Rebuild and verify a rich result instead of trusting its summary."""

    if not isinstance(payload, Mapping):
        raise TypeError("trajectory KL result must be a mapping")
    required = {
        "episode",
        "version_before",
        "version_after",
        "epsilon",
        "status",
        "measurement_status",
        "measurement_profile",
        "direction",
        "log_base",
        "rollout_source",
        "estimand",
        "information_gain_estimand",
        "aggregation",
        "information_gain_unit",
        "local_policy_kl_sum_unit",
        "decision_steps",
        "trace",
        "trajectory_kl_episode",
        "mean_local_policy_kl",
        "information_gain",
        "local_policy_kl_sum",
        "errors",
        "metadata",
        "decisions",
    }
    missing = sorted(required - set(payload))
    if missing:
        if "measurement_profile" in missing:
            raise ValueError("trajectory KL measurement profile is required")
        raise ValueError(f"trajectory KL result is missing fields: {missing}")
    if payload["status"] != payload["measurement_status"]:
        raise ValueError("trajectory KL status identity mismatch")
    if require_formal and (
        payload["measurement_profile"] != FORMAL_POLICY_INFORMATION_GAIN_PROFILE
    ):
        raise ValueError("formal trajectory KL measurement profile is required")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list):
        raise TypeError("trajectory KL decisions must be a list")
    decisions = []
    for raw in raw_decisions:
        if not isinstance(raw, Mapping):
            raise TypeError("trajectory KL decisions must be objects")
        decisions.append(
            TrajectoryKLDecisionRecord(
                decision_step=raw.get("decision_step"),
                context_ref=raw.get("context_ref"),
                action_schema_version=raw.get("action_schema_version"),
                support_id=raw.get("support_id"),
                legal_action_ids=tuple(raw.get("legal_action_ids", ())),
                selected_action_id=raw.get("selected_action_id"),
                new_distribution=raw.get("new_distribution"),
                old_distribution=raw.get("old_distribution"),
                new_probabilities=(
                    None
                    if raw.get("new_probabilities") is None
                    else tuple(raw.get("new_probabilities"))
                ),
                old_probabilities=(
                    None
                    if raw.get("old_probabilities") is None
                    else tuple(raw.get("old_probabilities"))
                ),
                local_policy_kl=raw.get("local_policy_kl"),
                errors=tuple(raw.get("errors", ())),
            )
        )
    result = TrajectoryKLEpisodeResult(
        episode=payload["episode"],
        version_before=payload["version_before"],
        version_after=payload["version_after"],
        epsilon=payload["epsilon"],
        status=payload["status"],
        decisions=tuple(decisions),
        trace=tuple(payload["trace"]),
        trajectory_kl_episode=payload["trajectory_kl_episode"],
        mean_local_policy_kl=payload["mean_local_policy_kl"],
        errors=tuple(payload["errors"]),
        metadata=payload["metadata"],
        measurement_profile=payload["measurement_profile"],
        direction=payload["direction"],
        log_base=payload["log_base"],
        rollout_source=payload["rollout_source"],
        estimand=payload["estimand"],
        information_gain_estimand=payload["information_gain_estimand"],
        aggregation=payload["aggregation"],
        information_gain_unit=payload["information_gain_unit"],
        local_policy_kl_sum_unit=payload["local_policy_kl_sum_unit"],
    )
    if (
        type(payload["decision_steps"]) is not int
        or payload["decision_steps"] != result.decision_steps
    ):
        raise ValueError("trajectory KL decision count mismatch")
    if not _same_number(payload["information_gain"], result.information_gain):
        raise ValueError("information gain summary must match decision records")
    if not _same_number(payload["local_policy_kl_sum"], result.local_policy_kl_sum):
        raise ValueError("local policy KL sum must match decision records")
    return result


class TrajectoryKLAgent:
    """Agent-compatible online wrapper for strict trajectory-KL measurement.

    The active policy is the only object that selects an environment action.
    The reference policy is queried on the same context and support, then both
    sessions observe the actual transition produced by the active rollout.
    """

    def __init__(
        self,
        active_policy: Any,
        reference_policy: Any,
        support_provider: Callable[[Any], ActionSupport],
        config: TrajectoryKLConfig,
        on_episode_complete: Optional[
            Callable[[TrajectoryKLEpisodeResult], None]
        ] = None,
    ):
        self.active_policy = active_policy
        self.reference_policy = reference_policy
        self.support_provider = support_provider
        self.config = config
        self.on_episode_complete = on_episode_complete
        self.name = getattr(active_policy, "name", "trajectory-kl-agent")
        self.metadata = getattr(active_policy, "metadata", {})
        self._episode = 0
        self._episode_metadata: dict[str, Any] = {}
        self._pending_episode_metadata: dict[str, Any] = {}
        self._decisions: list[TrajectoryKLDecisionRecord] = []
        self._errors: list[str] = []
        self._latest_result: Optional[TrajectoryKLEpisodeResult] = None

    def set_measurement_episode_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._pending_episode_metadata = dict(metadata)

    def reset(self) -> None:
        if self._episode > 0 and self._latest_result is None:
            self.abort_episode("reset before terminal transition")
        self._episode += 1
        self._episode_metadata = {
            **dict(self.config.metadata),
            **self._pending_episode_metadata,
        }
        self._pending_episode_metadata = {}
        self._decisions = []
        self._errors = []
        self._latest_result = None
        for policy in (self.active_policy, self.reference_policy):
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()

    def act(self, observation: Any) -> Any:
        support = self.support_provider(observation)
        if not isinstance(support, ActionSupport):
            raise TypeError("support_provider must return ActionSupport")

        decide = getattr(self.active_policy, "decide_with_distribution", None)
        if not callable(decide):
            raise TypeError(
                "active policy must implement decide_with_distribution"
            )
        decision = decide(observation, support)
        if not isinstance(decision, PolicyDecision):
            raise TypeError(
                "decide_with_distribution must return PolicyDecision"
            )
        action = support.resolve(decision.action_id)

        decision_errors: list[str] = []
        new_distribution = dict(decision.probabilities)
        new_probabilities: Optional[list[float]] = None
        try:
            new_probabilities = validate_policy_distribution(
                decision.probabilities, support
            )
        except (TypeError, ValueError) as exc:
            decision_errors.append(f"new policy distribution: {exc}")

        old_distribution: Optional[dict[str, float]] = None
        old_probabilities: Optional[list[float]] = None
        query = getattr(
            self.reference_policy, "distribution_for_measurement", None
        )
        if not callable(query):
            decision_errors.append(
                "reference policy must implement distribution_for_measurement"
            )
        else:
            try:
                raw_old = query(observation, support)
                if isinstance(raw_old, Mapping):
                    old_distribution = dict(raw_old)
                old_probabilities = validate_policy_distribution(
                    raw_old, support
                )
            except Exception as exc:
                decision_errors.append(f"reference policy distribution: {exc}")

        local_kl = None
        if new_probabilities is not None and old_probabilities is not None:
            local_kl = (
                formal_policy_kl(new_probabilities, old_probabilities)
                if self.config.is_formal_policy_information_gain
                else policy_kl(
                    new_probabilities,
                    old_probabilities,
                    epsilon=self.config.epsilon,
                )
            )
            if not math.isfinite(local_kl) or local_kl < 0.0:
                decision_errors.append(
                    "local policy KL is not finite and non-negative"
                )
                local_kl = None

        if decision_errors:
            self._errors.extend(
                f"decision {len(self._decisions) + 1}: {message}"
                for message in decision_errors
            )
        self._decisions.append(
            TrajectoryKLDecisionRecord(
                decision_step=len(self._decisions) + 1,
                context_ref=canonical_state_id(observation),
                action_schema_version=support.schema_version,
                support_id=support.support_id,
                legal_action_ids=support.action_ids,
                selected_action_id=decision.action_id,
                new_distribution=new_distribution,
                old_distribution=old_distribution,
                new_probabilities=(
                    tuple(new_probabilities)
                    if new_probabilities is not None
                    else None
                ),
                old_probabilities=(
                    tuple(old_probabilities)
                    if old_probabilities is not None
                    else None
                ),
                local_policy_kl=local_kl,
                errors=tuple(decision_errors),
            )
        )
        return action

    def observe_transition(self, transition: Mapping[str, Any]) -> None:
        transition_snapshot = copy.deepcopy(dict(transition))
        for label, policy in (
            ("new policy", self.active_policy),
            ("reference policy", self.reference_policy),
        ):
            observe = getattr(policy, "observe_transition", None)
            if callable(observe):
                try:
                    observe(copy.deepcopy(transition_snapshot))
                except Exception as exc:
                    self._errors.append(
                        f"{label} transition observation failed: {exc}"
                    )
        if transition_snapshot.get("terminated") or transition_snapshot.get(
            "truncated"
        ):
            self._finish_episode()

    def abort_episode(self, reason: str) -> TrajectoryKLEpisodeResult:
        """Finalize an unfinished episode as incomplete without discarding data."""

        if self._latest_result is not None:
            return self._latest_result
        self._errors.append(f"episode aborted: {reason}")
        return self._finish_episode()

    def _finish_episode(self) -> TrajectoryKLEpisodeResult:
        if self._latest_result is not None:
            return self._latest_result

        trace = tuple(decision.local_policy_kl for decision in self._decisions)
        complete = (
            bool(self._decisions)
            and not self._errors
            and all(value is not None for value in trace)
        )
        errors = list(self._errors)
        trajectory_kl_episode = None
        mean_local_policy_kl = None
        if complete:
            try:
                complete_trace = tuple(
                    value for value in trace if value is not None
                )
                trajectory_kl_episode = trajectory_kl_from_trace(complete_trace)
                mean_local_policy_kl = episode_information_gain_from_trace(
                    complete_trace
                )
            except (OverflowError, TypeError, ValueError) as exc:
                complete = False
                errors.append(f"episode KL aggregate is invalid: {exc}")
        if not self._decisions:
            errors.append("episode contains no target-agent decisions")
        result = TrajectoryKLEpisodeResult(
            episode=self._episode,
            version_before=self.config.version_before,
            version_after=self.config.version_after,
            epsilon=self.config.epsilon,
            status="complete" if complete else "incomplete",
            decisions=tuple(self._decisions),
            trace=trace,
            trajectory_kl_episode=trajectory_kl_episode,
            mean_local_policy_kl=mean_local_policy_kl,
            errors=tuple(errors),
            metadata=dict(self._episode_metadata),
            measurement_profile=self.config.measurement_profile,
        )
        self._latest_result = result
        if self.on_episode_complete is not None:
            self.on_episode_complete(result)
        return result

    @property
    def latest_trajectory_kl_result(
        self,
    ) -> Optional[TrajectoryKLEpisodeResult]:
        return self._latest_result

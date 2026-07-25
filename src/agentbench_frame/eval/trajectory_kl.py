"""Online trajectory-KL measurement on the new policy's actual rollout."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from agentbench_frame.eval.information_gain import (
    policy_kl,
    validate_policy_distribution,
)
from agentbench_frame.eval.measurement import (
    ActionSupport,
    PolicyDecision,
    canonical_state_id,
)


@dataclass(frozen=True)
class TrajectoryKLConfig:
    """Immutable identities and measurement-channel parameters."""

    version_before: str
    version_after: str
    epsilon: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.version_before, str) or not self.version_before:
            raise ValueError("version_before must be a non-empty string")
        if not isinstance(self.version_after, str) or not self.version_after:
            raise ValueError("version_after must be a non-empty string")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be finite and strictly between 0 and 1")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "metadata", dict(self.metadata))


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
    direction: str = "new||old"
    log_base: str = "e"
    rollout_source: str = "new_policy"

    @property
    def decision_steps(self) -> int:
        return len(self.decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "epsilon": self.epsilon,
            "status": self.status,
            "measurement_status": self.status,
            "direction": self.direction,
            "log_base": self.log_base,
            "rollout_source": self.rollout_source,
            "decision_steps": self.decision_steps,
            "trace": list(self.trace),
            "trajectory_kl_episode": self.trajectory_kl_episode,
            "mean_local_policy_kl": self.mean_local_policy_kl,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


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
            local_kl = policy_kl(
                new_probabilities,
                old_probabilities,
                epsilon=self.config.epsilon,
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
        for label, policy in (
            ("new policy", self.active_policy),
            ("reference policy", self.reference_policy),
        ):
            observe = getattr(policy, "observe_transition", None)
            if callable(observe):
                try:
                    observe(transition)
                except Exception as exc:
                    self._errors.append(
                        f"{label} transition observation failed: {exc}"
                    )
        if transition.get("terminated") or transition.get("truncated"):
            self._finish_episode()

    def _finish_episode(self) -> TrajectoryKLEpisodeResult:
        if self._latest_result is not None:
            return self._latest_result

        trace = tuple(decision.local_policy_kl for decision in self._decisions)
        complete = (
            bool(self._decisions)
            and not self._errors
            and all(value is not None for value in trace)
        )
        trajectory_kl_episode = (
            sum(value for value in trace if value is not None)
            if complete
            else None
        )
        mean_local_policy_kl = (
            trajectory_kl_episode / len(self._decisions)
            if trajectory_kl_episode is not None
            else None
        )
        errors = list(self._errors)
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


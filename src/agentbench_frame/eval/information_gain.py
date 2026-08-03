"""Policy-change and occupancy-shift calculations.

These functions operate on adapter-provided distributions and raw rollout
contexts. They do not attempt to construct a global game-tree measure.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional

from agentbench_frame.eval.measurement import ActionSupport


POLICY_SUM_TOLERANCE = 1e-9
MAIN_POLICY_KL_EPSILON = 0.01
POLICY_KL_SENSITIVITY_EPSILONS = (0.001, 0.01, 0.05)
GENERIC_TRAJECTORY_KL_PROFILE = "generic_trajectory_kl"
LEGACY_POLICY_KL_TRACE_PROFILE = "legacy_policy_kl_trace"
FORMAL_POLICY_INFORMATION_GAIN_PROFILE = (
    "24_miracle_policy_information_gain_v2"
)
FORMAL_POLICY_KL_DIRECTION = "new||old"
FORMAL_POLICY_KL_LOG_BASE = "e"
FORMAL_POLICY_KL_SMOOTHING = "symmetric_epsilon_uniform_full_support"
FORMAL_POLICY_KL_ROLLOUT_SOURCE = "new_policy"
FORMAL_POLICY_KL_SUM_ESTIMAND = (
    "epsilon_regularized_local_kl_sum_under_new_policy_occupancy"
)
FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND = (
    "epsilon_regularized_mean_local_policy_kl_under_new_policy_occupancy"
)
FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION = "arithmetic_mean"
FORMAL_POLICY_INFORMATION_GAIN_UNIT = "nats / decision"
FORMAL_POLICY_KL_SUM_UNIT = "nats / episode"


def _strict_number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an int or float, not bool")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _strict_probability_sequence(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("probability support cannot be empty")
    converted = [
        _strict_number(value, f"probability[{index}]")
        for index, value in enumerate(values)
    ]
    if not math.isclose(
        math.fsum(converted),
        1.0,
        rel_tol=0.0,
        abs_tol=POLICY_SUM_TOLERANCE,
    ):
        raise ValueError("policy probabilities must sum to 1")
    return converted


def validate_policy_distribution(
    distribution: Mapping[str, float],
    support: ActionSupport,
) -> List[float]:
    """Validate and align a strict action-ID probability distribution."""

    if not isinstance(distribution, Mapping):
        raise TypeError("policy distribution must be an action-ID mapping")
    expected = set(support.action_ids)
    actual = set(distribution)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(value) for value in actual - expected)
        raise ValueError(
            f"policy distribution support mismatch; missing={missing}, extra={extra}"
        )
    return _strict_probability_sequence(
        [distribution[action_id] for action_id in support.action_ids]
    )


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("probability support cannot be empty")
    converted = [
        _strict_number(value, f"mass[{index}]")
        for index, value in enumerate(values)
    ]
    total = math.fsum(converted)
    if total <= 0.0:
        raise ValueError("probability mass must be positive")
    return [value / total for value in converted]


def epsilon_regularize(
    probs: Sequence[float], epsilon: float, legal_count: Optional[int] = None
) -> List[float]:
    """Mix a distribution with uniform mass over the common legal support."""

    if type(epsilon) not in {int, float}:
        raise TypeError("epsilon must be an int or float, not bool")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    validated = _strict_probability_sequence(probs)
    support_size = legal_count if legal_count is not None else len(validated)
    if type(support_size) is not int or support_size != len(validated) or support_size <= 0:
        raise ValueError("legal_count must match the probability support")
    uniform = 1.0 / support_size
    return [(1.0 - epsilon) * value + epsilon * uniform for value in validated]


def policy_kl(
    new_probs: Sequence[float],
    old_probs: Sequence[float],
    epsilon: Optional[float] = None,
) -> float:
    """Compute ``KL(new || old)`` on an identical legal action support."""

    if len(new_probs) != len(old_probs) or not new_probs:
        raise ValueError("new and old distributions must share a non-empty support")
    if epsilon is None:
        new = _strict_probability_sequence(new_probs)
        old = _strict_probability_sequence(old_probs)
    else:
        new = epsilon_regularize(new_probs, epsilon)
        old = epsilon_regularize(old_probs, epsilon)

    value = 0.0
    for new_probability, old_probability in zip(new, old):
        if new_probability == 0.0:
            continue
        if old_probability == 0.0:
            return math.inf
        value += new_probability * math.log(new_probability / old_probability)
    return value


def formal_policy_kl(
    new_probs: Sequence[float],
    old_probs: Sequence[float],
) -> float:
    """Compute the formal 24_miracle local IG primitive.

    This entry point intentionally has no epsilon argument.  Research callers
    that need another epsilon (or no smoothing) must use :func:`policy_kl`,
    whose result does not carry the formal measurement profile.
    """

    return policy_kl(
        new_probs,
        old_probs,
        epsilon=MAIN_POLICY_KL_EPSILON,
    )


def policy_kl_sensitivity(
    new_probs: Sequence[float],
    old_probs: Sequence[float],
) -> dict[float, float]:
    """Derive the fixed sensitivity panel without changing the main identity."""

    return {
        epsilon: policy_kl(new_probs, old_probs, epsilon=epsilon)
        for epsilon in POLICY_KL_SENSITIVITY_EPSILONS
    }


def _distribution_for_context(policy: Callable[[Any], Any], context: Any, actions: List[Any]) -> List[float]:
    raw = policy(context)
    if isinstance(raw, Mapping):
        if set(raw) != set(actions):
            raise ValueError("policy distribution must exactly match legal actions")
        return [raw[action] for action in actions]
    values = list(raw)
    if len(values) != len(actions):
        raise ValueError("policy distribution length does not match legal actions")
    return values


def episode_policy_kl_trace(
    new_policy: Callable[[Any], Any],
    old_policy: Callable[[Any], Any],
    contexts: Iterable[Any],
    legal_actions: Callable[[Any], Iterable[Any]],
    epsilon: Optional[float] = None,
) -> List[float]:
    """Return one local KL value for each real target-agent decision context."""

    trace = []
    for context in contexts:
        actions = list(legal_actions(context))
        if not actions:
            raise ValueError("legal action support cannot be empty")
        new = _distribution_for_context(new_policy, context, actions)
        old = _distribution_for_context(old_policy, context, actions)
        trace.append(policy_kl(new, old, epsilon=epsilon))
    return trace


def occupancy_histogram(state_ids: Iterable[Any]) -> Dict[str, float]:
    """Return a normalized histogram over canonical state identifiers."""

    ids = [str(state_id) for state_id in state_ids]
    if not ids:
        raise ValueError("occupancy sample cannot be empty")
    counts = Counter(ids)
    total = float(len(ids))
    return {state_id: count / total for state_id, count in counts.items()}


def occupancy_shift(
    new_state_ids: Iterable[Any],
    old_state_ids: Iterable[Any],
    smoothing: float = 1e-12,
) -> float:
    """Compute occupancy ``KL(new occupancy || old occupancy)``.

    Additive smoothing is applied over the union of observed canonical state
    IDs so an unseen state does not produce an accidental infinity.
    """

    if smoothing < 0.0 or not math.isfinite(smoothing):
        raise ValueError("smoothing must be a finite non-negative number")
    new = occupancy_histogram(new_state_ids)
    old = occupancy_histogram(old_state_ids)
    states = sorted(set(new) | set(old))
    new_mass = [new.get(state, 0.0) + smoothing for state in states]
    old_mass = [old.get(state, 0.0) + smoothing for state in states]
    new_mass = _normalize(new_mass)
    old_mass = _normalize(old_mass)
    return sum(p * math.log(p / q) for p, q in zip(new_mass, old_mass) if p > 0.0)


def trajectory_kl_from_trace(trace: Iterable[float]) -> float:
    """Return the per-rollout trajectory-KL contribution derived from a trace."""

    values = [
        _strict_number(value, f"KL trace[{index}]")
        for index, value in enumerate(trace)
    ]
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("KL trace sum must be finite")
    return total


def episode_information_gain_from_trace(trace: Iterable[float]) -> float:
    """Return the primary episode IG: the unweighted local-KL trace mean."""

    values = tuple(trace)
    if not values:
        raise ValueError("episode information gain requires a non-empty trace")
    return trajectory_kl_from_trace(values) / len(values)

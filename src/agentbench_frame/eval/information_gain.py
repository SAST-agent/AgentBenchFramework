"""Policy-change and occupancy-shift calculations.

These functions operate on adapter-provided distributions and raw rollout
contexts. They do not attempt to construct a global game-tree measure.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("probability support cannot be empty")
    converted = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(converted)
    if total <= 0.0:
        raise ValueError("probability mass must be positive")
    return [value / total for value in converted]


def epsilon_regularize(
    probs: Sequence[float], epsilon: float, legal_count: Optional[int] = None
) -> List[float]:
    """Mix a distribution with uniform mass over the common legal support."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    normalized = _normalize(probs)
    support_size = legal_count if legal_count is not None else len(normalized)
    if support_size != len(normalized) or support_size <= 0:
        raise ValueError("legal_count must match the probability support")
    uniform = 1.0 / support_size
    return [(1.0 - epsilon) * value + epsilon * uniform for value in normalized]


def policy_kl(
    new_probs: Sequence[float],
    old_probs: Sequence[float],
    epsilon: Optional[float] = None,
) -> float:
    """Compute ``KL(new || old)`` on an identical legal action support."""

    if len(new_probs) != len(old_probs) or not new_probs:
        raise ValueError("new and old distributions must share a non-empty support")
    if epsilon is None:
        new = _normalize(new_probs)
        old = _normalize(old_probs)
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


def _distribution_for_context(policy: Callable[[Any], Any], context: Any, actions: List[Any]) -> List[float]:
    raw = policy(context)
    if isinstance(raw, Mapping):
        return [float(raw.get(action, 0.0)) for action in actions]
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

    values = [float(value) for value in trace]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("KL trace values must be finite and non-negative")
    return sum(values)

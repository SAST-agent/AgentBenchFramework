"""Pure controlled-reference policy-KL fact and aggregate computation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from agentbench_frame.eval.information_gain import (
    uniform_smoothed_deterministic_kl,
)


CanonicalAction = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PolicyKLComputation:
    facts: tuple[dict[str, Any], ...]
    metric: dict[str, Any]
    complete: bool


def compute_controlled_policy_kl(
    *,
    versions: tuple[str, ...],
    reference_state_ids: tuple[str, ...],
    counts: Mapping[str, int],
    actions: Mapping[tuple[str, str], CanonicalAction],
    missing_actions: Mapping[tuple[str, str], str],
    epsilons: tuple[str, ...],
    primary_epsilon: str,
    action_space_spec_id: str,
) -> PolicyKLComputation:
    """Compute ordered facts and strict full-coverage aggregates."""

    facts = []
    transition_values: dict[tuple[str, str, str], list[Decimal]] = {}
    transition_disagreements: dict[tuple[str, str, str], int] = {}
    for before, after in zip(versions, versions[1:]):
        for state_id in reference_state_ids:
            old = actions.get((before, state_id))
            new = actions.get((after, state_id))
            reasons = []
            if state_id not in counts:
                reasons.append("missing exact support count")
            if old is None:
                reasons.append(
                    missing_actions.get(
                        (before, state_id),
                        f"missing {before} policy action",
                    )
                )
            if new is None:
                reasons.append(
                    missing_actions.get(
                        (after, state_id),
                        f"missing {after} policy action",
                    )
                )
            for epsilon in epsilons:
                key = (before, after, epsilon)
                if reasons:
                    decimal_value = None
                    display_value = None
                    status = "missing"
                else:
                    value = uniform_smoothed_deterministic_kl(
                        new,
                        old,
                        counts[state_id],
                        epsilon,
                        precision=80,
                    )
                    decimal_value = str(value)
                    display_value = float(value)
                    status = "complete"
                    transition_values.setdefault(key, []).append(value)
                    transition_disagreements[key] = (
                        transition_disagreements.get(key, 0)
                        + int(new != old)
                    )
                facts.append(
                    {
                        "version_before": before,
                        "version_after": after,
                        "measurement_state_id": state_id,
                        "support_size": (
                            str(counts[state_id])
                            if state_id in counts
                            else None
                        ),
                        "epsilon": epsilon,
                        "kl_nats_decimal": decimal_value,
                        "kl_nats": display_value,
                        "actions_equal": (
                            new == old
                            if new is not None and old is not None
                            else None
                        ),
                        "status": status,
                        "missing_reasons": sorted(set(reasons)),
                        "action_space_spec_id": action_space_spec_id,
                    }
                )

    total = len(reference_state_ids)
    transitions = []
    all_primary_complete = True
    for before, after in zip(versions, versions[1:]):
        sensitivity = {}
        for epsilon in epsilons:
            key = (before, after, epsilon)
            values = transition_values.get(key, [])
            coverage = {"complete": len(values), "total": total}
            if len(values) == total:
                with localcontext() as context:
                    context.prec = 80
                    mean = +(sum(values, Decimal(0)) / Decimal(total))
                mean_decimal = str(mean)
                mean_display = float(mean)
                disagreement_rate = (
                    transition_disagreements.get(key, 0) / total
                )
            else:
                mean_decimal = None
                mean_display = None
                disagreement_rate = None
            sensitivity[epsilon] = {
                "mean_kl_nats_decimal": mean_decimal,
                "mean_kl_nats": mean_display,
                "coverage": coverage,
                "action_disagreement_rate": disagreement_rate,
            }
        primary = sensitivity[primary_epsilon]
        if primary["coverage"]["complete"] != total:
            all_primary_complete = False
        transitions.append(
            {
                "version_before": before,
                "version_after": after,
                "mean_kl_nats_decimal": primary["mean_kl_nats_decimal"],
                "mean_kl_nats": primary["mean_kl_nats"],
                "coverage": primary["coverage"],
                "action_disagreement_rate": primary[
                    "action_disagreement_rate"
                ],
                "sensitivity": sensitivity,
            }
        )

    support_values = list(counts.values())
    metric = {
        "metric": "controlled_reference_policy_kl",
        "direction": "new||old",
        "primary_epsilon": primary_epsilon,
        "epsilons": list(epsilons),
        "reference_state_count": total,
        "action_space_spec_id": action_space_spec_id,
        "support_size": {
            "complete": len(support_values),
            "total": total,
            "minimum": (
                str(min(support_values)) if support_values else None
            ),
            "maximum": (
                str(max(support_values)) if support_values else None
            ),
        },
        "transitions": transitions,
        "scientific_scope": (
            "deterministic policy disagreement weighted by exact "
            "uniform-smoothed support scale; not epistemic information gain"
        ),
    }
    return PolicyKLComputation(
        facts=tuple(facts),
        metric=metric,
        complete=all_primary_complete,
    )

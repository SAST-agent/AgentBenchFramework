"""Paired descriptive 2x2 attribution for Generals policy interventions."""

from __future__ import annotations

from dataclasses import dataclass
import random
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class FactorialCaseEffect:
    pair_id: str
    large_stack: float | None
    contact: float | None
    interaction: float | None


@dataclass(frozen=True)
class FactorialMetricEffect:
    metric: str
    large_stack: float | None
    contact: float | None
    interaction: float | None
    complete_pair_count: int
    intervals: Mapping[str, tuple[float, float] | None]
    case_effects: Mapping[str, FactorialCaseEffect]


@dataclass(frozen=True)
class AttributionReport:
    pair_ids: tuple[str, ...]
    metrics: Mapping[str, FactorialMetricEffect]
    bootstrap_seed: int
    bootstrap_replicates: int


def _number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric or missing")
    return float(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_interval(
    values: list[float],
    *,
    rng: random.Random,
    replicates: int,
) -> tuple[float, float] | None:
    if not values:
        return None
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    low = int((replicates - 1) * 0.025)
    high = int((replicates - 1) * 0.975)
    return samples[low], samples[high]


def compute_paired_attribution(
    metric_cells: Mapping[
        str,
        Mapping[str, Mapping[str, float | int | None]],
    ],
    *,
    bootstrap_seed: int = 9049,
    bootstrap_replicates: int = 10_000,
) -> AttributionReport:
    """Compute B-A, C-A, and D-B-C+A on identical case coordinates."""

    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not metric_cells:
        raise ValueError("attribution requires at least one metric")

    pair_ids: tuple[str, ...] | None = None
    metrics: dict[str, FactorialMetricEffect] = {}
    rng = random.Random(bootstrap_seed)
    for metric in sorted(metric_cells):
        cells = metric_cells[metric]
        if set(cells) != {"A", "B", "C", "D"}:
            raise ValueError(f"{metric} must contain cells A, B, C, and D")
        coordinate_sets = {cell: set(cells[cell]) for cell in ("A", "B", "C", "D")}
        if len({frozenset(value) for value in coordinate_sets.values()}) != 1:
            raise ValueError(f"{metric} cells must contain identical pair IDs")
        current_pair_ids = tuple(sorted(coordinate_sets["A"]))
        if not current_pair_ids:
            raise ValueError(f"{metric} pair set cannot be empty")
        if pair_ids is None:
            pair_ids = current_pair_ids
        elif current_pair_ids != pair_ids:
            raise ValueError("all metrics must contain identical pair IDs")

        case_effects: dict[str, FactorialCaseEffect] = {}
        effect_values = {
            "large_stack": [],
            "contact": [],
            "interaction": [],
        }
        complete = 0
        for pair_id in current_pair_ids:
            values = {
                cell: _number(
                    cells[cell][pair_id],
                    f"{metric}/{cell}/{pair_id}",
                )
                for cell in ("A", "B", "C", "D")
            }
            if any(value is None for value in values.values()):
                effect = FactorialCaseEffect(pair_id, None, None, None)
            else:
                a, b, c, d = (
                    values[cell] for cell in ("A", "B", "C", "D")
                )
                assert a is not None and b is not None
                assert c is not None and d is not None
                effect = FactorialCaseEffect(
                    pair_id=pair_id,
                    large_stack=b - a,
                    contact=c - a,
                    interaction=d - b - c + a,
                )
                complete += 1
                effect_values["large_stack"].append(effect.large_stack)
                effect_values["contact"].append(effect.contact)
                effect_values["interaction"].append(effect.interaction)
            case_effects[pair_id] = effect

        intervals = {
            name: _bootstrap_interval(
                values,
                rng=rng,
                replicates=bootstrap_replicates,
            )
            for name, values in effect_values.items()
        }
        metrics[metric] = FactorialMetricEffect(
            metric=metric,
            large_stack=_mean(effect_values["large_stack"]),
            contact=_mean(effect_values["contact"]),
            interaction=_mean(effect_values["interaction"]),
            complete_pair_count=complete,
            intervals=MappingProxyType(intervals),
            case_effects=MappingProxyType(case_effects),
        )

    assert pair_ids is not None
    return AttributionReport(
        pair_ids=pair_ids,
        metrics=MappingProxyType(metrics),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )

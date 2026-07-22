"""Budget-curve derived metrics."""

from collections.abc import Mapping
from typing import Iterable, Optional, Tuple


def trapezoid_auc(points: Iterable[Tuple[float, Optional[float]]]) -> float:
    """Integrate contiguous complete ``(x, score)`` points by trapezoids.

    A missing score breaks the curve. This avoids silently interpolating an
    incomplete evaluation.
    """

    total = 0.0
    previous = None
    for x, score in points:
        if score is None:
            previous = None
            continue
        current = (float(x), float(score))
        if previous is not None:
            dx = current[0] - previous[0]
            if dx < 0:
                raise ValueError("AUC x values must be non-decreasing")
            total += dx * (current[1] + previous[1]) / 2.0
        previous = current
    return total


_AUC_AXES = {
    "coding_agent_act": "auc_coding_agent_act",
    "episode": "auc_episode",
    "env_step": "auc_env_step",
    "token": "auc_token",
    "time_s": "auc_time_s",
}


def multi_axis_auc(points: Iterable[Mapping[str, object]]) -> dict[str, Optional[float]]:
    """Integrate one score history against every available budget axis.

    All axes use the same rule as :func:`trapezoid_auc`: an explicit missing
    score breaks the curve, and no interpolation is performed.  Missing budget
    coordinates simply make that axis unavailable for the corresponding pair.
    """

    rows = list(points)
    result: dict[str, Optional[float]] = {}
    for axis, output_name in _AUC_AXES.items():
        axis_points = []
        for row in rows:
            x = row.get(axis)
            score = row.get("score")
            if x is None:
                axis_points.append((None, None))
            else:
                axis_points.append((float(x), None if score is None else float(score)))
        total = 0.0
        previous = None
        complete = 0
        for x, score in axis_points:
            if x is None or score is None:
                previous = None
                continue
            current = (x, score)
            complete += 1
            if previous is not None:
                dx = current[0] - previous[0]
                if dx < 0:
                    raise ValueError(f"{axis} AUC x values must be non-decreasing")
                total += dx * (current[1] + previous[1]) / 2.0
            previous = current
        result[output_name] = total if complete else None
    # ``auc_time`` is the historical report label; keep it as a harmless
    # additive alias while ``auc_time_s`` makes the unit explicit in APIs.
    result["auc_time"] = result["auc_time_s"]
    return result

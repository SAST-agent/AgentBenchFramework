"""Budget-curve derived metrics."""

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

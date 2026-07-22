"""AquaWar subprocess evaluation support."""

from agentbench_frame.aquawar.evaluator import (
    AquaWarEvaluationResult,
    AquaWarEvaluator,
    Opponent,
)
from agentbench_frame.aquawar.match import AquaWarMatchError, run_match

__all__ = [
    "AquaWarEvaluationResult",
    "AquaWarEvaluator",
    "AquaWarMatchError",
    "Opponent",
    "run_match",
]

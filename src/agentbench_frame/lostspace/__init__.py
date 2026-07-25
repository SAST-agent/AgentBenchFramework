"""LostSpace subprocess evaluation support.

Four-player turn-based survival/escape game (contest 25). This package mirrors
``agentbench_frame.aquawar``: spawn the official game logic + AI clients as
subprocesses, shuttle the Saiblo wire protocol, track runs, and save replays.
"""

from agentbench_frame.lostspace.evaluator import (
    LostSpaceEvaluationResult,
    LostSpaceEvaluator,
    Opponent,
)
from agentbench_frame.lostspace.match import LostSpaceMatchError, run_match

__all__ = [
    "LostSpaceEvaluationResult",
    "LostSpaceEvaluator",
    "LostSpaceMatchError",
    "Opponent",
    "run_match",
]

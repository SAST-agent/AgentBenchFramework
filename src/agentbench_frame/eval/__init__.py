"""
Evaluation and trajectory recording.

Provides:
- TrajectoryRecorder: Record game episodes for analysis
- MetricsCalculator: Compute win rates, Elo, and other statistics
"""

from agentbench_frame.eval.trajectory import TrajectoryRecorder
from agentbench_frame.eval.metrics import MetricsCalculator

__all__ = ["TrajectoryRecorder", "MetricsCalculator"]

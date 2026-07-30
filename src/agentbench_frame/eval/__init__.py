"""
Evaluation and trajectory recording.

Provides:
- TrajectoryRecorder: Record game episodes for analysis
- MetricsCalculator: Compute win rates, Elo, and other statistics
"""

from agentbench_frame.eval.trajectory import TrajectoryRecorder
from agentbench_frame.eval.metrics import MetricsCalculator
from agentbench_frame.eval.benchmark import (
    BenchmarkCase,
    BenchmarkEvaluation,
    BenchmarkSpec,
    GameResult,
    evaluate_benchmark,
)
from agentbench_frame.eval.curves import trapezoid_auc
from agentbench_frame.eval.information_gain import (
    deterministic_measurement_distribution,
    epsilon_regularize,
    episode_policy_kl_trace,
    occupancy_histogram,
    occupancy_shift,
    policy_kl,
    trajectory_kl_from_trace,
)
from agentbench_frame.eval.measurement import (
    ActionSupport,
    PolicyDistributionProvider,
    StateIdProvider,
    canonical_state_id,
    canonical_state_payload,
)

__all__ = [
    "TrajectoryRecorder",
    "MetricsCalculator",
    "BenchmarkCase",
    "BenchmarkEvaluation",
    "BenchmarkSpec",
    "GameResult",
    "evaluate_benchmark",
    "trapezoid_auc",
    "deterministic_measurement_distribution",
    "epsilon_regularize",
    "episode_policy_kl_trace",
    "occupancy_histogram",
    "occupancy_shift",
    "policy_kl",
    "trajectory_kl_from_trace",
    "PolicyDistributionProvider",
    "ActionSupport",
    "StateIdProvider",
    "canonical_state_id",
    "canonical_state_payload",
]

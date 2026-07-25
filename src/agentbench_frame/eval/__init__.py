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
    epsilon_regularize,
    episode_policy_kl_trace,
    occupancy_histogram,
    occupancy_shift,
    policy_kl,
    trajectory_kl_from_trace,
    validate_policy_distribution,
)
from agentbench_frame.eval.measurement import (
    ActionCandidate,
    ActionSupport,
    PolicyDistributionProvider,
    PolicyDecision,
    StateIdProvider,
    canonical_state_id,
    canonical_state_payload,
)
from agentbench_frame.eval.trajectory_kl import (
    TrajectoryKLAgent,
    TrajectoryKLConfig,
    TrajectoryKLDecisionRecord,
    TrajectoryKLEpisodeResult,
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
    "epsilon_regularize",
    "episode_policy_kl_trace",
    "occupancy_histogram",
    "occupancy_shift",
    "policy_kl",
    "trajectory_kl_from_trace",
    "validate_policy_distribution",
    "ActionCandidate",
    "ActionSupport",
    "PolicyDistributionProvider",
    "PolicyDecision",
    "StateIdProvider",
    "canonical_state_id",
    "canonical_state_payload",
    "TrajectoryKLAgent",
    "TrajectoryKLConfig",
    "TrajectoryKLDecisionRecord",
    "TrajectoryKLEpisodeResult",
]

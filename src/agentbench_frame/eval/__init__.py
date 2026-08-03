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
    FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION,
    FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND,
    FORMAL_POLICY_INFORMATION_GAIN_PROFILE,
    FORMAL_POLICY_INFORMATION_GAIN_UNIT,
    FORMAL_POLICY_KL_DIRECTION,
    FORMAL_POLICY_KL_LOG_BASE,
    FORMAL_POLICY_KL_ROLLOUT_SOURCE,
    FORMAL_POLICY_KL_SMOOTHING,
    FORMAL_POLICY_KL_SUM_ESTIMAND,
    FORMAL_POLICY_KL_SUM_UNIT,
    GENERIC_TRAJECTORY_KL_PROFILE,
    LEGACY_POLICY_KL_TRACE_PROFILE,
    MAIN_POLICY_KL_EPSILON,
    POLICY_KL_SENSITIVITY_EPSILONS,
    episode_information_gain_from_trace,
    epsilon_regularize,
    episode_policy_kl_trace,
    occupancy_histogram,
    occupancy_shift,
    policy_kl,
    formal_policy_kl,
    policy_kl_sensitivity,
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
    "episode_information_gain_from_trace",
    "episode_policy_kl_trace",
    "occupancy_histogram",
    "occupancy_shift",
    "policy_kl",
    "formal_policy_kl",
    "policy_kl_sensitivity",
    "trajectory_kl_from_trace",
    "MAIN_POLICY_KL_EPSILON",
    "POLICY_KL_SENSITIVITY_EPSILONS",
    "GENERIC_TRAJECTORY_KL_PROFILE",
    "LEGACY_POLICY_KL_TRACE_PROFILE",
    "FORMAL_POLICY_INFORMATION_GAIN_PROFILE",
    "FORMAL_POLICY_KL_DIRECTION",
    "FORMAL_POLICY_KL_LOG_BASE",
    "FORMAL_POLICY_KL_SMOOTHING",
    "FORMAL_POLICY_KL_ROLLOUT_SOURCE",
    "FORMAL_POLICY_KL_SUM_ESTIMAND",
    "FORMAL_POLICY_INFORMATION_GAIN_ESTIMAND",
    "FORMAL_POLICY_INFORMATION_GAIN_AGGREGATION",
    "FORMAL_POLICY_INFORMATION_GAIN_UNIT",
    "FORMAL_POLICY_KL_SUM_UNIT",
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

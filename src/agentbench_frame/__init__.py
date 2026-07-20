"""
AgentBench Unified Framework
=============================

A unified framework for agent-based game AI development supporting:
- Rule-based agent iteration with Skills and MCP tools
- Reinforcement learning training
- Adversarial arena (tournaments, matchmaking)
- Trajectory recording and evaluation
- Transparent experiment tracking and reporting
- CLI entry points for all workflows

Built for the THU Saiblo / 智能体大赛 ecosystem.
Games communicate via stdio using the THUAC RFC protocol.
"""

__version__ = "0.1.0"
__author__ = "AgentBench Team"

# Environment
from agentbench_frame.env import (
    BaseEnv, EnvMode, ActionSpace, Observation,
    StdioProtocol, GeneralsEnv,
    ENV_REGISTRY, make_env, register_env,
)

# Agent
from agentbench_frame.agent import (
    BaseAgent, RandomAgent,
    RuleBasedAgent,
    RLAgent, PolicyNetwork,
    AgentRegistry, register_agent,
)

# Arena
from agentbench_frame.arena import Arena, Match, EloTracker

# Evaluation
from agentbench_frame.eval import TrajectoryRecorder, MetricsCalculator

# Training
from agentbench_frame.training import (
    PPOTrainer, PPOConfig,
    RLTrainer, RLTrainConfig,
    RuleIterator, RuleIterConfig,
    SelfPlayTrainer, SelfPlayConfig,
)

# Skills
from agentbench_frame.skills import (
    Skill, SkillMeta, SkillStats, FunctionSkill,
    SkillRegistry, SkillVariant,
    ReplayReaderSkill, OpponentModelerSkill, MapAnalyzerSkill,
)

# MCP Tools
from agentbench_frame.mcp import (
    MCPTool, MCPToolRegistry, MCPServer,
    ReadReplayTool, AnalyzeGameTool, QueryHistoryTool,
)

# Tracking
from agentbench_frame.tracking import (
    Run,
    StepRecord, EpisodeRecord, EvalRecord,
    ResourceRecord, LogRecord, RunMeta,
    JSONLWriter, ResourceSampler,
    TrackedEnv, TimedAgent,
)

# Runners
from agentbench_frame.runner import (
    BaseRunner, BaseRLRunner, BaseRuleRunner, BaseEvalRunner,
)

# Report
from agentbench_frame.report import ReportBuilder, build_report

__all__ = [
    # Version
    "__version__",
    # Environment
    "BaseEnv", "EnvMode", "ActionSpace", "Observation",
    "StdioProtocol", "GeneralsEnv",
    "ENV_REGISTRY", "make_env", "register_env",
    # Agent
    "BaseAgent", "RandomAgent",
    "RuleBasedAgent",
    "RLAgent", "PolicyNetwork",
    "AgentRegistry", "register_agent",
    # Arena
    "Arena", "Match", "EloTracker",
    # Evaluation
    "TrajectoryRecorder", "MetricsCalculator",
    # Training
    "PPOTrainer", "PPOConfig",
    "RLTrainer", "RLTrainConfig",
    "RuleIterator", "RuleIterConfig",
    "SelfPlayTrainer", "SelfPlayConfig",
    # Skills
    "Skill", "SkillMeta", "SkillStats", "FunctionSkill",
    "SkillRegistry", "SkillVariant",
    "ReplayReaderSkill", "OpponentModelerSkill", "MapAnalyzerSkill",
    # MCP Tools
    "MCPTool", "MCPToolRegistry", "MCPServer",
    "ReadReplayTool", "AnalyzeGameTool", "QueryHistoryTool",
    # Tracking
    "Run",
    "StepRecord", "EpisodeRecord", "EvalRecord",
    "ResourceRecord", "LogRecord", "RunMeta",
    "JSONLWriter", "ResourceSampler",
    "TrackedEnv", "TimedAgent",
    # Runners
    "BaseRunner", "BaseRLRunner", "BaseRuleRunner", "BaseEvalRunner",
    # Report
    "ReportBuilder", "build_report",
]

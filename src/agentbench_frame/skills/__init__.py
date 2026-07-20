"""
Skills system for composable, trackable agent capabilities.

Skills are modular strategy components with:
- Metadata (name, version, description, tags)
- Activation conditions
- Execution logic
- Performance tracking
- MCP bridge for external tool access

Built-in skills:
- ReplayReaderSkill: Analyze opponent replay files
- OpponentModelerSkill: Build behavioral models of opponents
- MapAnalyzerSkill: Strategic map analysis
"""

from agentbench_frame.skills.base import Skill, SkillMeta, SkillStats, FunctionSkill
from agentbench_frame.skills.registry import SkillRegistry, SkillVariant
from agentbench_frame.skills.replay_reader import ReplayReaderSkill
from agentbench_frame.skills.opponent_modeler import OpponentModelerSkill
from agentbench_frame.skills.map_analyzer import MapAnalyzerSkill

__all__ = [
    "Skill",
    "SkillMeta",
    "SkillStats",
    "FunctionSkill",
    "SkillRegistry",
    "SkillVariant",
    "ReplayReaderSkill",
    "OpponentModelerSkill",
    "MapAnalyzerSkill",
]

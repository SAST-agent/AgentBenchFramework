"""
Skill Registry — discovery, versioning, and performance analysis for skills.

Manages:
- Registration and discovery of skills
- Performance tracking across episodes
- A/B testing (compare two versions of a skill)
- Export as MCP tool catalog
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
import time
import json

from agentbench_frame.skills.base import Skill, SkillMeta, SkillStats


@dataclass
class SkillVariant:
    """A variant of a skill for A/B testing."""
    skill: Skill
    weight: float = 1.0  # selection weight
    stats: SkillStats = field(default_factory=SkillStats)
    is_active: bool = True


class SkillRegistry:
    """
    Central registry for skills.

    Supports:
    - Registration by name
    - Versioning (multiple versions of the same skill)
    - A/B testing between variants
    - Performance tracking
    - MCP tool catalog export

    Example:
        registry = SkillRegistry()
        registry.register(ReplayReaderSkill())
        registry.register(OpponentModelerSkill())

        # Get a skill by name
        skill = registry.get("replay_reader")

        # Enable A/B testing
        registry.add_variant("replay_reader", ReplayReaderSkillV2(), weight=0.3)
    """

    def __init__(self):
        self._skills: Dict[str, List[SkillVariant]] = {}
        self._history: List[Dict[str, Any]] = []
        self._global_context: Dict[str, Any] = {}

    # ---- Registration ----

    def register(self, skill: Skill, weight: float = 1.0):
        """Register a skill. Multiple versions of the same name are treated as variants."""
        name = skill.meta.name
        variant = SkillVariant(skill=skill, weight=weight)

        if name not in self._skills:
            self._skills[name] = []
        self._skills[name].append(variant)

    def unregister(self, name: str, version: Optional[str] = None):
        """Remove a skill or specific version."""
        if name not in self._skills:
            return
        if version:
            self._skills[name] = [
                v for v in self._skills[name]
                if v.skill.meta.version != version
            ]
            if not self._skills[name]:
                del self._skills[name]
        else:
            del self._skills[name]

    # ---- Retrieval ----

    def get(self, name: str) -> Optional[Skill]:
        """Get the best (highest weight active) variant of a skill."""
        if name not in self._skills:
            return None
        active = [v for v in self._skills[name] if v.is_active]
        if not active:
            return None

        # If only one, return it
        if len(active) == 1:
            return active[0].skill

        # Weighted random selection for A/B testing
        import random
        total = sum(v.weight for v in active)
        r = random.random() * total
        cumulative = 0
        for v in active:
            cumulative += v.weight
            if r <= cumulative:
                return v.skill
        return active[-1].skill

    def get_all(self, name: str) -> List[Skill]:
        """Get all variants of a skill."""
        if name not in self._skills:
            return []
        return [v.skill for v in self._skills[name]]

    def get_by_tag(self, tag: str) -> List[Skill]:
        """Get all skills with a given tag."""
        result = []
        for variants in self._skills.values():
            for v in variants:
                if tag in v.skill.meta.tags:
                    result.append(v.skill)
        return result

    def get_by_game(self, game: str) -> List[Skill]:
        """Get all skills for a specific game."""
        result = []
        for variants in self._skills.values():
            for v in variants:
                if v.skill.meta.game == game:
                    result.append(v.skill)
        return result

    # ---- A/B Testing ----

    def add_variant(self, name: str, skill: Skill, weight: float = 0.5):
        """Add a variant of an existing skill for A/B testing."""
        # Mark existing variants with reduced weight
        if name in self._skills:
            total_existing = sum(v.weight for v in self._skills[name])
            scale = (1.0 - weight) / max(0.001, total_existing)
            for v in self._skills[name]:
                v.weight *= scale

        self.register(skill, weight=weight)

    def set_variant_weight(self, name: str, version: str, weight: float):
        """Adjust weight of a specific variant."""
        if name not in self._skills:
            return
        for v in self._skills[name]:
            if v.skill.meta.version == version:
                v.weight = weight
                return

    def get_ab_report(self, name: str) -> Dict[str, Any]:
        """Get A/B testing comparison report for a skill."""
        if name not in self._skills:
            return {}

        variants = self._skills[name]
        report = {
            "skill_name": name,
            "num_variants": len(variants),
            "variants": [],
        }

        for v in variants:
            report["variants"].append({
                "version": v.skill.meta.version,
                "weight": v.weight,
                "is_active": v.is_active,
                "stats": v.stats.to_dict(),
            })

        return report

    # ---- Performance Tracking ----

    def record(self, name: str, version: str,
               success: bool, reward: float = 0.0,
               decision_time_ms: float = 0.0,
               opponent: str = ""):
        """Record a skill activation for performance tracking."""
        if name not in self._skills:
            return
        for v in self._skills[name]:
            if v.skill.meta.version == version:
                v.skill.record_activation(success, reward, decision_time_ms, opponent)
                v.stats = v.skill.stats
                break

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance stats for all skills or a specific one."""
        if name:
            if name not in self._skills:
                return {}
            return {
                version: v.stats.to_dict()
                for v in self._skills[name]
                for version in [v.skill.meta.version]
            }

        return {
            skill_name: {
                v.skill.meta.version: v.stats.to_dict()
                for v in variants
            }
            for skill_name, variants in self._skills.items()
        }

    def get_best_variant(self, name: str) -> Optional[str]:
        """Return the version string of the best-performing variant."""
        if name not in self._skills:
            return None
        best = None
        best_rate = -1.0
        for v in self._skills[name]:
            if v.stats.success_rate > best_rate:
                best_rate = v.stats.success_rate
                best = v.skill.meta.version
        return best

    # ---- MCP Integration ----

    def export_mcp_catalog(self) -> List[Dict[str, Any]]:
        """Export all registered skills as an MCP tool catalog."""
        tools = []
        seen = set()
        for name, variants in self._skills.items():
            for v in variants:
                key = f"{name}@{v.skill.meta.version}"
                if key not in seen:
                    tools.append(v.skill.as_mcp_tool_definition())
                    seen.add(key)
        return tools

    # ---- Iteration ----

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all registered skills with metadata."""
        result = []
        for name, variants in self._skills.items():
            for v in variants:
                result.append({
                    "name": name,
                    "version": v.skill.meta.version,
                    "description": v.skill.meta.description,
                    "tags": v.skill.meta.tags,
                    "game": v.skill.meta.game,
                    "weight": v.weight,
                    "is_active": v.is_active,
                    "success_rate": v.stats.success_rate,
                })
        return result

    def save_snapshot(self, path: str):
        """Save registry state to JSON."""
        data = {
            "skills": self.list_skills(),
            "ab_reports": {
                name: self.get_ab_report(name)
                for name in self._skills
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def global_context(self) -> Dict[str, Any]:
        return self._global_context

    def update_context(self, **kwargs):
        """Update global shared context (e.g., replay database path)."""
        self._global_context.update(kwargs)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"SkillRegistry({len(self._skills)} skills, {sum(len(v) for v in self._skills.values())} variants)"

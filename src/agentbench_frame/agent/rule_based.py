"""
Rule-based agent with configurable strategy pipeline.

RuleBasedAgent allows composing multiple strategy rules into a decision pipeline.
Each rule is a function that takes (observation, state) and returns either an action
or None (delegating to the next rule). This enables:
- Priority-based decision making (try aggressive move, fall back to defensive)
- Modular strategy composition
- Easy rule iteration and A/B testing

Skills integration:
- Skills are more structured than raw rules (metadata, tracking, MCP bridge)
- Skills run BEFORE rules: info-gathering skills update the context
- Strategy skills can produce actions directly
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import time

from agentbench_frame.agent.base import BaseAgent
from agentbench_frame.skills.base import Skill
from agentbench_frame.skills.registry import SkillRegistry


# Rule function signature: (observation, agent_state) -> Optional[action]
RuleFunc = Callable[[Dict[str, Any], Dict[str, Any]], Optional[Any]]


class RuleBasedAgent(BaseAgent):
    """
    An agent that makes decisions using a configurable pipeline of rules and skills.

    Execution order:
    1. Info skills (can_activate → execute, store results in context)
    2. Strategy rules (first non-None action wins)
    3. Default action (if no rule fired)

    Skills provide:
    - Structured metadata and versioning
    - Performance tracking per skill
    - A/B testing via SkillRegistry
    - MCP bridge for external tool access

    Example:
        # With rules only
        agent = RuleBasedAgent(
            name="AggressiveGenerals",
            rules=[attack_rule, reinforce_rule, expand_rule, end_turn_rule],
        )

        # With skills + rules
        from agentbench_frame.skills import ReplayReaderSkill, MapAnalyzerSkill
        agent = RuleBasedAgent(
            name="SkilledGenerals",
            skills=[ReplayReaderSkill(), MapAnalyzerSkill()],
            rules=[attack_rule, expand_rule, end_turn_rule],
        )
    """

    def __init__(self,
                 name: str = "RuleBasedAgent",
                 rules: Optional[List[RuleFunc]] = None,
                 skills: Optional[List[Skill]] = None,
                 skill_registry: Optional[SkillRegistry] = None,
                 default_action: Any = None):
        """
        Args:
            name: Agent identifier
            rules: List of rule functions in priority order
            skills: List of skills (run before rules for info gathering)
            skill_registry: Shared skill registry for A/B testing
            default_action: Action taken when no rule fires
        """
        super().__init__(name=name)
        self.rules = rules or []
        self.skills = skills or []
        self.skill_registry = skill_registry
        self.default_action = default_action or [[8]]  # end turn
        self._state: Dict[str, Any] = {}
        self._stats: Dict[str, int] = {"calls": 0, "rule_hits": {}, "skill_activations": {}}
        self._skill_timing: Dict[str, float] = {}

    # ---- Rule management ----

    def add_rule(self, rule: RuleFunc, priority: Optional[int] = None):
        """Add a rule to the pipeline."""
        if priority is not None:
            self.rules.insert(priority, rule)
        else:
            self.rules.append(rule)

    def remove_rule(self, rule_index: int):
        """Remove a rule by index."""
        if 0 <= rule_index < len(self.rules):
            self.rules.pop(rule_index)

    # ---- Skill management ----

    def add_skill(self, skill: Skill):
        """Add a skill to the agent."""
        self.skills.append(skill)

    def remove_skill(self, skill_name: str):
        """Remove a skill by name."""
        self.skills = [s for s in self.skills if s.meta.name != skill_name]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        # First check skill registry (for A/B testing)
        if self.skill_registry and name in self.skill_registry:
            return self.skill_registry.get(name)
        # Then local skills
        for s in self.skills:
            if s.meta.name == name:
                return s
        return None

    # ---- Main decision loop ----

    def act(self, observation: Dict[str, Any]) -> Any:
        """
        Execute skills for info gathering, then evaluate rules for action.

        1. Phase 1 — Skills: info-gathering skills that update context
        2. Phase 2 — Rules: strategy rules in priority order
        3. Phase 3 — Default: end turn if nothing fired
        """
        self._stats["calls"] += 1

        # Phase 1: Run info-gathering skills
        self._run_skills(observation)

        # Phase 2: Evaluate rules in priority order
        action = self._run_rules(observation)

        # Phase 3: Default
        if action is None:
            action = self.default_action

        return action

    def _run_skills(self, observation: Dict[str, Any]):
        """Execute all skills that want to activate. Store results in context."""
        # Resolve skills from registry (for A/B testing)
        active_skills = []
        for skill in self.skills:
            name = skill.meta.name
            if self.skill_registry and name in self.skill_registry:
                resolved = self.skill_registry.get(name)
                if resolved:
                    active_skills.append(resolved)
                else:
                    active_skills.append(skill)
            else:
                active_skills.append(skill)

        for skill in active_skills:
            if skill.can_activate(observation, self._state):
                start = time.time()
                try:
                    result = skill.execute(observation, self._state)
                    elapsed = (time.time() - start) * 1000

                    # Track activation
                    skill_name = skill.meta.name
                    self._stats["skill_activations"][skill_name] = \
                        self._stats["skill_activations"].get(skill_name, 0) + 1
                    self._skill_timing[skill_name] = elapsed

                    # Report to registry
                    if self.skill_registry:
                        self.skill_registry.record(
                            name=skill.meta.name,
                            version=skill.meta.version,
                            success=True,
                            reward=0.0,
                            decision_time_ms=elapsed,
                        )

                except Exception as e:
                    skill_name = skill.meta.name
                    self._state[f"_skill_error_{skill_name}"] = str(e)

                    if self.skill_registry:
                        self.skill_registry.record(
                            name=skill.meta.name,
                            version=skill.meta.version,
                            success=False,
                        )

    def _run_rules(self, observation: Dict[str, Any]) -> Optional[Any]:
        """Evaluate rules in priority order. Return first non-None action."""
        for i, rule in enumerate(self.rules):
            try:
                action = rule(observation, self._state)
                if action is not None:
                    rule_name = getattr(rule, "__name__", f"rule_{i}")
                    self._stats["rule_hits"][rule_name] = \
                        self._stats["rule_hits"].get(rule_name, 0) + 1
                    return action
            except Exception:
                continue

        # Check if any skill produced a direct action
        action_key = "_skill_action"
        if action_key in self._state:
            return self._state.pop(action_key)

        return None

    def reset(self):
        """Reset agent state for new episode."""
        self._state = {}
        # Reset all skills
        for skill in self.skills:
            skill.reset()
        # Also reset skills from registry
        if self.skill_registry:
            for skill_info in self.skill_registry.list_skills():
                skill = self.skill_registry.get(skill_info["name"])
                if skill:
                    skill.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Return rule and skill hit statistics for analysis."""
        return {
            "calls": self._stats["calls"],
            "rule_hits": self._stats["rule_hits"],
            "skill_activations": self._stats["skill_activations"],
            "skill_timing_ms": self._skill_timing,
        }

    def get_skill_stats(self) -> Dict[str, Any]:
        """Get detailed skill performance from the registry."""
        if self.skill_registry:
            return self.skill_registry.get_stats()
        return {
            s.meta.name: s.stats.to_dict() for s in self.skills
        }

    def on_episode_end(self, winner: int, reward: float):
        """Notify skills that the episode ended."""
        for skill in self.skills:
            skill.on_episode_end({}, winner, reward)


# ---- Built-in Rule Templates for Generals ----

def expand_rule(obs: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    Expand territory by moving armies from strong cells to adjacent neutral cells.

    For Generals: finds owned cells with army > 1 adjacent to neutral cells,
    and moves half the army there.
    """
    if not obs.get("state"):
        return None

    gs = obs["state"]
    board = gs.get("board", [])
    player = obs.get("player_id", 0)
    actions = []

    for r in range(len(board)):
        for c in range(len(board[r])):
            cell = board[r][c]
            if cell["player"] != player or cell["army"] <= 1:
                continue

            # Check adjacent cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                    neighbor = board[nr][nc]
                    # Target neutral or enemy cells
                    if neighbor["player"] != player:
                        direction = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}[(dr, dc)]
                        amount = max(1, cell["army"] // 2)
                        actions.append([1, r, c, direction, amount])

    if actions:
        actions.append([8])  # end turn
        return actions
    return None


def reinforce_rule(obs: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    Reinforce front-line cells by moving armies from rear cells.

    Finds cells bordering enemy territory and moves reinforcements to them.
    """
    if not obs.get("state"):
        return None

    gs = obs["state"]
    board = gs.get("board", [])
    player = obs.get("player_id", 0)
    actions = []

    if len(board) == 0:
        return None

    # Find front-line cells (friendly cells adjacent to enemy)
    frontline = set()
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c]["player"] == player:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                        if board[nr][nc]["player"] == 1 - player:
                            frontline.add((r, c))
                            break

    # Move reinforcements from nearby strong cells to frontline
    for fr, fc in frontline:
        front_cell = board[fr][fc]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                neighbor = board[nr][nc]
                if neighbor["player"] == player and neighbor["army"] > 2:
                    direction = {(1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4}[(dr, dc)]
                    # Reverse direction to move FROM rear TO front
                    rev_dir = {1: 2, 2: 1, 3: 4, 4: 3}[direction]
                    actions.append([1, nr, nc, rev_dir, neighbor["army"] // 2])
                    break

    if actions:
        actions.append([8])
        return actions
    return None


def upgrade_rule(obs: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    Upgrade generals' production when we have enough coins.
    """
    if not obs.get("state"):
        return None

    gs = obs["state"]
    generals = gs.get("generals", [])
    coins = gs.get("coins", [0, 0])
    player = obs.get("player_id", 0)
    actions = []

    for gen in generals:
        if gen["player"] != player:
            continue
        # Upgrade production if affordable
        cost = 10 * gen["produce_level"]
        if coins[player] >= cost and gen["produce_level"] < 3:
            actions.append([3, gen["id"], 1])  # upgrade production

    if actions:
        actions.append([8])
        return actions
    return None


def attack_rule(obs: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    Attack enemy cells when we have overwhelming force.
    """
    if not obs.get("state"):
        return None

    gs = obs["state"]
    board = gs.get("board", [])
    player = obs.get("player_id", 0)
    actions = []

    if len(board) == 0:
        return None

    for r in range(len(board)):
        for c in range(len(board[r])):
            cell = board[r][c]
            if cell["player"] != player or cell["army"] < 4:
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                    neighbor = board[nr][nc]
                    if neighbor["player"] == 1 - player and cell["army"] > neighbor["army"] + 1:
                        direction = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}[(dr, dc)]
                        amount = min(cell["army"] - 1, neighbor["army"] + 1)
                        actions.append([1, r, c, direction, amount])

    if actions:
        actions.append([8])
        return actions
    return None


def end_turn_rule(obs: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[List[int]]]:
    """Always fires: end the current turn."""
    return [[8]]


# Pre-built agent configurations
def make_expansionist_agent(name: str = "Expansionist") -> RuleBasedAgent:
    """An agent focused on rapid expansion."""
    return RuleBasedAgent(
        name=name,
        rules=[expand_rule, upgrade_rule, end_turn_rule],
    )


def make_aggressive_agent(name: str = "Aggressive") -> RuleBasedAgent:
    """An agent that prioritizes attacking."""
    return RuleBasedAgent(
        name=name,
        rules=[attack_rule, expand_rule, reinforce_rule, upgrade_rule, end_turn_rule],
    )


def make_balanced_agent(name: str = "Balanced") -> RuleBasedAgent:
    """An agent with a balanced strategy."""
    return RuleBasedAgent(
        name=name,
        rules=[reinforce_rule, attack_rule, expand_rule, upgrade_rule, end_turn_rule],
    )

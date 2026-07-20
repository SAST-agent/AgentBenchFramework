"""
Replay Reader Skill

Reads and analyzes game replay files to extract:
- Opponent strategy patterns
- Opening move preferences
- Win/loss conditions
- Resource management style
- Action frequency distributions

Supports replay formats:
- Generals JSON replay (newline-delimited JSON per round)
- LostSpace JSON replay
- Generic Saiblo replay

This skill can be used:
- During a game (to analyze past opponent replays)
- Between games (to scout opponents)
- In training (to generate training data from replays)
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from agentbench_frame.skills.base import Skill, SkillMeta, SkillStats


class ReplayReaderSkill(Skill):
    """
    Reads and analyzes opposing AI replay files to understand their strategy.

    Capabilities:
    - Parse Generals/LostSpace replay formats
    - Extract action frequency distributions
    - Identify opening patterns (first N moves)
    - Detect aggression level
    - Map opponent expansion patterns
    - Generate counter-strategy suggestions

    Usage:
        reader = ReplayReaderSkill(replay_dir="./replays/")
        if reader.can_activate(obs, context):
            analysis = reader.analyze_opponent("OpponentName")
            counter_strategy = reader.suggest_counter(analysis)
    """

    def __init__(self, replay_dir: str = "./replays"):
        meta = SkillMeta(
            name="replay_reader",
            version="1.0.0",
            description="Reads and analyzes game replay files to model opponent strategies",
            tags=["analysis", "replay", "opponent-modeling", "scouting"],
            game="generals",
            dependencies=[],
        )
        super().__init__(meta)
        self.replay_dir = replay_dir
        self._replay_cache: Dict[str, List[Dict]] = {}
        self._opponent_profiles: Dict[str, Dict[str, Any]] = {}
        self._max_cache_size = 50

    # ---- Skill interface ----

    def can_activate(self, observation: Dict[str, Any],
                     context: Dict[str, Any]) -> bool:
        """
        Activate when:
        - We have opponent replays to analyze
        - We haven't already analyzed this opponent recently
        - It's early in the game (first 5 rounds)
        """
        opponent = context.get("opponent_name", "")
        if not opponent:
            return False

        round_num = observation.get("round_num", 0)
        state = observation.get("state", {})
        game_round = state.get("round", round_num)

        # Only activate early to influence strategy
        if game_round > 5:
            return False

        # Check if we already analyzed this opponent
        last_analysis = context.get(f"_replay_analysis_{opponent}_round", -1)
        if last_analysis >= 0 and game_round - last_analysis < 3:
            return False

        # Check if we have replays for this opponent
        return self._has_replays_for(opponent)

    def execute(self, observation: Dict[str, Any],
                context: Dict[str, Any]) -> Optional[Any]:
        """
        Analyze opponent replays and store insights in context.
        Returns None (not an action — this is a pure information skill).
        """
        opponent = context.get("opponent_name", "")
        if not opponent:
            return None

        try:
            profile = self.analyze_opponent(opponent)
            counter = self.suggest_counter(profile)

            # Store analysis in context for other skills to use
            context[f"_replay_analysis_{opponent}"] = profile
            context[f"_replay_analysis_{opponent}_round"] = observation.get("round_num", 0)
            context[f"_counter_strategy_{opponent}"] = counter
            context["_last_opponent_analysis"] = profile

            return None  # Pure information skill, action handled by strategy skills

        except Exception as e:
            context["_replay_analysis_error"] = str(e)
            return None

    # ---- Core analysis ----

    def analyze_opponent(self, opponent_name: str) -> Dict[str, Any]:
        """
        Analyze all available replays for an opponent.

        Returns a profile dict with:
        - action_distribution: frequency of each action type
        - opening_patterns: most common opening sequences
        - aggression_score: 0.0 (passive) to 1.0 (aggressive)
        - expansion_style: "wide", "tall", or "balanced"
        - preferred_tech: most researched technologies
        - win_conditions: how they typically win
        - avg_game_length: average game duration
        - weakness_signals: detected weaknesses
        """
        replays = self._load_opponent_replays(opponent_name)
        if not replays:
            return self._empty_profile(opponent_name)

        profile = {
            "opponent": opponent_name,
            "replays_analyzed": len(replays),
            "action_distribution": self._analyze_actions(replays),
            "opening_patterns": self._analyze_openings(replays),
            "aggression_score": self._analyze_aggression(replays),
            "expansion_style": self._analyze_expansion(replays),
            "preferred_tech": self._analyze_tech_preferences(replays),
            "win_conditions": self._analyze_win_conditions(replays),
            "avg_game_length": self._analyze_game_length(replays),
            "weakness_signals": self._detect_weaknesses(replays),
        }

        # Cache the profile
        self._opponent_profiles[opponent_name] = profile

        # Prune cache
        while len(self._opponent_profiles) > self._max_cache_size:
            oldest = next(iter(self._opponent_profiles))
            del self._opponent_profiles[oldest]

        return profile

    def suggest_counter(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest a counter-strategy based on opponent analysis.

        Returns:
            Dict with recommended strategy adjustments
        """
        if not profile or profile.get("replays_analyzed", 0) == 0:
            return {"strategy": "unknown", "confidence": 0.0}

        suggestions = {
            "strategy": "balanced",
            "confidence": 0.5,
            "adjustments": [],
        }

        aggression = profile.get("aggression_score", 0.5)
        expansion = profile.get("expansion_style", "balanced")
        weaknesses = profile.get("weakness_signals", [])
        openings = profile.get("opening_patterns", {})

        # Counter aggression
        if aggression > 0.7:
            suggestions["strategy"] = "defensive"
            suggestions["confidence"] = 0.7
            suggestions["adjustments"].append({
                "type": "counter_aggression",
                "action": "reinforce_frontline",
                "reason": f"Opponent is highly aggressive (score={aggression:.2f})",
            })
        elif aggression < 0.3:
            suggestions["strategy"] = "aggressive"
            suggestions["confidence"] = 0.6
            suggestions["adjustments"].append({
                "type": "exploit_passive",
                "action": "early_rush",
                "reason": f"Opponent is passive (score={aggression:.2f})",
            })

        # Counter expansion
        if expansion == "wide":
            suggestions["adjustments"].append({
                "type": "counter_wide",
                "action": "concentrate_force",
                "reason": "Opponent spreads thin — use concentrated attacks",
            })
        elif expansion == "tall":
            suggestions["adjustments"].append({
                "type": "counter_tall",
                "action": "surround_and_choke",
                "reason": "Opponent builds tall — cut off their production",
            })

        # Exploit weaknesses
        for weakness in weaknesses:
            suggestions["adjustments"].append({
                "type": "exploit_weakness",
                "weakness": weakness,
                "action": self._weakness_to_action(weakness),
            })

        # Counter common opening
        if openings:
            most_common = max(openings.items(), key=lambda x: x[1])
            suggestions["adjustments"].append({
                "type": "counter_opening",
                "common_opening": most_common[0],
                "frequency": most_common[1],
                "action": "prepare_counter_opening",
            })

        return suggestions

    # ---- Internal analysis methods ----

    def _has_replays_for(self, opponent_name: str) -> bool:
        """Check if we have replay files for an opponent."""
        # Check cache
        if opponent_name in self._replay_cache:
            return len(self._replay_cache[opponent_name]) > 0
        # Check disk
        return len(self._find_replay_files(opponent_name)) > 0

    def _find_replay_files(self, opponent_name: str) -> List[str]:
        """Find replay files for a specific opponent on disk."""
        if not os.path.isdir(self.replay_dir):
            return []

        files = []
        for fname in os.listdir(self.replay_dir):
            if opponent_name.lower() in fname.lower():
                if fname.endswith((".json", ".jsonl", ".replay")):
                    files.append(os.path.join(self.replay_dir, fname))
        return files

    def _load_opponent_replays(self, opponent_name: str) -> List[Dict]:
        """Load all replay data for an opponent."""
        if opponent_name in self._replay_cache:
            return self._replay_cache[opponent_name]

        files = self._find_replay_files(opponent_name)
        replays = []
        for fpath in files:
            try:
                replay = self._parse_replay_file(fpath)
                if replay:
                    replays.append(replay)
            except Exception:
                continue

        self._replay_cache[opponent_name] = replays
        return replays

    def _parse_replay_file(self, path: str) -> Optional[Dict]:
        """
        Parse a replay file into a structured format.

        Supports:
        - Generals: JSON-lines, one object per round
        - Standard: single JSON object/array
        """
        try:
            with open(path, "r") as f:
                content = f.read().strip()

            # Try JSON-lines (Generals format)
            if content.startswith("{") and "\n{" in content:
                lines = content.split("\n")
                rounds = []
                for line in lines:
                    try:
                        rounds.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if rounds:
                    return {
                        "format": "json_lines",
                        "rounds": rounds,
                        "metadata": self._extract_metadata(rounds),
                    }

            # Try single JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return {
                        "format": "json_array",
                        "rounds": data,
                        "metadata": self._extract_metadata(data),
                    }
                elif isinstance(data, dict):
                    return {
                        "format": "json_object",
                        "rounds": [data],
                        "metadata": self._extract_metadata([data]),
                    }
            except json.JSONDecodeError:
                pass

            # Try to extract game data from raw content
            return {
                "format": "raw",
                "rounds": [],
                "metadata": {"raw_content": content[:1000]},
            }

        except Exception:
            return None

    def _extract_metadata(self, rounds: List[Dict]) -> Dict[str, Any]:
        """Extract metadata from parsed rounds."""
        if not rounds:
            return {}

        first = rounds[0]
        last = rounds[-1]

        return {
            "num_rounds": len(rounds),
            "players": first.get("players", first.get("player_list", [])),
            "winner": last.get("winner", -1),
            "final_scores": last.get("end_info", {}),
        }

    def _analyze_actions(self, replays: List[Dict]) -> Dict[str, float]:
        """Analyze action frequency distribution across replays."""
        action_counter = Counter()
        total = 0

        for replay in replays:
            rounds = replay.get("rounds", [])
            for round_data in rounds:
                actions = self._extract_actions_from_round(round_data)
                for action_type in actions:
                    action_counter[action_type] += 1
                    total += 1

        if total == 0:
            return {}

        return {k: v / total for k, v in action_counter.most_common()}

    def _extract_actions_from_round(self, round_data: Dict) -> List[str]:
        """Extract action types from a round of replay data."""
        actions = []

        # Try Generals format (Content field with space-separated commands)
        content = round_data.get("Content", round_data.get("content", ""))
        if isinstance(content, str):
            for line in content.strip().split("\n"):
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    cmd_type = int(parts[0])
                    cmd_names = {
                        1: "move_army", 2: "move_general", 3: "upgrade",
                        4: "skill", 5: "tech", 6: "superweapon",
                        7: "call_sub", 8: "end_turn", 9: "surrender",
                    }
                    actions.append(cmd_names.get(cmd_type, f"cmd_{cmd_type}"))

        # Try action list format
        action_list = round_data.get("action", round_data.get("actions", []))
        if isinstance(action_list, list):
            for action in action_list:
                if isinstance(action, dict):
                    actions.append(action.get("type", "unknown"))
                elif isinstance(action, list) and action:
                    actions.append(str(action[0]))
                else:
                    actions.append(str(action))

        return actions

    def _analyze_openings(self, replays: List[Dict]) -> Dict[str, int]:
        """Analyze opening move patterns (first few rounds)."""
        opening_counter = Counter()

        for replay in replays:
            rounds = replay.get("rounds", [])
            opening_actions = []
            for round_data in rounds[:5]:  # first 5 rounds
                actions = self._extract_actions_from_round(round_data)
                opening_actions.extend(actions)

            if opening_actions:
                key = " -> ".join(opening_actions[:5])  # first 5 actions
                opening_counter[key] += 1

        return dict(opening_counter.most_common(10))

    def _analyze_aggression(self, replays: List[Dict]) -> float:
        """Calculate aggression score from action patterns."""
        aggressive_actions = {"move_army", "skill", "superweapon", "attack"}
        passive_actions = {"upgrade", "tech", "end_turn"}

        total_aggressive = 0
        total_passive = 0

        for replay in replays:
            action_dist = replay.get("_cached_action_dist")
            if not action_dist:
                rounds = replay.get("rounds", [])
                all_actions = []
                for r in rounds:
                    all_actions.extend(self._extract_actions_from_round(r))
                action_counts = Counter(all_actions)
            else:
                action_counts = Counter(action_dist)

            for action, count in action_counts.items():
                if action in aggressive_actions:
                    total_aggressive += count
                elif action in passive_actions:
                    total_passive += count

        total = total_aggressive + total_passive
        if total == 0:
            return 0.5  # neutral default

        return total_aggressive / total

    def _analyze_expansion(self, replays: List[Dict]) -> str:
        """
        Determine expansion style: "wide" (many cells, few army),
        "tall" (few cells, high army), or "balanced".
        """
        # Heuristic based on game behavior
        aggression = self._analyze_aggression(replays)
        action_dist = self._analyze_actions(replays)

        move_pct = action_dist.get("move_army", 0)
        upgrade_pct = action_dist.get("upgrade", 0)

        if move_pct > 0.4:
            return "wide"
        elif upgrade_pct > 0.3:
            return "tall"
        else:
            return "balanced"

    def _analyze_tech_preferences(self, replays: List[Dict]) -> List[str]:
        """Determine preferred technology path."""
        tech_counter = Counter()
        for replay in replays:
            rounds = replay.get("rounds", [])
            for round_data in rounds:
                actions = self._extract_actions_from_round(round_data)
                for a in actions:
                    if a == "tech":
                        tech_counter["tech"] += 1

        return [t for t, _ in tech_counter.most_common(5)]

    def _analyze_win_conditions(self, replays: List[Dict]) -> Dict[str, int]:
        """Analyze how this opponent typically wins."""
        conditions = Counter()
        for replay in replays:
            meta = replay.get("metadata", {})
            winner = meta.get("winner", -1)
            conditions[f"winner_{winner}"] += 1

        return dict(conditions)

    def _analyze_game_length(self, replays: List[Dict]) -> float:
        """Calculate average game length."""
        lengths = []
        for replay in replays:
            meta = replay.get("metadata", {})
            num_rounds = meta.get("num_rounds", len(replay.get("rounds", [])))
            lengths.append(num_rounds)

        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def _detect_weaknesses(self, replays: List[Dict]) -> List[str]:
        """Detect potential weaknesses from replay analysis."""
        weaknesses = []

        action_dist = self._analyze_actions(replays)
        openings = self._analyze_openings(replays)

        # Low tech usage = possibly tech-averse
        if action_dist.get("tech", 0) < 0.05:
            weaknesses.append("low_tech_investment")

        # Predictable openings
        if openings and list(openings.values())[0] > 0.7 * sum(openings.values()):
            weaknesses.append("predictable_opening")

        # Low skill usage
        if action_dist.get("skill", 0) < 0.03:
            weaknesses.append("underuses_skills")

        # Low superweapon usage
        if action_dist.get("superweapon", 0) < 0.01:
            weaknesses.append("ignores_superweapons")

        return weaknesses

    def _weakness_to_action(self, weakness: str) -> str:
        """Convert a weakness into a concrete counter-action."""
        mapping = {
            "low_tech_investment": "rush_tech_advantage",
            "predictable_opening": "use_counter_opening",
            "underuses_skills": "aggressive_skill_use",
            "ignores_superweapons": "rush_superweapon",
        }
        return mapping.get(weakness, "exploit_weakness")

    def _empty_profile(self, opponent_name: str) -> Dict[str, Any]:
        """Return an empty profile for an unknown opponent."""
        return {
            "opponent": opponent_name,
            "replays_analyzed": 0,
            "action_distribution": {},
            "opening_patterns": {},
            "aggression_score": 0.5,
            "expansion_style": "unknown",
            "preferred_tech": [],
            "win_conditions": {},
            "avg_game_length": 0.0,
            "weakness_signals": [],
        }

    def on_episode_end(self, observation: Dict[str, Any],
                       winner: int, reward: float):
        """Save this game's trajectory for future opponent analysis."""
        # Could auto-save the trajectory as a replay file
        pass

    # ---- MCP integration ----

    def as_mcp_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "read_replay",
            "description": f"[Skill v{self.meta.version}] Read and analyze game replay files. "
                          "Extract opponent strategy patterns, openings, aggression, and weaknesses.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "opponent_name": {
                        "type": "string",
                        "description": "Name of the opponent to analyze",
                    },
                    "replay_path": {
                        "type": "string",
                        "description": "Optional path to a specific replay file",
                    },
                },
                "required": ["opponent_name"],
            },
        }

    def call_as_mcp(self, **kwargs) -> Dict[str, Any]:
        opponent = kwargs.get("opponent_name", "")
        replay_path = kwargs.get("replay_path", "")

        if replay_path:
            # Load specific replay
            replay = self._parse_replay_file(replay_path)
            if replay:
                self._replay_cache[f"manual_{opponent}"] = [replay]

        profile = self.analyze_opponent(opponent)
        counter = self.suggest_counter(profile)

        return {
            "profile": profile,
            "counter_strategy": counter,
            "success": True,
        }

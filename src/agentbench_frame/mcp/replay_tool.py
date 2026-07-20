"""
Read Replay MCP Tool

MCP-compatible tool for reading and analyzing game replay files.
Exposes the replay reading capability to LLM agents and external systems.
"""

import os
import json
from typing import Any, Dict, List, Optional

from agentbench_frame.mcp.base import MCPTool


class ReadReplayTool(MCPTool):
    """
    MCP Tool: read_replay

    Reads a game replay file and returns structured analysis including:
    - Game metadata (players, winner, rounds)
    - Action timelines
    - Key game events
    - Player statistics

    This tool can be called by LLM agents to understand past games
    and inform strategy decisions.
    """

    def __init__(self, replay_dir: str = "./replays"):
        super().__init__(
            name="read_replay",
            description="Read and analyze a game replay file. "
                       "Extracts game metadata, player actions, key events, "
                       "and strategic insights from replay data.",
        )
        self.replay_dir = replay_dir

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the replay file to analyze",
                },
                "player_perspective": {
                    "type": "integer",
                    "description": "Analyze from this player's perspective (0 or 1)",
                    "default": 0,
                },
                "include_timeline": {
                    "type": "boolean",
                    "description": "Include full action timeline",
                    "default": False,
                },
                "include_statistics": {
                    "type": "boolean",
                    "description": "Include detailed statistics",
                    "default": True,
                },
            },
            "required": ["path"],
        }

    def call(self, **kwargs) -> Dict[str, Any]:
        path = kwargs.get("path", "")
        player_perspective = kwargs.get("player_perspective", 0)
        include_timeline = kwargs.get("include_timeline", False)
        include_statistics = kwargs.get("include_statistics", True)

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(self.replay_dir, path)

        if not os.path.exists(path):
            return {
                "success": False,
                "error": f"Replay file not found: {path}",
                "replay_dir": self.replay_dir,
            }

        try:
            data = self._parse(path)
            result = {
                "success": True,
                "path": path,
                "file_size": os.path.getsize(path),
            }

            if data:
                result["metadata"] = self._extract_metadata(data)

                if include_statistics:
                    result["statistics"] = self._compute_statistics(
                        data, player_perspective
                    )

                if include_timeline:
                    result["timeline"] = self._extract_timeline(data)

                result["insights"] = self._generate_insights(data, player_perspective)

            else:
                result["error"] = "Could not parse replay file"
                result["success"] = False

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": path,
            }

    def _parse(self, path: str) -> Optional[Any]:
        """Parse a replay file into rounds."""
        with open(path, "r") as f:
            content = f.read().strip()

        if not content:
            return None

        # Try JSON-lines format (common for Saiblo replays)
        lines = content.split("\n")
        rounds = []
        for line in lines:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rounds.append(obj)
            except json.JSONDecodeError:
                continue

        if rounds:
            return rounds

        # Try single JSON
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

        # Return raw content for manual inspection
        return [{"raw": content[:5000]}]

    def _extract_metadata(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract metadata from parsed replay data."""
        first = data[0] if data else {}
        last = data[-1] if data else {}

        return {
            "total_rounds": len(data),
            "game_format": "json_lines" if len(data) > 1 else "single",
            "initial_state_keys": list(first.keys())[:10] if first else [],
            "final_state": {
                k: str(v)[:100]
                for k, v in list(last.items())[:10]
            } if last else {},
        }

    def _compute_statistics(self, data: List[Dict],
                            player: int) -> Dict[str, Any]:
        """Compute player statistics from replay data."""
        stats = {
            "player": player,
            "total_rounds": len(data),
            "actions_by_type": {},
            "territory_changes": [],
        }

        for round_data in data:
            # Extract actions from various possible formats
            actions = []

            # Generals format: Content field
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
                        action_name = cmd_names.get(cmd_type, f"cmd_{cmd_type}")
                        actions.append(action_name)
                        stats["actions_by_type"][action_name] = \
                            stats["actions_by_type"].get(action_name, 0) + 1

            # Round info
            round_num = round_data.get("Round", round_data.get("round", 0))
            player_field = round_data.get("Player", round_data.get("player", -1))

            if player_field == player:
                stats.setdefault("my_rounds", 0)
                stats["my_rounds"] += 1

        return stats

    def _extract_timeline(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract a simplified action timeline."""
        timeline = []
        for i, round_data in enumerate(data):
            content = round_data.get("Content", round_data.get("content", ""))
            if isinstance(content, str) and content.strip():
                # First line only for timeline
                first_line = content.strip().split("\n")[0]
                timeline.append({
                    "round": i + 1,
                    "action": first_line[:100],
                })
        return timeline[:50]  # Limit to 50 entries

    def _generate_insights(self, data: List[Dict],
                           player: int) -> List[str]:
        """Generate strategic insights from replay data."""
        insights = []

        total_rounds = len(data)
        if total_rounds < 10:
            insights.append("Very short game — likely early rush or quick error")
        elif total_rounds > 400:
            insights.append("Long game — both players played conservatively")
        else:
            insights.append(f"Average-length game ({total_rounds} rounds)")

        # Check for varied action usage
        stats = self._compute_statistics(data, player)
        action_count = len(stats.get("actions_by_type", {}))
        if action_count <= 2:
            insights.append("Limited action variety — predictable strategy")
        elif action_count >= 5:
            insights.append("Diverse action usage — adaptable player")

        return insights

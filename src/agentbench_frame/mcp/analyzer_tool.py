"""
Analyze Game MCP Tool

MCP-compatible tool for real-time game state analysis.
Can be called mid-game by LLM agents to get strategic advice.
"""

import json
from typing import Any, Dict, List, Optional

from agentbench_frame.mcp.base import MCPTool


class AnalyzeGameTool(MCPTool):
    """
    MCP Tool: analyze_game

    Analyzes a game state snapshot and returns strategic analysis including:
    - Position evaluation (who is winning?)
    - Recommended actions
    - Threat assessment
    - Resource efficiency analysis

    Accepts game state JSON or an observation dictionary.
    """

    def __init__(self):
        super().__init__(
            name="analyze_game",
            description="Analyze a game state snapshot and provide strategic advice. "
                       "Evaluates position, identifies threats, and recommends actions.",
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "game_state": {
                    "type": "object",
                    "description": "The game state to analyze (observation dict)",
                },
                "player": {
                    "type": "integer",
                    "description": "Player to analyze for (0 or 1)",
                    "default": 0,
                },
                "analysis_depth": {
                    "type": "string",
                    "enum": ["quick", "normal", "deep"],
                    "description": "Depth of analysis",
                    "default": "normal",
                },
            },
            "required": ["game_state"],
        }

    def call(self, **kwargs) -> Dict[str, Any]:
        game_state = kwargs.get("game_state", {})
        player = kwargs.get("player", 0)
        depth = kwargs.get("analysis_depth", "normal")

        if not game_state:
            return {"success": False, "error": "No game state provided"}

        # Handle nested observation format
        if "state" in game_state:
            game_state = game_state["state"]

        try:
            analysis = {
                "success": True,
                "player": player,
                "depth": depth,
                "position_evaluation": self._evaluate_position(game_state, player),
                "recommendations": self._recommend_actions(game_state, player),
                "threats": self._assess_threats(game_state, player),
                "resources": self._analyze_resources(game_state, player),
            }

            if depth in ("normal", "deep"):
                analysis["territory_analysis"] = self._analyze_territory(
                    game_state, player
                )

            if depth == "deep":
                analysis["detailed_metrics"] = self._compute_metrics(
                    game_state, player
                )

            return analysis

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _evaluate_position(self, state: Dict, player: int) -> Dict[str, Any]:
        """Evaluate who is winning and by how much."""
        board = state.get("board", [])
        coins = state.get("coins", [0, 0])
        opponent = 1 - player

        my_cells = 0
        my_army = 0
        opp_cells = 0
        opp_army = 0

        for row in board:
            for cell in row:
                if cell.get("player") == player:
                    my_cells += 1
                    my_army += cell.get("army", 0)
                elif cell.get("player") == opponent:
                    opp_cells += 1
                    opp_army += cell.get("army", 0)

        advantage = "even"
        score_diff = (my_cells * 2 + my_army) - (opp_cells * 2 + opp_army)

        if score_diff > 20:
            advantage = "winning"
        elif score_diff > 5:
            advantage = "slight_advantage"
        elif score_diff < -20:
            advantage = "losing"
        elif score_diff < -5:
            advantage = "slight_disadvantage"

        return {
            "advantage": advantage,
            "score_differential": score_diff,
            "my_cells": my_cells,
            "my_army": my_army,
            "opponent_cells": opp_cells,
            "opponent_army": opp_army,
            "coin_advantage": coins[player] - coins[opponent] if len(coins) > 1 else 0,
        }

    def _recommend_actions(self, state: Dict, player: int) -> List[Dict[str, Any]]:
        """Recommend strategic actions based on game state."""
        evaluation = self._evaluate_position(state, player)
        board = state.get("board", [])
        recommendations = []

        advantage = evaluation["advantage"]

        if advantage in ("losing", "slight_disadvantage"):
            recommendations.append({
                "priority": "high",
                "action": "consolidate_defense",
                "reason": "Behind in position — focus on defending key territory",
            })
            recommendations.append({
                "priority": "medium",
                "action": "seek_tech_advantage",
                "reason": "Tech upgrades can help overcome position deficit",
            })

        if advantage in ("winning", "slight_advantage"):
            recommendations.append({
                "priority": "high",
                "action": "press_advantage",
                "reason": "Ahead in position — apply pressure to close out the game",
            })
            recommendations.append({
                "priority": "medium",
                "action": "deny_resources",
                "reason": "Prevent opponent from recovering",
            })

        if advantage == "even":
            recommendations.append({
                "priority": "high",
                "action": "balanced_development",
                "reason": "Even position — develop economy while scouting",
            })

        # Check for general safety
        for row in board:
            for cell in row:
                if (cell.get("has_general") and cell.get("player") == player
                        and cell.get("army", 0) < 3):
                    recommendations.append({
                        "priority": "critical",
                        "action": "reinforce_general",
                        "reason": f"General at {cell.get('position')} is vulnerable",
                    })

        return recommendations

    def _assess_threats(self, state: Dict, player: int) -> List[Dict[str, Any]]:
        """Identify immediate threats."""
        board = state.get("board", [])
        opponent = 1 - player
        threats = []

        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0

        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if cell.get("player") != player:
                    continue

                my_army = cell.get("army", 0)

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = board[nr][nc]
                        if neighbor.get("player") == opponent:
                            enemy_army = neighbor.get("army", 0)
                            if enemy_army >= my_army:
                                threats.append({
                                    "position": (r, c),
                                    "threat_from": (nr, nc),
                                    "my_army": my_army,
                                    "enemy_army": enemy_army,
                                    "severity": "critical" if enemy_army > my_army + 2
                                                else "warning",
                                    "has_general": cell.get("has_general", False),
                                })

        return sorted(threats, key=lambda t: t["enemy_army"] - t["my_army"], reverse=True)[:5]

    def _analyze_resources(self, state: Dict, player: int) -> Dict[str, Any]:
        """Analyze resource efficiency."""
        coins = state.get("coins", [0, 0])
        generals = state.get("generals", [])
        opponent = 1 - player

        my_generals = [g for g in generals if g.get("player") == player]
        total_production = sum(g.get("produce_level", 0) for g in my_generals)

        return {
            "coins": coins[player] if len(coins) > player else 0,
            "coin_advantage": (coins[player] - coins[opponent])
                              if len(coins) > 1 else 0,
            "num_generals": len(my_generals),
            "total_production": total_production,
            "estimated_income": total_production + 1,
        }

    def _analyze_territory(self, state: Dict, player: int) -> Dict[str, Any]:
        """Analyze territory control."""
        board = state.get("board", [])
        opponent = 1 - player

        my_frontier = 0  # border with neutral
        my_frontline = 0  # border with enemy
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0

        for r in range(rows):
            for c in range(cols):
                if board[r][c].get("player") != player:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = board[nr][nc]
                        if neighbor.get("player") == -1:
                            my_frontier += 1
                            break
                        elif neighbor.get("player") == opponent:
                            my_frontline += 1
                            break

        return {
            "frontier_length": my_frontier,
            "frontline_length": my_frontline,
            "expansion_potential": "high" if my_frontier > my_frontline
                                   else "limited",
        }

    def _compute_metrics(self, state: Dict, player: int) -> Dict[str, Any]:
        """Compute detailed metrics for deep analysis."""
        board = state.get("board", [])
        opponent = 1 - player

        # Count cell types controlled
        terrain_control = {"plain": 0, "bog": 0, "mountain": 0}
        for row in board:
            for cell in row:
                if cell.get("player") == player:
                    cell_type = cell.get("type", 0)
                    type_name = ["plain", "bog", "mountain"][cell_type] if cell_type < 3 else "unknown"
                    terrain_control[type_name] = terrain_control.get(type_name, 0) + 1

        return {
            "terrain_control": terrain_control,
            "terrain_efficiency": (
                terrain_control.get("plain", 0) /
                max(1, sum(terrain_control.values()))
            ),
        }

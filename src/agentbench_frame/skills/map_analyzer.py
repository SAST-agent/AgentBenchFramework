"""
Map Analyzer Skill

Analyzes the game map to provide strategic intelligence:
- Terrain distribution (plain, bog, mountain ratios)
- Key strategic positions (chokepoints, high-ground, resource-rich areas)
- Expansion paths and front-line analysis
- Distance and connectivity calculations
- Danger assessment (weak borders, exposed generals)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

from agentbench_frame.skills.base import Skill, SkillMeta


class MapAnalyzerSkill(Skill):
    """
    Analyzes the game map for strategic decision-making.

    Provides:
    - Terrain analysis
    - Chokepoint detection
    - Front-line identification
    - Expansion path suggestions
    - Danger zone detection
    - Distance maps
    """

    def __init__(self):
        meta = SkillMeta(
            name="map_analyzer",
            version="1.0.0",
            description="Analyzes the game map for strategic intelligence",
            tags=["analysis", "map", "strategy", "pathfinding"],
            game="generals",
            dependencies=[],
        )
        super().__init__(meta)
        self._analysis_cache: Dict[int, Dict[str, Any]] = {}

    def can_activate(self, observation: Dict[str, Any],
                     context: Dict[str, Any]) -> bool:
        """Always available — provides strategic map analysis."""
        return True

    def execute(self, observation: Dict[str, Any],
                context: Dict[str, Any]) -> Optional[Any]:
        """
        Analyze the map and store results in context.
        """
        state = observation.get("state", {})
        board = state.get("board", [])

        if not board:
            return None

        round_num = observation.get("round_num", 0)
        player = observation.get("player_id", 0)

        # Cache analysis per round to avoid redundant computation
        if round_num in self._analysis_cache:
            analysis = self._analysis_cache[round_num]
        else:
            analysis = self._analyze(board, player)
            self._analysis_cache[round_num] = analysis
            # Keep cache small
            if len(self._analysis_cache) > 10:
                oldest = min(self._analysis_cache.keys())
                del self._analysis_cache[oldest]

        # Store in context
        context["map_analysis"] = analysis
        context["frontline"] = analysis.get("frontline", [])
        context["chokepoints"] = analysis.get("chokepoints", [])
        context["expansion_paths"] = analysis.get("expansion_paths", [])
        context["danger_zones"] = analysis.get("danger_zones", [])

        return None  # Pure analysis skill

    def _analyze(self, board: List[List[Dict]], player: int) -> Dict[str, Any]:
        """Run full map analysis."""
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0

        terrain = self._analyze_terrain(board)
        frontline = self._find_frontline(board, player)
        chokepoints = self._find_chokepoints(board)
        expansion = self._find_expansion_paths(board, player)
        danger = self._find_danger_zones(board, player)
        strategic = self._find_strategic_positions(board, player)

        return {
            "dimensions": (rows, cols),
            "terrain": terrain,
            "frontline": frontline,
            "chokepoints": chokepoints,
            "expansion_paths": expansion,
            "danger_zones": danger,
            "strategic_positions": strategic,
            "summary": self._generate_summary(terrain, frontline, danger),
        }

    def _analyze_terrain(self, board: List[List[Dict]]) -> Dict[str, Any]:
        """Analyze terrain distribution."""
        counts = {"plain": 0, "bog": 0, "mountain": 0}
        for row in board:
            for cell in row:
                cell_type = cell.get("type", 0)
                if cell_type == 0:
                    counts["plain"] += 1
                elif cell_type == 1:
                    counts["bog"] += 1
                elif cell_type == 2:
                    counts["mountain"] += 1

        total = max(1, sum(counts.values()))
        return {
            "counts": counts,
            "ratios": {k: v / total for k, v in counts.items()},
            "total_cells": total,
        }

    def _find_frontline(self, board: List[List[Dict]],
                        player: int) -> List[Tuple[int, int]]:
        """Find cells on the border between friendly and enemy territory."""
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        frontline = []

        for r in range(rows):
            for c in range(cols):
                if board[r][c].get("player") == player:
                    # Check if adjacent to enemy
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor = board[nr][nc]
                            if (neighbor.get("player") == 1 - player or
                                    neighbor.get("player") == -1):
                                frontline.append((r, c))
                                break

        return frontline

    def _find_chokepoints(self, board: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        Find chokepoints — narrow passages between mountains.

        A chokepoint is a plain/bog cell with at least 2 mountain neighbors
        and limited passage directions.
        """
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        chokepoints = []

        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if cell.get("type") == 2:  # mountain
                    continue

                # Count passable neighbors and mountain neighbors
                mountain_neighbors = 0
                passable_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = board[nr][nc]
                        if neighbor.get("type") == 2:
                            mountain_neighbors += 1
                        else:
                            passable_neighbors += 1

                # Chokepoint: surrounded by mountains on 2+ sides, limited passage
                if mountain_neighbors >= 2 and passable_neighbors <= 2:
                    chokepoints.append({
                        "position": (r, c),
                        "mountain_neighbors": mountain_neighbors,
                        "passable_neighbors": passable_neighbors,
                        "player": cell.get("player", -1),
                    })

        return chokepoints

    def _find_expansion_paths(self, board: List[List[Dict]],
                              player: int) -> List[Dict[str, Any]]:
        """Find promising expansion directions from friendly territory."""
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        paths = []

        # Find neutral cells adjacent to friendly territory, sorted by safety
        candidates = []
        for r in range(rows):
            for c in range(cols):
                if board[r][c].get("player") == player:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor = board[nr][nc]
                            if neighbor.get("player") == -1:
                                # Score based on terrain quality and safety
                                terrain = neighbor.get("type", 0)
                                score = 3 if terrain == 0 else (1 if terrain == 1 else 0)

                                # Count friendly neighbors (more = safer)
                                friendly_nearby = 0
                                enemy_nearby = 0
                                for d2r, d2c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    n2r, n2c = nr + d2r, nc + d2c
                                    if 0 <= n2r < rows and 0 <= n2c < cols:
                                        p2 = board[n2r][n2c].get("player")
                                        if p2 == player:
                                            friendly_nearby += 1
                                        elif p2 == 1 - player:
                                            enemy_nearby += 1

                                score += friendly_nearby * 1 - enemy_nearby * 2

                                candidates.append({
                                    "target": (nr, nc),
                                    "from": (r, c),
                                    "direction": {(1, 0): "down", (-1, 0): "up",
                                                  (0, 1): "right", (0, -1): "left"}.get((dr, dc), "?"),
                                    "score": score,
                                    "terrain": ["plain", "bog", "mountain"][terrain],
                                })

        # Sort by score and deduplicate targets
        candidates.sort(key=lambda x: x["score"], reverse=True)
        seen = set()
        for c in candidates:
            if c["target"] not in seen:
                paths.append(c)
                seen.add(c["target"])

        return paths[:10]  # Top 10 expansion paths

    def _find_danger_zones(self, board: List[List[Dict]],
                           player: int) -> List[Dict[str, Any]]:
        """Find friendly cells that are in danger of being captured."""
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        danger = []

        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if cell.get("player") != player:
                    continue

                # Check if adjacent to a stronger enemy force
                my_army = cell.get("army", 0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = board[nr][nc]
                        if neighbor.get("player") == 1 - player:
                            enemy_army = neighbor.get("army", 0)
                            if enemy_army > my_army:
                                danger.append({
                                    "position": (r, c),
                                    "my_army": my_army,
                                    "enemy_army": enemy_army,
                                    "threat_from": (nr, nc),
                                    "threat_ratio": enemy_army / max(1, my_army),
                                    "has_general": cell.get("has_general", False),
                                })

        # Sort by most dangerous first
        danger.sort(key=lambda x: x["threat_ratio"], reverse=True)
        return danger

    def _find_strategic_positions(self, board: List[List[Dict]],
                                  player: int) -> List[Dict[str, Any]]:
        """Find strategically important positions to hold or target."""
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 0
        positions = []

        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]

                # General positions are always strategic
                if cell.get("has_general"):
                    positions.append({
                        "position": (r, c),
                        "type": "general",
                        "player": cell.get("player", -1),
                        "priority": "high",
                    })

                # Cells surrounded by friendly territory (key defensive positions)
                if cell.get("player") == player:
                    friendly_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if board[nr][nc].get("player") == player:
                                friendly_neighbors += 1
                    if friendly_neighbors == 4:
                        positions.append({
                            "position": (r, c),
                            "type": "safe_zone",
                            "player": player,
                            "priority": "medium",
                        })

        return positions

    def _generate_summary(self, terrain: Dict, frontline: List,
                          danger: List) -> Dict[str, Any]:
        """Generate a human-readable map analysis summary."""
        open_ratio = terrain["ratios"]["plain"]
        frontline_len = len(frontline)
        danger_count = len(danger)

        # Overall map assessment
        if open_ratio > 0.5:
            terrain_advice = "Open map — favors fast expansion"
        elif open_ratio > 0.3:
            terrain_advice = "Mixed terrain — use chokepoints for defense"
        else:
            terrain_advice = "Tight map — control key passages"

        # Frontline assessment
        if frontline_len > 10:
            frontline_advice = "Long frontline — consolidate before pushing"
        elif frontline_len > 5:
            frontline_advice = "Moderate frontline — reinforce weak points"
        else:
            frontline_advice = "Short frontline — opportunity for breakthrough"

        # Danger assessment
        if danger_count > 3:
            danger_advice = "Multiple threats — prioritize defense"
        elif danger_count > 0:
            danger_advice = "Minor threats — reinforce vulnerable cells"
        else:
            danger_advice = "No immediate threats — expand safely"

        return {
            "terrain_assessment": terrain_advice,
            "frontline_assessment": frontline_advice,
            "danger_assessment": danger_advice,
            "frontline_length": frontline_len,
            "danger_count": danger_count,
        }

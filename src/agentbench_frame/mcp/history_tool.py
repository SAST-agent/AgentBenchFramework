"""
Query History MCP Tool

MCP-compatible tool for querying agent match history and tournament results.
Provides historical data for opponent scouting and strategy planning.
"""

import os
import json
from typing import Any, Dict, List, Optional
from collections import defaultdict

from agentbench_frame.mcp.base import MCPTool


class QueryHistoryTool(MCPTool):
    """
    MCP Tool: query_history

    Queries historical match data including:
    - Head-to-head records between agents
    - Tournament results and rankings
    - Agent performance trends over time
    - Opponent win rates and statistics
    """

    def __init__(self, data_dir: str = "./training_logs"):
        super().__init__(
            name="query_history",
            description="Query historical match data, head-to-head records, "
                       "tournament results, and agent performance trends.",
        )
        self.data_dir = data_dir
        self._history_cache: Optional[List[Dict]] = None

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["head_to_head", "agent_stats", "tournament",
                            "rankings", "recent_matches", "all"],
                    "description": "Type of historical query",
                    "default": "agent_stats",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Agent name to query for",
                },
                "opponent_name": {
                    "type": "string",
                    "description": "Opponent name (for head_to_head queries)",
                },
                "tournament_name": {
                    "type": "string",
                    "description": "Tournament name (for tournament queries)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 20,
                },
            },
            "required": ["query_type"],
        }

    def call(self, **kwargs) -> Dict[str, Any]:
        query_type = kwargs.get("query_type", "all")
        agent_name = kwargs.get("agent_name", "")
        opponent_name = kwargs.get("opponent_name", "")
        tournament_name = kwargs.get("tournament_name", "")
        limit = kwargs.get("limit", 20)

        history = self._load_history()

        try:
            if query_type == "head_to_head":
                result = self._query_head_to_head(history, agent_name, opponent_name)
            elif query_type == "agent_stats":
                result = self._query_agent_stats(history, agent_name)
            elif query_type == "tournament":
                result = self._query_tournament(history, tournament_name)
            elif query_type == "rankings":
                result = self._query_rankings(history, limit)
            elif query_type == "recent_matches":
                result = self._query_recent(history, agent_name, limit)
            else:
                result = self._query_all(history, agent_name, limit)

            return {"success": True, "query_type": query_type, **result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _load_history(self) -> List[Dict]:
        """Load historical data from disk."""
        if self._history_cache is not None:
            return self._history_cache

        history = []

        if not os.path.isdir(self.data_dir):
            self._history_cache = history
            return history

        for fname in os.listdir(self.data_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.data_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            history.extend(data)
                        elif isinstance(data, dict):
                            history.append(data)
                except (json.JSONDecodeError, IOError):
                    continue

        self._history_cache = history
        return history

    def _query_head_to_head(self, history: List[Dict],
                            agent: str, opponent: str) -> Dict[str, Any]:
        """Query head-to-head record between two agents."""
        games = []
        wins = losses = draws = 0

        for entry in history:
            if isinstance(entry, dict):
                a1 = entry.get("agent", entry.get("agent1", ""))
                a2 = entry.get("opponent", entry.get("agent2", ""))
                winner = entry.get("winner", -1)

                if (a1 == agent and a2 == opponent):
                    games.append(entry)
                    if winner == 0:
                        wins += 1
                    elif winner == 1:
                        losses += 1
                    else:
                        draws += 1
                elif (a2 == agent and a1 == opponent):
                    games.append(entry)
                    if winner == 1:
                        wins += 1
                    elif winner == 0:
                        losses += 1
                    else:
                        draws += 1

        total = max(1, wins + losses + draws)
        return {
            "agent": agent,
            "opponent": opponent,
            "total_games": wins + losses + draws,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total,
            "recent_form": self._recent_form(games, agent, 5),
        }

    def _query_agent_stats(self, history: List[Dict],
                           agent: str) -> Dict[str, Any]:
        """Query statistics for a specific agent."""
        games = []
        opponents = set()
        wins = losses = draws = 0

        for entry in history:
            if isinstance(entry, dict):
                a1 = entry.get("agent", entry.get("agent1", ""))
                a2 = entry.get("opponent", entry.get("agent2", ""))

                if agent in (a1, a2):
                    games.append(entry)
                    opponents.add(a2 if a1 == agent else a1)

                    winner = entry.get("winner", -1)
                    if (a1 == agent and winner == 0) or (a2 == agent and winner == 1):
                        wins += 1
                    elif (a1 == agent and winner == 1) or (a2 == agent and winner == 0):
                        losses += 1
                    else:
                        draws += 1

        total = max(1, wins + losses + draws)

        # Per-opponent breakdown
        per_opponent = {}
        for opp in opponents:
            if opp:
                h2h = self._query_head_to_head(history, agent, opp)
                per_opponent[opp] = {
                    "win_rate": h2h["win_rate"],
                    "games": h2h["total_games"],
                }

        return {
            "agent": agent,
            "total_games": wins + losses + draws,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "overall_win_rate": wins / total,
            "opponents_faced": len(opponents),
            "per_opponent": per_opponent,
        }

    def _query_tournament(self, history: List[Dict],
                          name: str) -> Dict[str, Any]:
        """Query tournament results."""
        matches = []
        for entry in history:
            if isinstance(entry, dict):
                t_name = entry.get("tournament", entry.get("name", ""))
                if name.lower() in t_name.lower():
                    matches.append(entry)

        # Aggregate rankings
        rankings = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
        for match in matches:
            a1 = match.get("agent", match.get("agent1", ""))
            a2 = match.get("opponent", match.get("agent2", ""))
            winner = match.get("winner", -1)

            if a1:
                rankings[a1]["games"] += 1
                if winner == 0:
                    rankings[a1]["wins"] += 1
                else:
                    rankings[a1]["losses"] += 1
            if a2:
                rankings[a2]["games"] += 1
                if winner == 1:
                    rankings[a2]["wins"] += 1
                else:
                    rankings[a2]["losses"] += 1

        # Sort by win rate
        sorted_rankings = sorted(
            rankings.items(),
            key=lambda x: x[1]["wins"] / max(1, x[1]["games"]),
            reverse=True,
        )

        return {
            "tournament": name,
            "matches_found": len(matches),
            "rankings": [
                {
                    "agent": agent,
                    "rank": i + 1,
                    "win_rate": stats["wins"] / max(1, stats["games"]),
                    "games": stats["games"],
                }
                for i, (agent, stats) in enumerate(sorted_rankings)
            ],
        }

    def _query_rankings(self, history: List[Dict],
                        limit: int) -> Dict[str, Any]:
        """Compute global rankings from all historical data."""
        ratings = defaultdict(lambda: {"rating": 1500.0, "games": 0})

        # Simple Elo-like ranking
        for entry in history:
            if not isinstance(entry, dict):
                continue
            a1 = entry.get("agent", entry.get("agent1", ""))
            a2 = entry.get("opponent", entry.get("agent2", ""))
            winner = entry.get("winner", -1)

            if not a1 or not a2:
                continue

            # Update ratings
            for a in (a1, a2):
                if ratings[a]["games"] == 0:
                    ratings[a]["rating"] = 1500.0

            r1 = ratings[a1]["rating"]
            r2 = ratings[a2]["rating"]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

            k = 32.0
            if winner == 0:
                ratings[a1]["rating"] += k * (1.0 - e1)
                ratings[a2]["rating"] += k * (0.0 - e2)
            elif winner == 1:
                ratings[a1]["rating"] += k * (0.0 - e1)
                ratings[a2]["rating"] += k * (1.0 - e2)
            else:
                ratings[a1]["rating"] += k * (0.5 - e1)
                ratings[a2]["rating"] += k * (0.5 - e2)

            ratings[a1]["games"] += 1
            ratings[a2]["games"] += 1

        sorted_ratings = sorted(
            ratings.items(),
            key=lambda x: x[1]["rating"],
            reverse=True,
        )[:limit]

        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "agent": agent,
                    "rating": round(stats["rating"], 0),
                    "games": stats["games"],
                }
                for i, (agent, stats) in enumerate(sorted_ratings)
            ],
        }

    def _query_recent(self, history: List[Dict],
                      agent: str, limit: int) -> Dict[str, Any]:
        """Query recent matches for an agent."""
        matches = []
        for entry in reversed(history):
            if isinstance(entry, dict):
                if agent in (entry.get("agent", ""), entry.get("agent1", ""),
                            entry.get("opponent", ""), entry.get("agent2", "")):
                    matches.append(entry)
                    if len(matches) >= limit:
                        break

        return {
            "agent": agent,
            "recent_matches": matches,
            "count": len(matches),
        }

    def _query_all(self, history: List[Dict],
                   agent: str, limit: int) -> Dict[str, Any]:
        """Return all available information."""
        return {
            "total_records": len(history),
            "agent_stats": self._query_agent_stats(history, agent) if agent else None,
            "rankings": self._query_rankings(history, limit),
        }

    def _recent_form(self, games: List[Dict], agent: str, n: int) -> str:
        """Get recent form string (W/L/D)."""
        form = []
        for game in games[-n:]:
            winner = game.get("winner", -1)
            a1 = game.get("agent", game.get("agent1", ""))
            if winner == -1:
                form.append("D")
            elif (a1 == agent and winner == 0) or (a1 != agent and winner == 1):
                form.append("W")
            else:
                form.append("L")
        return "".join(form)

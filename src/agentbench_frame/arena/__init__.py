"""
Adversarial arena for agent competition.

Provides:
- Match: Run matches between two agents
- Arena: Tournament system (round-robin, elimination, ladder)
- EloTracker: Elo/Glicko rating system for agent ranking
"""

from agentbench_frame.arena.match import Match
from agentbench_frame.arena.tournament import Arena
from agentbench_frame.arena.rating import EloTracker

__all__ = ["Match", "Arena", "EloTracker"]

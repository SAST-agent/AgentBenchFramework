"""Population split: train / test / validation opponent pools.

Three mutually-exclusive opponent pools for honest evaluation:

  TRAIN       -- opponents the iteration loop sees (used to optimize).
  TEST        -- held-out opponents for evaluation after each version.
  VALIDATION  -- final blind opponents, never touched until deployment.

Design principle: a strategy that only does well on TRAIN but tanks on
TEST/VALIDATION is overfitting.  The three-way split catches this.

Each pool is a dict {name: agent_instance}.  Opponents are deterministic
(fixed seeds for RandomAgent) so results are reproducible.
"""
from __future__ import annotations

from typing import Dict

from .agents import (
    BaseAgent,
    RandomAgent,
    GreedyAgent,
    WallHuggerAgent,
    CenterSeekerAgent,
    AggressiveChaserAgent,
    TerritoryMaximizerAgent,
    ItemHoarderAgent,
)


# --------------------------------------------------------------------------- #
# population definitions
# --------------------------------------------------------------------------- #


def _make_population(*specs) -> Dict[str, BaseAgent]:
    """Build a population dict from (name, constructor) pairs."""
    return {name: ctor() for name, ctor in specs}


# TRAIN pool: seen during iteration.  Weak-to-medium opponents.
# The iteration loop optimizes against these.
TRAIN_POOL = _make_population(
    ("random_0", lambda: RandomAgent(0)),
    ("random_1", lambda: RandomAgent(1)),
    ("wall_hugger", WallHuggerAgent),
    ("center_seeker", CenterSeekerAgent),
)

# TEST pool: held-out for evaluation.  Different styles, never seen
# during iteration.  Used to measure generalization.
TEST_POOL = _make_population(
    ("random_2", lambda: RandomAgent(2)),
    ("aggressive_chaser", AggressiveChaserAgent),
    ("item_hoarder", ItemHoarderAgent),
)

# VALIDATION pool: final blind opponents.  Strongest opponents, kept
# secret until the very end.  Includes the built-in greedy.
VALIDATION_POOL = _make_population(
    ("greedy_builtin", GreedyAgent),
    ("territory_max", TerritoryMaximizerAgent),
    ("random_3", lambda: RandomAgent(3)),
)


# Combined full population (for reference / all-vs-all tournaments).
ALL_OPPONENTS = {}
ALL_OPPONENTS.update(TRAIN_POOL)
ALL_OPPONENTS.update(TEST_POOL)
ALL_OPPONENTS.update(VALIDATION_POOL)


POOLS = {
    "train": TRAIN_POOL,
    "test": TEST_POOL,
    "validation": VALIDATION_POOL,
    "all": ALL_OPPONENTS,
}


def get_pool(name: str) -> Dict[str, BaseAgent]:
    """Return a named opponent pool.

    Args:
        name: one of "train", "test", "validation", "all".

    Returns:
        dict of {opponent_name: agent_instance}.
    """
    if name not in POOLS:
        raise ValueError(
            f"Unknown pool '{name}'. Choose from: {list(POOLS.keys())}"
        )
    return POOLS[name]


def pool_summary() -> str:
    """Human-readable summary of the population split."""
    lines = ["Population split (mutually exclusive opponent pools):"]
    for pool_name, pool in POOLS.items():
        if pool_name == "all":
            continue
        lines.append(f"\n  [{pool_name.upper()}]  ({len(pool)} opponents)")
        for opp_name in pool:
            lines.append(f"    - {opp_name}")
    lines.append(f"\n  Total unique opponents: {len(ALL_OPPONENTS)}")
    return "\n".join(lines)


__all__ = [
    "TRAIN_POOL", "TEST_POOL", "VALIDATION_POOL",
    "ALL_OPPONENTS", "POOLS",
    "get_pool", "pool_summary",
]

"""
Unified entry points for AntWAR2.

Three entry points that wire the full loop together:
  - eval_entry: single/multi-strategy evaluation with payoff matrix
  - iter_entry: coding-agent strategy iteration
  - rl_entry: RL training (PPO) with greedy fallback smoke test
"""

__all__ = ["eval_entry", "iter_entry", "rl_entry"]

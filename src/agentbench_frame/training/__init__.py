"""
Training loops for agent improvement.

Provides:
- PPOTrainer: Self-contained PPO training with torch (real RL)
- RLTrainer: Generic RL training scaffold (SB3/Tianshou)
- RuleIterator: Systematic rule-based agent improvement
- SelfPlayTrainer: Self-play training with opponent pools
"""

from agentbench_frame.training.ppo_trainer import PPOTrainer, PPOConfig
from agentbench_frame.training.rl_trainer import RLTrainer, RLTrainConfig
from agentbench_frame.training.rule_iterator import RuleIterator, RuleIterConfig
from agentbench_frame.training.self_play import SelfPlayTrainer, SelfPlayConfig

__all__ = [
    "PPOTrainer", "PPOConfig",
    "RLTrainer", "RLTrainConfig",
    "RuleIterator", "RuleIterConfig",
    "SelfPlayTrainer", "SelfPlayConfig",
]

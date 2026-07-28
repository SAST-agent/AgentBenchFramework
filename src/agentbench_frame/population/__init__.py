"""
Strategy population store and data split.

A Population is the on-disk registry of every strategy version (rule code,
RL checkpoints, external bots). DataSplit partitions the population into
train / validation / hidden-test buckets so training never touches the
hidden set.
"""

from agentbench_frame.population.store import Population, default_root
from agentbench_frame.population.split import DataSplit, SplitAssignment

__all__ = ["Population", "DataSplit", "SplitAssignment", "default_root"]

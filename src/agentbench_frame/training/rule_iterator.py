"""
Rule-Based Agent Iteration

Provides systematic improvement of rule-based agents through:
- Rule performance tracking (which rules fire, how often, with what outcome)
- A/B testing of rule modifications
- Parameter optimization for rule thresholds
- Automated rule discovery through simulation
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import itertools
import random

from agentbench_frame.env.base import BaseEnv
from agentbench_frame.agent.rule_based import RuleBasedAgent
from agentbench_frame.arena.match import Match


@dataclass
class RuleIterConfig:
    """Configuration for rule iteration."""
    n_eval_games: int = 50
    n_iterations: int = 10
    improvement_threshold: float = 0.55  # win rate to accept improvement
    log_dir: str = "./rule_iter_logs"


class RuleIterator:
    """
    Systematic rule-based agent improvement through iterative testing.

    Process:
    1. Profile current rules: which fire, success rate
    2. Generate variants: modify rule order, add/remove rules, tune parameters
    3. Evaluate variants against baseline
    4. Select best variant
    5. Repeat

    Example:
        iterator = RuleIterator(env, base_agent, opponent)
        improved_agent = iterator.iterate(n_iterations=10)
    """

    def __init__(self,
                 env: BaseEnv,
                 base_agent: RuleBasedAgent,
                 opponent: Any,
                 config: Optional[RuleIterConfig] = None):
        self.env = env
        self.base_agent = base_agent
        self.opponent = opponent
        self.config = config or RuleIterConfig()
        self._iteration_history: List[Dict] = []

    def iterate(self) -> Tuple[RuleBasedAgent, Dict[str, Any]]:
        """
        Run the rule iteration process.

        Returns:
            (best_agent, iteration_summary)
        """
        current_best = copy.deepcopy(self.base_agent)
        baseline_winrate = self._evaluate_agent(current_best)
        print(f"Baseline win rate: {baseline_winrate:.3f}")

        for iteration in range(self.config.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.config.n_iterations}")

            # Profile current rules
            profile = self._profile_rules(current_best)

            # Generate variants
            variants = self._generate_variants(current_best, profile)

            # Evaluate variants
            best_variant = None
            best_winrate = baseline_winrate

            for i, variant in enumerate(variants):
                winrate = self._evaluate_agent(variant)
                print(f"  Variant {i}: win_rate={winrate:.3f}")

                if winrate > best_winrate:
                    best_winrate = winrate
                    best_variant = variant

            # Update if improved
            if best_variant and best_winrate > baseline_winrate + 0.01:
                current_best = best_variant
                baseline_winrate = best_winrate
                print(f"  Improved! New baseline: {baseline_winrate:.3f}")
            else:
                print(f"  No improvement found.")

            self._iteration_history.append({
                "iteration": iteration,
                "baseline_winrate": baseline_winrate,
                "rule_profile": profile,
                "variants_tested": len(variants),
            })

        return current_best, {
            "initial_winrate": self._evaluate_agent(self.base_agent),
            "final_winrate": baseline_winrate,
            "iterations": self.config.n_iterations,
            "history": self._iteration_history,
        }

    def _evaluate_agent(self, agent: RuleBasedAgent) -> float:
        """Evaluate agent win rate against opponent."""
        match = Match(self.env, agent, self.opponent)
        result = match.run(n_games=self.config.n_eval_games)
        return result["win_rate"]

    def _profile_rules(self, agent: RuleBasedAgent) -> Dict[str, Any]:
        """Profile which rules fire and their success rates."""
        stats = agent.get_stats()
        return {
            "total_calls": stats.get("calls", 0),
            "rule_hits": stats.get("rule_hits", {}),
            "num_rules": len(agent.rules),
        }

    def _generate_variants(self, agent: RuleBasedAgent,
                           profile: Dict) -> List[RuleBasedAgent]:
        """Generate variant agents to test."""
        variants = []

        # Variant 1: Reorder rules (move least-used rules to front)
        if profile.get("rule_hits"):
            hits = profile["rule_hits"]
            sorted_rules = sorted(
                enumerate(agent.rules),
                key=lambda x: hits.get(getattr(x[1], "__name__", f"rule_{x[0]}"), 0),
                reverse=True,
            )
            reordered = copy.deepcopy(agent)
            reordered.rules = [r for _, r in sorted_rules]
            reordered.name = f"{agent.name}_reordered"
            variants.append(reordered)

        # Variant 2: Remove least-used rule
        if len(agent.rules) > 2 and profile.get("rule_hits"):
            hits = profile["rule_hits"]
            worst_idx = min(
                range(len(agent.rules)),
                key=lambda i: hits.get(getattr(agent.rules[i], "__name__", f"rule_{i}"), 0),
            )
            trimmed = copy.deepcopy(agent)
            trimmed.rules.pop(worst_idx)
            trimmed.name = f"{agent.name}_trimmed"
            variants.append(trimmed)

        # Variant 3: Simplified (only first and last rules)
        if len(agent.rules) > 2:
            simplified = copy.deepcopy(agent)
            simplified.rules = [agent.rules[0], agent.rules[-1]]
            simplified.name = f"{agent.name}_simple"
            variants.append(simplified)

        return variants

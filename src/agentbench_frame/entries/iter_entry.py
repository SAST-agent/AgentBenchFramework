"""
Coding-agent iteration entry point for AntWAR2.

Implements the PSRO-style improve loop:
  1. load the current best strategy from the Population
  2. invoke a pluggable coding-agent hook to produce a candidate variant
  3. register the variant as a new version in the Population
  4. evaluate it against the train population via PayoffMatrix
  5. accept or reject based on average win-rate improvement

The coding-agent hook is a callable ``edit_fn(strategy, feedback) -> strategy``
that defaults to a parameter-space mutation when no real LLM is wired. This
keeps the full loop runnable end-to-end without external dependencies while
remaining a drop-in slot for a real coding agent.

Usage:
    python -m agentbench_frame.entries.iter_entry --base greedy \\
        --iterations 3 --n-games 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random as _random
import sys
from typing import Any, Callable, Dict, List, Optional

from agentbench_frame.env.antwar2_env import AntWar2Env
from agentbench_frame.population.store import Population
from agentbench_frame.population.split import DataSplit
from agentbench_frame.arena.payoff import PayoffMatrix
from agentbench_frame.arena.match import Match
from agentbench_frame.strategy.base import BaseStrategy, StrategyMeta
from agentbench_frame.strategy.rule_strategy import RuleStrategy
from agentbench_frame.tracking.run import Run

GAME = "30_antwar2"


def default_edit_fn(strategy: BaseStrategy,
                    feedback: Dict[str, Any]) -> BaseStrategy:
    """Default no-LLM edit: mutate the rule policy or parameters.

    Tries to cycle through rule policies (greedy -> pragmatic -> random) to
    produce a genuinely different candidate. For external/RL strategies the
    same object is returned (no-op) so the loop still runs.
    """
    if isinstance(strategy, RuleStrategy):
        cycle = ["greedy", "pragmatic", "random"]
        current = getattr(strategy, "policy_name", "greedy")
        try:
            idx = cycle.index(current)
        except ValueError:
            idx = 0
        new_policy = cycle[(idx + 1) % len(cycle)]
        return RuleStrategy(
            name=strategy.name + "_" + new_policy,
            policy=new_policy,
            meta=StrategyMeta(
                strategy_id=strategy.name + "_" + new_policy,
                game=GAME,
                kind="rule",
                parent_id=strategy.meta.strategy_id,
            ),
        )
    return copy.deepcopy(strategy)


def _build_strategies(ids: List[str], pop: Population) -> Dict[str, Any]:
    strategies: Dict[str, Any] = {}
    for sid in ids:
        if pop.exists(sid):
            try:
                strategies[sid] = pop.get(sid)
            except Exception:
                strategies[sid] = RuleStrategy(name=sid, policy=sid)
        else:
            strategies[sid] = RuleStrategy(name=sid, policy=sid)
    return strategies


def _evaluate(strategy: BaseStrategy,
              opponents: Dict[str, Any],
              env: AntWar2Env,
              n_games: int,
              seed: int) -> float:
    if not opponents:
        return 0.5
    total = 0.0
    for opp_id, opp in opponents.items():
        result = Match(env, strategy, opp, seed=seed).run(n_games=n_games)
        total += result.win_rate
    return total / len(opponents)


def run_iteration(base_id: str,
                  iterations: int,
                  n_games: int,
                  seed: int,
                  data_dir: Optional[str],
                  edit_fn: Optional[Callable] = None,
                  population_root: Optional[str] = None) -> Dict[str, Any]:
    pop = Population(root=population_root, game=GAME)
    split = DataSplit(pop)
    train_ids = split.train_ids()
    if not train_ids:
        train_ids = [s for s in ["hold", "greedy", "pragmatic"] if s != base_id]
    opponents = _build_strategies(train_ids, pop)
    env = AntWar2Env()
    edit_fn = edit_fn or default_edit_fn

    if pop.exists(base_id):
        current = pop.get(base_id)
    else:
        current = RuleStrategy(name=base_id, policy=base_id)
        pop.register(current, note="initial baseline")

    run = Run.start(
        game=GAME, agent=base_id, run_type="rule_iter", data_dir=data_dir,
        config={"iterations": iterations, "n_games": n_games, "seed": seed,
                "base": base_id, "train_population": train_ids},
    )

    best_wr = _evaluate(current, opponents, env, n_games, seed)
    run.log_interactions(len(opponents) * n_games)
    history: List[Dict[str, Any]] = [{
        "iteration": 0,
        "strategy_id": current.meta.strategy_id,
        "version": current.meta.version,
        "avg_win_rate": best_wr,
        "accepted": True,
    }]
    run.log_elo(1500.0)

    for i in range(1, iterations + 1):
        feedback = {"iteration": i, "best_wr": best_wr,
                     "history": history[-5:]}
        candidate = edit_fn(current, feedback)
        sid, version = pop.register(
            candidate, parent_id=current.meta.strategy_id,
            note=f"coding-agent iteration {i}")
        candidate = pop.get(sid)
        cand_wr = _evaluate(candidate, opponents, env, n_games, seed)
        run.log_interactions(len(opponents) * n_games)

        accepted = cand_wr >= best_wr
        if accepted:
            current = candidate
            best_wr = cand_wr
        run.log_elo(1500.0 + (best_wr - 0.5) * 200 * i)

        entry = {
            "iteration": i,
            "strategy_id": sid,
            "version": version,
            "avg_win_rate": cand_wr,
            "accepted": accepted,
        }
        history.append(entry)
        run.write("iteration", **entry)

    result = {
        "base": base_id,
        "iterations": iterations,
        "best_strategy_id": current.meta.strategy_id,
        "best_avg_win_rate": best_wr,
        "history": history,
    }

    summary = run.finish()
    summary.update(result)
    return summary


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="antwar2-iterate",
        description="AntWAR2 coding-agent iteration entry point",
    )
    parser.add_argument("--base", default="greedy",
                        help="base strategy id to iterate from")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--n-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--population-root", default=None)
    args = parser.parse_args(argv)

    summary = run_iteration(
        base_id=args.base, iterations=args.iterations, n_games=args.n_games,
        seed=args.seed, data_dir=args.data_dir,
        population_root=args.population_root,
    )

    print(json.dumps({
        "run_id": summary.get("run_id"),
        "best_strategy_id": summary.get("best_strategy_id"),
        "best_avg_win_rate": summary.get("best_avg_win_rate"),
        "cost_summary": summary.get("cost_summary"),
        "iterations_completed": summary.get("iterations"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

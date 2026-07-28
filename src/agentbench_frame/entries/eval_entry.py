"""
Evaluation entry point for AntWAR2.

Two modes:
  - single-strategy eval: evaluate one strategy against the train/validation
    population and report head-to-head win rates.
  - multi-strategy comparison: run a full round-robin PayoffMatrix over a set
    of strategies and produce a ranked summary.

Both modes write a CI-compatible run.toml + summary.json via Run, including a
cost_summary block from BudgetRecorder.

Usage:
    python -m agentbench_frame.entries.eval_entry --mode compare \\
        --strategies hold greedy pragmatic --n-games 4
    python -m agentbench_frame.entries.eval_entry --mode single \\
        --target greedy --n-games 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from agentbench_frame.env.antwar2_env import AntWar2Env
from agentbench_frame.population.store import Population
from agentbench_frame.population.split import DataSplit
from agentbench_frame.arena.payoff import PayoffMatrix
from agentbench_frame.arena.match import Match
from agentbench_frame.strategy.rule_strategy import RuleStrategy
from agentbench_frame.strategy.base import load_strategy
from agentbench_frame.tracking.run import Run


GAME = "30_antwar2"


def _build_strategies(strategy_ids: List[str], pop: Population
                      ) -> Dict[str, Any]:
    strategies: Dict[str, Any] = {}
    for sid in strategy_ids:
        if pop.exists(sid):
            try:
                strategies[sid] = pop.get(sid)
            except Exception:
                strategies[sid] = RuleStrategy(name=sid, policy=sid)
        else:
            strategies[sid] = RuleStrategy(name=sid, policy=sid)
    return strategies


def run_compare(strategy_ids: List[str],
                n_games: int,
                seed: int,
                data_dir: Optional[str],
                split_set: str = "visible",
                population_root: Optional[str] = None) -> Dict[str, Any]:
    pop = Population(root=population_root, game=GAME)
    split = DataSplit(pop)

    if split_set == "hidden":
        ids = split.hidden_ids() or strategy_ids
        report_set = "hidden"
    elif split_set == "validation":
        ids = split.validation_ids() or strategy_ids
        report_set = "validation"
    elif split_set == "train":
        ids = split.train_ids() or strategy_ids
        report_set = "train"
    else:
        ids = split.visible_ids() or strategy_ids
        report_set = "visible"

    if not ids:
        ids = list(strategy_ids)
        report_set = "cli"

    strategies = _build_strategies(ids, pop)
    env = AntWar2Env()

    run = Run.start(
        game=GAME, agent="eval_compare", run_type="eval", data_dir=data_dir,
        config={"mode": "compare", "strategies": ids, "set": report_set,
                "n_games": n_games, "seed": seed},
    )

    payoff = PayoffMatrix(game=GAME)
    stats = payoff.update(env, strategies, n_games=n_games, seed=seed)
    run.log_interactions(stats["total_pairs"] * n_games)

    ranking = payoff.ranking(ids)
    summary_data = payoff.summary(ids)

    run.log_h2h({
        a: {b: payoff.get(a, b) for b in ids if b != a and payoff.has(a, b)}
        for a in ids
    })

    result = {
        "eval_mode": "compare",
        "eval_set": report_set,
        "strategies": ids,
        "n_games_per_pair": n_games,
        "payoff_stats": stats,
        "ranking": [
            {"strategy_id": s, "avg_win_rate": wr, "wins": w, "games": g}
            for s, wr, w, g in ranking
        ],
        "h2h_matrix": summary_data["matrix"],
    }

    summary = run.finish()
    summary.update(result)

    payoff_path = os.path.join(run.run_dir, "payoff.json")
    payoff.save(payoff_path)

    return summary


def run_single(target: str,
               n_games: int,
               seed: int,
               data_dir: Optional[str],
               split_set: str = "visible",
               population_root: Optional[str] = None) -> Dict[str, Any]:
    pop = Population(root=population_root, game=GAME)
    split = DataSplit(pop)

    if split_set == "hidden":
        opp_ids = split.hidden_ids()
    elif split_set == "validation":
        opp_ids = split.validation_ids()
    elif split_set == "train":
        opp_ids = split.train_ids()
    else:
        opp_ids = split.visible_ids()

    if not opp_ids:
        opp_ids = [s for s in ["hold", "greedy", "pragmatic"] if s != target]

    if pop.exists(target):
        target_strategy = pop.get(target)
    else:
        target_strategy = RuleStrategy(name=target, policy=target)

    opponents = _build_strategies(opp_ids, pop)
    env = AntWar2Env()

    run = Run.start(
        game=GAME, agent=target, run_type="eval", data_dir=data_dir,
        config={"mode": "single", "target": target, "set": split_set,
                "opponents": opp_ids, "n_games": n_games, "seed": seed},
    )

    h2h: Dict[str, Dict[str, float]] = {target: {}}
    for opp_id, opp_strategy in opponents.items():
        result = Match(env, target_strategy, opp_strategy, seed=seed).run(
            n_games=n_games)
        wr = result.win_rate
        h2h[target][opp_id] = wr
        run.write("eval_result", opponent=opp_id, win_rate=wr,
                  wins=result.agent1_wins, losses=result.agent2_wins,
                  draws=result.draws)
        run.log_interactions(n_games)

    run.log_h2h(h2h)
    avg_wr = sum(h2h[target].values()) / max(1, len(h2h[target]))

    result = {
        "eval_mode": "single",
        "target": target,
        "eval_set": split_set,
        "opponents": opp_ids,
        "n_games_per_opponent": n_games,
        "h2h": h2h,
        "avg_win_rate": avg_wr,
    }

    summary = run.finish()
    summary.update(result)
    return summary


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="antwar2-eval",
        description="AntWAR2 evaluation entry point",
    )
    parser.add_argument("--mode", choices=["compare", "single"],
                        default="compare")
    parser.add_argument("--target", default="greedy",
                        help="strategy id for single mode")
    parser.add_argument("--strategies", nargs="*",
                        default=["hold", "greedy", "pragmatic"],
                        help="strategy ids for compare mode")
    parser.add_argument("--n-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--set", choices=["visible", "train", "validation",
                                          "hidden"], default="visible")
    parser.add_argument("--population-root", default=None)
    args = parser.parse_args(argv)

    if args.mode == "single":
        summary = run_single(
            target=args.target, n_games=args.n_games, seed=args.seed,
            data_dir=args.data_dir, split_set=args.set,
            population_root=args.population_root,
        )
    else:
        summary = run_compare(
            strategy_ids=args.strategies, n_games=args.n_games, seed=args.seed,
            data_dir=args.data_dir, split_set=args.set,
            population_root=args.population_root,
        )

    print(json.dumps({
        "run_id": summary.get("run_id"),
        "eval_mode": summary.get("eval_mode"),
        "cost_summary": summary.get("cost_summary"),
        "ranking": summary.get("ranking"),
        "avg_win_rate": summary.get("avg_win_rate"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Unified evaluation entry point.

Provides a single CLI and API for evaluating a strategy against any
population pool.  Writes CI-compatible ``run.toml`` + ``summary.json``
following the AgentBenchFrame data contract:

    runs/{game}/{agent}/{run_id}/
        run.toml        # metadata (type, created, git_commit, ...)
        summary.json    # win_rate, elo_history, h2h, ...

Usage (CLI):

    python -m snakego.evaluate --agent greedy --population validation
    python -m snakego.evaluate --agent greedy --population all --n-seeds 4
    python -m snakego.evaluate --strategy snake_ai:v3 --population test

Usage (API):

    from snakego.evaluate import evaluate
    result = evaluate(GreedyAgent(), pool="validation", n_seeds=4)
    print(result["win_rate"], result["h2h"])
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import pathlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agents import BaseAgent, BUILTIN_AGENTS
from .arena import EloTracker
from .population import get_pool, POOLS
from .smoke_test import play_game


GAME_ID = "26_snakego"


# --------------------------------------------------------------------------- #
# core evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    candidate: BaseAgent,
    pool: str = "validation",
    n_seeds: int = 4,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate *candidate* against an opponent pool.

    Each opponent is played *n_seeds* games each direction (candidate as
    P0 and P1) for side-balance.

    Returns a dict with win_rate, h2h, elo, and per-opponent details.
    """
    opponents = get_pool(pool)
    candidate.name = candidate.name or "candidate"

    h2h: Dict[str, float] = {}
    total_wins = 0
    total_games = 0
    elo = EloTracker()
    elo.ensure(candidate.name)
    per_opp: Dict[str, Any] = {}

    for opp_name, opp in opponents.items():
        elo.ensure(opp_name)
        wins = 0
        games = 0
        for s in range(n_seeds):
            seed = base_seed + s
            # candidate as P0
            r = play_game(candidate, opp, seed=seed)
            games += 1
            if r["winner"] == 0:
                wins += 1
                elo.update(candidate.name, 1.0, opp_name)
                elo.update(opp_name, 0.0, candidate.name)
            elif r["winner"] == 1:
                elo.update(candidate.name, 0.0, opp_name)
                elo.update(opp_name, 1.0, candidate.name)
            else:
                elo.update(candidate.name, 0.5, opp_name)
                elo.update(opp_name, 0.5, candidate.name)
            # candidate as P1
            r2 = play_game(opp, candidate, seed=seed + 1000)
            games += 1
            if r2["winner"] == 1:
                wins += 1
                elo.update(candidate.name, 1.0, opp_name)
                elo.update(opp_name, 0.0, candidate.name)
            elif r2["winner"] == 0:
                elo.update(candidate.name, 0.0, opp_name)
                elo.update(opp_name, 1.0, candidate.name)
            else:
                elo.update(candidate.name, 0.5, opp_name)
                elo.update(opp_name, 0.5, candidate.name)
        rate = wins / games if games else 0.0
        h2h[opp_name] = round(rate, 4)
        per_opp[opp_name] = {
            "win_rate": round(rate, 4),
            "wins": wins,
            "games": games,
        }
        total_wins += wins
        total_games += games

    overall = total_wins / total_games if total_games else 0.0
    candidate_elo = elo.get(candidate.name)

    return {
        "agent": candidate.name,
        "pool": pool,
        "win_rate": round(overall, 4),
        "h2h": h2h,
        "elo": round(candidate_elo),
        "per_opponent": per_opp,
        "total_games": total_games,
        "n_seeds": n_seeds,
    }


# --------------------------------------------------------------------------- #
# data-contract output (run.toml + summary.json)
# --------------------------------------------------------------------------- #


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    h = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return f"{ts}_{h}"


def _data_root(data_dir: Optional[str] = None) -> str:
    return data_dir or os.environ.get(
        "AGENTBENCH_DATA", str(Path.cwd() / "agentbench_data"))


def write_run(
    candidate_name: str,
    eval_result: Dict[str, Any],
    pool: str,
    data_dir: Optional[str] = None,
) -> Path:
    """Write run.toml + summary.json in the AgentBenchFrame data contract.

    Layout::

        {data_dir}/runs/26_snakego/{agent}/{run_id}/
            run.toml
            summary.json

    Returns the run directory path.
    """
    root = _data_root(data_dir)
    rid = _run_id()
    run_dir = Path(root) / "runs" / GAME_ID / candidate_name / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = _git_commit()
    total_games = eval_result["total_games"]

    # run.toml
    toml_lines = [
        "[run]",
        f'run_id = "{rid}"',
        f'game = "{GAME_ID}"',
        f'agent = "{candidate_name}"',
        f'type = "eval"',
        f'created = "{now}"',
        f'git_commit = "{commit}"',
        f'started_at = {time.time():.1f}',
        f'finished_at = {time.time():.1f}',
        f'total_steps = {total_games}',
        f'total_episodes = {total_games}',
        "",
        "[config]",
        f'pool = "{pool}"',
        f'n_seeds = {eval_result["n_seeds"]}',
    ]
    (run_dir / "run.toml").write_text(
        "\n".join(toml_lines) + "\n", encoding="utf-8")

    # summary.json
    summary = {
        "run_id": rid,
        "game": GAME_ID,
        "agent": candidate_name,
        "run_type": "eval",
        "created": now,
        "git_commit": commit,
        "wall_hours": 0.0,
        "total_episodes": total_games,
        "total_steps": total_games,
        "win_rate": eval_result["win_rate"],
        "best_elo": eval_result["elo"],
        "final_elo": eval_result["elo"],
        "elo_history": [{"step": total_games, "elo": eval_result["elo"]}],
        "h2h": {candidate_name: eval_result["h2h"]},
        "resource_summary": {},
        "config": {"pool": pool, "n_seeds": eval_result["n_seeds"]},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    return run_dir


# --------------------------------------------------------------------------- #
# strategy loading helper
# --------------------------------------------------------------------------- #


def load_strategy_agent(spec):
    """Load a strategy from a spec string.

    Formats:
      "greedy"        -- builtin agent name
      "snake_ai:v3"   -- versioned strategy from VersionStore
      "snake_ai"      -- latest version of a strategy
    """
    from .versioning import VersionStore

    if ":" in spec:
        name, ver = spec.split(":", 1)
        ver = ver.lstrip("v")
        store = VersionStore(pathlib.Path("outputs/snakego_strategies"))
        return store.load_agent(name, int(ver))

    if spec in BUILTIN_AGENTS:
        return BUILTIN_AGENTS[spec]()

    store = VersionStore(pathlib.Path("outputs/snakego_strategies"))
    agent = store.load_latest_agent(spec)
    if agent is None:
        raise ValueError(f"Unknown agent: '{spec}'")
    return agent


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snakego.evaluate",
        description="Evaluate a SnakeGo strategy against an opponent pool.",
    )
    parser.add_argument("--agent", default="greedy",
                        help='builtin name, "name:version", or strategy name')
    parser.add_argument("--population", default="validation",
                        choices=list(POOLS.keys()),
                        help="opponent pool to evaluate against")
    parser.add_argument("--n-seeds", type=int, default=4,
                        help="games per opponent per side")
    parser.add_argument("--data-dir", default=None,
                        help="data root for run.toml/summary.json "
                             "(default: $AGENTBENCH_DATA)")
    parser.add_argument("--save", action="store_true",
                        help="write run.toml + summary.json to data dir")
    args = parser.parse_args(argv)

    candidate = load_strategy_agent(args.agent)
    print(f"Evaluating {candidate.name} vs [{args.population}] pool "
          f"({args.n_seeds} seeds/side)...\n")

    result = evaluate(candidate, pool=args.population, n_seeds=args.n_seeds)

    print(f"Agent:      {result['agent']}")
    print(f"Pool:       {result['pool']}  ({len(result['h2h'])} opponents)")
    print(f"Win rate:   {result['win_rate']:.1%}")
    print(f"Elo:        {result['elo']}")
    print(f"Total games: {result['total_games']}")
    print(f"\nPer-opponent (h2h):")
    for opp, rate in sorted(result["h2h"].items(), key=lambda x: -x[1]):
        tag = "WIN" if rate > 0.5 else ("TIE" if rate == 0.5 else "LOSE")
        print(f"  {opp:20s} {rate:.1%}  [{tag}]")

    if args.save:
        run_dir = write_run(
            candidate.name, result, args.population, args.data_dir)
        print(f"\nSaved to: {run_dir}")
        print(f"  run.toml + summary.json (AgentBenchFrame data contract)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "write_run", "load_strategy_agent", "GAME_ID"]

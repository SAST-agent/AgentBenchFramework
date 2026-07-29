"""
CLI entry point for AgentBench.

Subcommands:
    train   — RL training
    eval    — Agent evaluation
    iterate — Rule-based agent iteration
    arena   — Multi-agent tournament
    report  — Build static report site
    mcp     — Start MCP server
    complexity — Reproducible research complexity metrics
"""

import argparse
import json, sys, tomllib
from pathlib import Path
from typing import List, Optional


def _cmd_train(args):
    """Run RL training."""
    from agentbench_frame.runner.rl_runner import BaseRLRunner

    runner = BaseRLRunner(
        game=args.game,
        agent=args.agent,
        data_dir=args.data_dir,
        total_timesteps=args.total_timesteps,
        config={
            "max_episodes": args.max_episodes,
            "learning_rate": args.learning_rate,
        },
    )
    summary = runner.run()
    print(f"Training complete. Summary: {summary.get('run_id', '?')}")


def _cmd_eval(args):
    """Run evaluation."""
    from agentbench_frame.runner.eval_runner import BaseEvalRunner

    opponents = args.opponents.split(",") if args.opponents else ["random"]
    runner = BaseEvalRunner(
        game=args.game,
        agent=args.agent,
        data_dir=args.data_dir,
        opponents=opponents,
        config={"n_eval_games": args.n_games},
    )
    summary = runner.run()
    print("Evaluation complete.")
    for opp, result in summary.get("eval_results", {}).items():
        print(f"  vs {opp}: win_rate={result['win_rate']:.2%} "
              f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")


def _cmd_iterate(args):
    """Run rule-based iteration."""
    from agentbench_frame.runner.rule_runner import BaseRuleRunner

    runner = BaseRuleRunner(
        game=args.game,
        agent=args.agent,
        data_dir=args.data_dir,
        opponent=args.opponent,
        config={"n_games": args.n_games},
    )
    summary = runner.run()
    print(f"Iteration complete. Win rate: {summary.get('win_rate', 0):.2%}")


def _cmd_arena(args):
    """Run adversarial tournament."""
    from agentbench_frame.env.registry import make_env
    from agentbench_frame.env.base import EnvMode
    from agentbench_frame.agent.registry import AgentRegistry
    from agentbench_frame.agent.base import RandomAgent
    from agentbench_frame.arena.tournament import Arena

    env = make_env(args.game, mode=EnvMode.DIRECT)
    agent_names = [a.strip() for a in args.agents.split(",")]
    agents = []
    for name in agent_names:
        try:
            agent = AgentRegistry.create(name)
        except KeyError:
            agent = RandomAgent(name=name)
        agents.append(agent)

    arena = Arena(env, agents, name="CLI Tournament")
    result = arena.round_robin(n_games=args.n_games)
    print(f"Tournament complete!")
    for rank, name, elo in result.rankings:
        print(f"  {rank}. {name} (Elo: {elo:.0f})")


def _cmd_report(args):
    """Build static report site."""
    from agentbench_frame.report.builder import build_report

    out_dir = build_report(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"Report built at: {out_dir}")


def _cmd_mcp(args):
    """Start MCP server."""
    from agentbench_frame.mcp.server import create_default_server

    print(f"Starting MCP server...", file=sys.stderr)
    server = create_default_server()
    server.run()


def _cmd_data_check(args):
    """Validate data directory against CI schema."""
    import json, tomllib
    from agentbench_frame.tracking.run import _data_root
    data_dir = args.data_dir or _data_root()
    runs_root = Path(data_dir) / "runs"
    if not runs_root.is_dir():
        print(f"ERROR: runs/ not found in {data_dir}")
        print(f"Expected: {data_dir}/runs/{{game}}/{{agent}}/{{run_id}}/")
        sys.exit(1)

    errors, ok = [], 0
    for run_toml_path in sorted(runs_root.rglob("run.toml")):
        run_dir = run_toml_path.parent
        rel = str(run_dir.relative_to(data_dir))
        errs = []
        # Check run.toml
        try:
            meta = tomllib.loads(run_toml_path.read_text())
            for field in ["run_id", "game", "agent", "type", "created"]:
                val = meta.get("run", {}).get(field, "")
                if not val:
                    errs.append(f"run.toml: [run] missing '{field}'")
        except Exception as e:
            errs.append(f"run.toml: parse error: {e}")

        # Check summary.json
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            errs.append("missing summary.json")
        else:
            try:
                s = json.loads(summary_path.read_text())
                for field in ["run_id", "game", "agent", "wall_hours", "total_steps", "win_rate"]:
                    if field not in s:
                        errs.append(f"summary.json: missing '{field}'")
            except Exception as e:
                errs.append(f"summary.json: parse error: {e}")

        if errs:
            print(f"FAIL {rel}:")
            for e in errs: print(f"  - {e}")
            errors.append(rel)
        else:
            ok += 1

    print(f"\n{ok} valid, {len(errors)} invalid" + (" (all good!)" if not errors else ""))
    sys.exit(1 if errors else 0)


def _cmd_data_list(args):
    """List all runs in data directory."""
    from agentbench_frame.tracking.run import _data_root
    data_dir = args.data_dir or _data_root()
    runs_root = Path(data_dir) / "runs"
    if not runs_root.is_dir():
        print("No runs yet.")
        return

    found = 0
    for run_dir in sorted(runs_root.rglob("summary.json")):
        rel = run_dir.parent.relative_to(data_dir)
        try:
            s = json.loads(run_dir.read_text())
            print(f"  {s.get('game','?')}/{s.get('agent','?')}  "
                  f"type={s.get('run_type','?')}  "
                  f"Elo={s.get('best_elo','-')}  "
                  f"win={s.get('win_rate',0):.0%}  "
                  f"steps={s.get('total_steps','-')}  "
                  f"hours={s.get('wall_hours','-')}h")
            found += 1
        except Exception:
            print(f"  {rel}  (unreadable)")
            found += 1
    if found == 0:
        print("No runs yet. Start one with: agentbench train --game 28_generals --agent my_agent")


def _cmd_ludi_complexity(args):
    """Calculate AB-Ludi/1 source-description complexity upper bounds."""
    from pathlib import Path

    from agentbench_frame.research.ludi_k import (
        measure_agentbench_repository,
        write_json_report,
        write_markdown_report,
    )

    json_output = Path(args.json_output).resolve()
    markdown_output = Path(args.markdown_output).resolve()
    same_output = json_output == markdown_output
    if json_output.exists() and markdown_output.exists():
        same_output = same_output or json_output.samefile(markdown_output)
    if same_output:
        raise ValueError(
            "--json-output and --markdown-output must resolve to different paths"
        )

    report = measure_agentbench_repository(args.agentbench_repo)
    json_path = write_json_report(report, json_output)
    markdown_path = write_markdown_report(report, markdown_output)

    print("AB-Ludi/1 complexity upper bounds:")
    for rank, game in enumerate(report["games"], start=1):
        print(
            f"  {rank:2d}. {game['game_id']:<16} "
            f"{game['k_upper_bits']:>10} bits  "
            f"modules={game['module_count']}"
        )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="agentbench",
        description="AgentBench — unified agent framework for Saiblo games",
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # --- train ---
    p_train = sub.add_parser("train", help="RL training")
    p_train.add_argument("--game", default="generals")
    p_train.add_argument("--agent", default="ppo-v1")
    p_train.add_argument("--data-dir", default="./runs")
    p_train.add_argument("--total-timesteps", type=int, default=200_000)
    p_train.add_argument("--max-episodes", type=int, default=5000)
    p_train.add_argument("--learning-rate", type=float, default=3e-4)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate agent")
    p_eval.add_argument("--game", default="generals")
    p_eval.add_argument("--agent", default="ppo-v1")
    p_eval.add_argument("--data-dir", default="./runs")
    p_eval.add_argument("--opponents", default="random")
    p_eval.add_argument("--n-games", type=int, default=50)

    # --- iterate ---
    p_iter = sub.add_parser("iterate", help="Rule-based iteration")
    p_iter.add_argument("--game", default="generals")
    p_iter.add_argument("--agent", default="expansionist")
    p_iter.add_argument("--data-dir", default="./runs")
    p_iter.add_argument("--opponent", default="random")
    p_iter.add_argument("--n-games", type=int, default=100)

    # --- arena ---
    p_arena = sub.add_parser("arena", help="Tournament/arena")
    p_arena.add_argument("--game", default="generals")
    p_arena.add_argument("--agents", default="random,expansionist,aggressive")
    p_arena.add_argument("--n-games", type=int, default=20)

    # --- report ---
    p_report = sub.add_parser("report", help="Build report site")
    p_report.add_argument("--data-dir", default="./runs")
    p_report.add_argument("--output-dir", default="_site")

    # --- mcp ---
    p_mcp = sub.add_parser("mcp", help="Start MCP server")

    # --- data ---
    p_data = sub.add_parser("data", help="Data management")
    p_data_sub = p_data.add_subparsers(dest="data_command")
    p_check = p_data_sub.add_parser("check", help="Validate data against CI schema")
    p_check.add_argument("--data-dir", default=None, help="Data directory (default: $AGENTBENCH_DATA)")
    p_list = p_data_sub.add_parser("list", help="List all runs in data directory")
    p_list.add_argument("--data-dir", default=None, help="Data directory (default: $AGENTBENCH_DATA)")

    # --- complexity ---
    p_complexity = sub.add_parser(
        "complexity",
        help="Reproducible research complexity metrics",
    )
    p_complexity_sub = p_complexity.add_subparsers(dest="complexity_command")
    p_ludi = p_complexity_sub.add_parser(
        "ludi",
        help="Calculate AB-Ludi/1 game-logic K upper bounds",
    )
    p_ludi.add_argument(
        "--agentbench-repo",
        required=True,
        help="Local checkout of https://github.com/Aoraku/AgentBench",
    )
    p_ludi.add_argument(
        "--json-output",
        required=True,
        help="Machine-readable report path",
    )
    p_ludi.add_argument(
        "--markdown-output",
        required=True,
        help="Human-readable report path",
    )

    args = parser.parse_args(argv)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "iterate":
        _cmd_iterate(args)
    elif args.command == "arena":
        _cmd_arena(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "mcp":
        _cmd_mcp(args)
    elif args.command == "data":
        if args.data_command == "check":
            _cmd_data_check(args)
        elif args.data_command == "list":
            _cmd_data_list(args)
        else:
            p_data.print_help()
    elif args.command == "complexity":
        if args.complexity_command == "ludi":
            _cmd_ludi_complexity(args)
        else:
            p_complexity.print_help()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

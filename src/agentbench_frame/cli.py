"""
CLI entry point for AgentBench.

Subcommands:
    train   — RL training
    eval    — Agent evaluation
    iterate — Rule-based agent iteration
    arena   — Multi-agent tournament
    report  — Build static report site
    mcp     — Start MCP server
"""

import argparse
import sys
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

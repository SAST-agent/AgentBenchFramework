"""Miracle 核心入口：运行对战与解析回放。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent_bridge import AGENTS


def _cmd_match(args) -> int:
    from .match import run_match

    result = run_match(
        AGENTS[args.agent0](),
        AGENTS[args.agent1](),
        seed=args.seed,
        replay_dir=args.output_dir,
        tag=args.tag or f"{args.agent0}_vs_{args.agent1}",
    )
    print(json.dumps({
        "agent0": args.agent0,
        "agent1": args.agent1,
        "winner": result.winner,
        "scores": list(result.scores),
        "rounds": result.rounds,
        "terminated_by": result.terminated_by,
        "errors": list(result.errors),
        "replay": result.replay_path,
        "trace": result.trace_path,
    }, ensure_ascii=False, indent=2))
    return 0 if result.terminated_by in {"normal", "timeout"} else 1


def _cmd_replay(args) -> int:
    from .replay import parse_replay, save_replay_json, summarize

    events = parse_replay(args.path)
    print(json.dumps(summarize(events), ensure_ascii=False, indent=2))
    if args.jsonl:
        save_replay_json(events, args.jsonl)
        print(f"事件时间线已写: {args.jsonl}")
    return 0


def _cmd_ig(args) -> int:
    from .ig import (
        build_ig_curve,
        compare_agents_on_trace,
        load_episode_ig,
        save_episode_ig,
        save_ig_curve,
        versions_from_episodes,
    )

    episode = compare_agents_on_trace(
        args.trace,
        AGENTS[args.old](),
        AGENTS[args.new](),
        camp=args.camp,
        iteration=args.iteration,
        old_version=args.old,
        new_version=args.new,
    )
    episode_path = save_episode_ig(episode, args.output_dir)
    episodes = load_episode_ig(args.output_dir)
    curve = build_ig_curve(episodes, versions=versions_from_episodes(episodes))
    curve_path = args.output_dir / "ig_curve.json"
    save_ig_curve(curve, curve_path)
    print(json.dumps({
        "episode": str(episode_path.resolve()),
        "curve": str(curve_path.resolve()),
        "iteration": args.iteration,
        "old_version": args.old,
        "new_version": args.new,
        "n_decisions": episode["n_decisions"],
        "finite_kl_mean": episode["finite_kl_mean"],
        "unchanged_ratio": episode["unchanged_ratio"],
        "infinite_ratio": episode["infinite_ratio"],
        "missing_ratio": episode["missing_ratio"],
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_loop(args) -> int:
    from .loop import run_loop
    from .loop_config import LoopConfig

    config = LoopConfig.from_toml(args.config)
    run_dir = run_loop(config, data_dir=args.data_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "run_dir": str(run_dir.resolve()),
        "run_id": summary["run_id"],
        "status": summary["status"],
        "score_curve": str((run_dir / "score_curve.json").resolve()),
        "ig_curve": str((run_dir / "ig_curve.json").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miracle", description="Miracle 对战、回放与严格 KL 状态工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    names = sorted(AGENTS)

    match = sub.add_parser("match", help="运行一局官方逻辑对战")
    match.add_argument("--agent0", default="sample", choices=names, help="先手 Agent")
    match.add_argument("--agent1", default="endround", choices=names, help="后手 Agent")
    match.add_argument("--seed", type=int, default=11)
    match.add_argument(
        "--output-dir", type=Path,
        default=Path("agentbench_data") / "replays" / "24_miracle",
        help=".mrc 与 .trace.jsonl 的统一输出目录",
    )
    match.add_argument("--tag", default="", help="写入文件名的简短标识")
    match.set_defaults(func=_cmd_match)

    replay = sub.add_parser("replay", help="解析官方 .mrc 回放")
    replay.add_argument("--path", type=Path, required=True)
    replay.add_argument("--jsonl", type=Path, help="可选的事件时间线输出路径")
    replay.set_defaults(func=_cmd_replay)

    ig = sub.add_parser("ig", help="在真实 trace 上比较新旧确定性策略")
    ig.add_argument("--trace", type=Path, required=True, help="match 生成的 .trace.jsonl")
    ig.add_argument("--old", required=True, choices=names, help="更新前 Agent 版本")
    ig.add_argument("--new", required=True, choices=names, help="更新后 Agent 版本")
    ig.add_argument("--camp", type=int, choices=(0, 1), required=True, help="trace 中待比较的阵营")
    ig.add_argument("--iteration", type=int, required=True, help="新版本 iteration，须大于 0")
    ig.add_argument(
        "--output-dir", type=Path,
        default=Path("agentbench_data") / "ig" / "24_miracle",
        help="episode IG 与 ig_curve.json 的统一输出目录",
    )
    ig.set_defaults(func=_cmd_ig)

    loop = sub.add_parser("loop", help="执行一次可追溯的 LLM 策略迭代 Run")
    loop.add_argument("--config", type=Path, required=True, help="Loop TOML 配置")
    loop.add_argument(
        "--data-dir", type=Path,
        help="结果根目录；默认读取 AGENTBENCH_DATA，否则使用 agentbench_data",
    )
    loop.set_defaults(func=_cmd_loop)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ig" and args.iteration <= 0:
        raise SystemExit("--iteration 必须大于 0")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

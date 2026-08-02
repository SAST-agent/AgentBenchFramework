"""CLI（要求 1/3/4/5 的操作入口）：

    python -m agentbench_frame.miracle.cli match    --agent sample --seed 11
    python -m agentbench_frame.miracle.cli evaluate --agent sample_v2 --seed 11 --iteration 1
    python -m agentbench_frame.miracle.cli iterate  --agents sample,sample_v2 --seed 11
    python -m agentbench_frame.miracle.cli replay   --path <xxx.mrc>
    python -m agentbench_frame.miracle.cli versions
    python -m agentbench_frame.miracle.cli curves   --agent sample
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_match(args) -> int:
    from .agent_bridge import EndRoundAgent
    from .iterate import AGENTS
    from .match import run_match
    agent = AGENTS[args.agent]()
    match = run_match(agent, EndRoundAgent(), seed=args.seed,
                      tag=f"cli_{args.agent}_seed{args.seed}")
    print(json.dumps({
        "winner": match.winner, "scores": list(match.scores),
        "rounds": match.rounds, "terminated_by": match.terminated_by,
        "replay": match.replay_path, "trace": match.trace_path,
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_evaluate(args) -> int:
    from .iterate import run_iteration, export_run
    ev = run_iteration(args.agent, seed=args.seed, iteration=args.iteration)
    if args.save:
        export_run(ev, agent_dir_name=args.agent_dir)
    print(json.dumps({
        "iteration": ev.iteration, "version": ev.version,
        "score": ev.score, "winner": ev.match.winner,
        "rounds": ev.match.rounds,
        "ig": ev.ig.get("episode_kl") if ev.ig else None,
        "ig_missing": ev.ig.get("missing") if ev.ig else {},
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_iterate(args) -> int:
    from .loop import run_loop
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    run_loop(agents, seed=args.seed, agent_dir_name=args.agent_dir)
    print(f"iterate 完成：{agents}，events 见 agentbench_data/events/24_miracle/iterations.jsonl")
    return 0


def _cmd_replay(args) -> int:
    from .replay import parse_replay, summarize, save_replay_json
    events = parse_replay(args.path)
    s = summarize(events)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.jsonl:
        out = args.jsonl
        save_replay_json(events, out)
        print(f"事件时间线已写: {out}")
    return 0


def _cmd_versions(args) -> int:
    from .versions import list_versions
    for v in list_versions(game=args.game):
        print(json.dumps(v, ensure_ascii=False))
    return 0


def _cmd_curves(args) -> int:
    from .iterate import build_curves, curves_ascii
    curves = build_curves(agent=args.agent, game=args.game)
    print(curves_ascii(curves))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(curves, f, ensure_ascii=False, indent=2)
        print(f"曲线 JSON 已写: {args.json}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="miracle", description="Miracle 24 届迭代工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("match", help="跑一局对战（官方逻辑子进程）")
    pm.add_argument("--agent", default="sample", choices=["sample", "sample_v2"])
    pm.add_argument("--seed", type=int, default=11)
    pm.set_defaults(func=_cmd_match)

    pe = sub.add_parser("evaluate", help="单次迭代评测（含 IG 计算）")
    pe.add_argument("--agent", default="sample_v2", choices=["sample", "sample_v2"])
    pe.add_argument("--seed", type=int, default=11)
    pe.add_argument("--iteration", type=int, default=1)
    pe.add_argument("--agent-dir", default="sample")
    pe.add_argument("--no-save", action="store_true")
    pe.set_defaults(func=_cmd_evaluate)

    pi = sub.add_parser("iterate", help="迭代编排：改策略→存版本→再评测")
    pi.add_argument("--agents", default="sample,sample_v2")
    pi.add_argument("--seed", type=int, default=11)
    pi.add_argument("--agent-dir", default="sample")
    pi.set_defaults(func=_cmd_iterate)

    pr = sub.add_parser("replay", help="解析 .mrc 二进制回放")
    pr.add_argument("--path", required=True)
    pr.add_argument("--jsonl")
    pr.set_defaults(func=_cmd_replay)

    pv = sub.add_parser("versions", help="列出版本快照")
    pv.add_argument("--game", default="24_miracle")
    pv.set_defaults(func=_cmd_versions)

    pc = sub.add_parser("curves", help="score/IG 曲线")
    pc.add_argument("--agent", default="sample")
    pc.add_argument("--game", default="24_miracle")
    pc.add_argument("--json")
    pc.set_defaults(func=_cmd_curves)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

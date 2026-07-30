"""
Multi-model HL comparison driver.

Runs the existing HL iteration loop once per configured model, each into its
own round directory (``<experiment>/<model_label>/``) so streams never
collide, under identical conditions (same ν, opponents, seed candidate). Then
renders the per-step policy-KL overlay via ``plot_kl_compare``.

One model's failure (every act errors, or it cannot construct a client) is
logged and skipped — it never aborts the experiment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger(__name__)


def run_one_round(*, label: str, entry, experiment: str, acts: int,
                  codebase_root: Path, hl_args: Dict[str, Any]) -> Path:
    """Run one model's HL loop for ``acts`` acts.

    Reuses ``hl/cli.py`` internals (opponent resolution, evaluator factory,
    seeding, system prompt, context builder) and ``HLIterationController``.
    Returns the path to that model's ``events.jsonl``. Raises on hard failure;
    the caller isolates failures per model.
    """
    import sys
    from agentbench_frame.hl import cli as hl_cli
    from agentbench_frame.hl.codebase import HLCodebase
    from agentbench_frame.hl.controller import HLIterationController
    from agentbench_frame.hl.context import ContextBuilder
    from agentbench_frame.hl.llm import build_client
    from agentbench_frame.hl.reference import BenchmarkSpec, ReferenceStateSet
    from agentbench_frame.hl.runner import ApiCodingRunner
    from agentbench_frame.tracking.run import _data_root

    round_root = codebase_root / experiment / label
    workspace = round_root / "workspace"
    store = round_root / "store"
    stage_root = round_root / "stage"
    events_path = round_root / "events.jsonl"
    workspace.mkdir(parents=True, exist_ok=True)

    hl_cli._seed_codebase(hl_args["initial_candidate"], workspace)
    codebase = HLCodebase(root=workspace, store=store)
    spec = BenchmarkSpec(
        spec_id=f"hl-{experiment}-{label}",
        opponents=tuple(o.name for o in hl_args["opponents"]),
        pairs=hl_args["pairs"], seats=hl_args["seats"],
        timeout=hl_args["timeout"], notes={"map": "mapconf2.map"})
    reference = ReferenceStateSet.load(hl_args["reference"])
    context_builder = ContextBuilder(
        codebase=codebase, data_root=Path(_data_root()), game=hl_cli.GAME,
        agent_name=f"{experiment}-{label}", spec=spec,
        playback_skill_path=hl_args["playback_skill_path"],
        timeout=hl_args["claude_timeout"], experience=None)
    runner = ApiCodingRunner(client=build_client(entry),
                             system_prompt=hl_cli._system_prompt(),
                             max_turns=hl_args["max_turns"],
                             timeout=hl_args["claude_timeout"])
    eval_factory = hl_cli._evaluator_factory(
        hl_args["logic_command"], hl_args["opponents"], hl_args["filler"],
        codebase=codebase, stage_root=stage_root,
        data_root=Path(_data_root()))
    ctrl = HLIterationController(
        codebase=codebase, runner=runner, spec=spec, reference=reference,
        run_id=f"{experiment}-{label}", events_path=events_path,
        epsilon=hl_args["epsilon"], evaluator_factory=eval_factory,
        stage_root=stage_root, context_builder=context_builder,
        curriculum=hl_args.get("curriculum", False))
    version = None
    for _ in range(acts):
        version = ctrl.act(version_before=version)
    return events_path


def run_experiment(*, models: Dict[str, Any], experiment: str, acts: int,
                   codebase_root: Path, hl_args: Dict[str, Any]) -> Dict[str, Path]:
    """Run every model in ``models`` for ``acts`` acts. Per-model failures are
    logged and skipped. Returns {label: events.jsonl} for models that completed."""
    out: Dict[str, Path] = {}
    for label, entry in models.items():
        try:
            out[label] = run_one_round(
                label=label, entry=entry, experiment=experiment, acts=acts,
                codebase_root=codebase_root, hl_args=hl_args)
            log.info("model %s done -> %s", label, out[label])
        except Exception as e:
            log.warning("model %s FAILED, skipping: %s: %s",
                        label, type(e).__name__, e)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from agentbench_frame.hl.models_config import load_models

    p = argparse.ArgumentParser(prog="python -m agentbench_frame.hl.compare")
    p.add_argument("--models", default="all", help="'all' or comma-list of keys")
    p.add_argument("--experiment", required=True)
    p.add_argument("--acts", type=int, default=4)
    p.add_argument("--logic", required=True)
    p.add_argument("--logic-python", default=None)
    p.add_argument("--reference", required=True)
    p.add_argument("--ladder-opponent", action="append", default=[])
    p.add_argument("--opponent", action="append", default=[])
    p.add_argument("--filler", default=None)
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--seats", choices=("all", "0", "1", "2", "3"), default="0")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--claude-timeout", type=float, default=600.0)
    p.add_argument("--plot", action="store_true", default=True,
                   help="render the KL overlay after runs (default on)")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    args = p.parse_args(argv)

    from agentbench_frame.hl import cli as hl_cli
    codebase_root = Path.cwd() / ".hl_codebase"
    all_models = load_models()
    if args.models == "all":
        models = all_models
    else:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        models = {k: all_models[k] for k in keys if k in all_models}
    if not models:
        raise SystemExit("[compare] no models selected")

    logic_command = hl_cli._rewrite_logic_python(args.logic, args.logic_python)
    hl_cli._probe_logic_antlr4(logic_command)
    opponents = hl_cli._resolve_opponents(args)
    filler = args.filler or hl_cli._default_filler_command()
    skill_path = (hl_cli._PLAYBACK_SKILL if hl_cli._PLAYBACK_SKILL.exists()
                  else hl_cli._PLAYBACK_SKILL_FALLBACK)
    initial = (Path(hl_cli.__file__).resolve().parent.parent
               / "lostspace" / "candidates" / "v1")
    hl_args = dict(logic_command=logic_command, opponents=opponents, filler=filler,
                   pairs=args.pairs, seats=args.seats, timeout=args.timeout,
                   reference=args.reference, max_turns=args.max_turns,
                   epsilon=args.epsilon, claude_timeout=args.claude_timeout,
                   curriculum=False, playback_skill_path=skill_path,
                   initial_candidate=initial)

    streams = run_experiment(models=models, experiment=args.experiment,
                             acts=args.acts, codebase_root=codebase_root,
                             hl_args=hl_args)
    print(f"[compare] completed models: {list(streams)}")
    if args.plot and streams:
        from agentbench_frame.hl import plot_kl_compare
        out = plot_kl_compare.render(
            experiment_dir=codebase_root / args.experiment,
            out_dir=codebase_root / args.experiment / "figures")
        print(f"[compare] overlay -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

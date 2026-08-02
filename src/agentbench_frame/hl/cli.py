"""CLI driver for the Heuristic-Learning iteration loop.

Drives one HL run end-to-end: seed a codebase from an initial candidate, build
the frozen BenchmarkSpec + load the frozen ReferenceStateSet ν, wire the real
``claude`` CLI (via ``ClaudeCodeRunner``) + a context builder that feeds the
coding agent match history / replay pointers / the playback recipe, then run
N acts through ``HLIterationController``.

Example::

    cd AgentBenchFramework
    export PYTHONPATH=src
    export AGENTBENCH_DATA=./agentbench_data
    python -m agentbench_frame.hl \\
      --logic "cd /d \"$BACKEND/gamecode_logic\" && python main.py" \\
      --initial-candidate ./candidates/v1 \\
      --name hl-v1 \\
      --reference ./agentbench_data/reference/nu-v1.json \\
      --ladder-opponent rank=6 --ladder-opponent rank=12 \\
      --acts 5 --pairs 3 --seats 0 --timeout 15 \\
      --dangerously-skip-permissions

See ``hl/README.md`` for the full operator guide (env setup, run procedure,
data layout, customizing opponents/ν/acts, gotchas).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

from agentbench_frame.lostspace.evaluator import LostSpaceEvaluator, Opponent
from agentbench_frame.lostspace import ladder


GAME = "25_lostspace"
_PLAYBACK_SKILL = (
    Path(__file__).resolve().parents[4]   # E:\HL_Agent
    / "AgentBenchResults" / "skills" / "lostspace-playback" / "SKILL.md"
)
# Fallback if the sibling AgentBenchResults tree isn't present.
_PLAYBACK_SKILL_FALLBACK = (
    Path(__file__).resolve().parent.parent / "lostspace" / "replay_format.md"
)


def _default_filler_command() -> str:
    import shlex, subprocess
    sample = (Path(__file__).resolve().parent.parent / "lostspace"
              / "baselines" / "sample_ai" / "main.py")
    parts = [sys.executable, str(sample)]
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(p) for p in parts)


def _rewrite_logic_python(logic_command: str, logic_python: str) -> str:
    """Rewrite the ``python`` token in a logic command to ``logic_python``.

    The ``--logic`` string is typically ``cd /d "$BACKEND/gamecode_logic" &&
    python main.py``. We swap the bare ``python`` (which may resolve to an
    interpreter without ``antlr4``) for an explicit interpreter path. Only
    a bare ``python``/``pythonw`` token is rewritten — a quoted/absolute
    path is left untouched (the user already chose an interpreter).

    No-op if ``logic_python`` is falsy.
    """
    if not logic_python:
        return logic_command
    import re
    # Replace a bare 'python' or 'python.exe' word boundary that is the
    # command to run (not inside a path). Conservative: only the first
    # bare-python token after any 'cd ... && ' prefix.
    pattern = re.compile(r"(?<![\w./\\])python(?:\.exe)?\b")
    new = pattern.sub(lambda m: logic_python, logic_command, count=1)
    return new


def _probe_logic_antlr4(logic_command: str) -> None:
    """Fail fast if the logic interpreter can't ``import antlr4``.

    The LostSpace logic (``gamecode_logic``) imports ``antlr4`` at startup;
    if the interpreter the ``--logic`` command resolves to lacks
    ``antlr4-python3-runtime==4.9.*``, the logic subprocess dies on import
    and every eval match errors with "logic exited while reading 4 bytes" —
    a silent all-error run. This probe turns that into a fast, named failure
    before any act runs.

    Extracts the python interpreter from the ``--logic`` command (the first
    ``python``-like token after stripping any ``cd ... &&`` prefix) and runs
    ``<python> -c "import antlr4"``. Exits non-zero with an actionable
    message on failure.
    """
    import shlex, subprocess
    from agentbench_frame.lostspace.match import _split_cwd, _to_argv, _child_env

    rest, cwd = _split_cwd(logic_command)
    argv = _to_argv(rest) if sys.platform.startswith("win") else shlex.split(rest)
    if not argv:
        return  # nothing to probe
    # The interpreter is the first token of the command (after the cd prefix).
    # Common cases: ``python``, ``python.exe``, ``C:/.../python.exe``. We
    # only probe when it looks like a python interpreter — a bare ``main`` or
    # ``echo`` is not python and probing it would be meaningless.
    py_token = argv[0]
    base = os.path.basename(py_token).lower()
    is_python = (
        base in ("python", "python.exe", "pythonw", "pythonw.exe")
        or base.startswith("python")             # python3, python3.10, ...
        or base.startswith("python3")             # explicit guard
        or base.endswith("python.exe")            # absolute conda/venv path
        or base.endswith("python")
    )
    if not is_python:
        # Not a python-launched logic (e.g. a compiled binary); skip the probe.
        return
    try:
        # Use the harness's own child env: it scrubs PYTHONHOME/PYTHONPATH
        # that ``uv run`` poisons, which would otherwise break the logic
        # interpreter's importlib (false-negative probe under uv).
        proc = subprocess.run(
            [py_token, "-c", "import antlr4; print('antlr4 ok')"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
            env=_child_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise SystemExit(
            f"[hl] logic interpreter probe failed: could not run "
            f"{py_token!r}: {e}"
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"[hl] logic interpreter {py_token!r} cannot import antlr4.\n"
            f"  The LostSpace logic needs antlr4-python3-runtime==4.9.*.\n"
            f"  Install it in that interpreter, or pass --logic-python PATH "
            f"pointing at an interpreter that has it.\n"
            f"  probe stderr: {proc.stderr.strip()[:300]}"
        )


#: Named frozen eval pools (doc Fix-E item 2): a reusable, cross-version-fixed
#: opponent set so different runs share the same benchmark. The pool name is
#: recorded in ``BenchmarkSpec.notes["eval_pool"]``.
EVAL_POOLS: Dict[str, Tuple[str, ...]] = {
    "default": ("rank01", "rank03", "rank06", "rank09"),
}


def _opponents_from_selectors(selectors: List[str]) -> List[Opponent]:
    opponents: List[Opponent] = []
    for selector in selectors:
        entry = ladder.resolve(selector)
        cmd, _cwd = ladder.launch_command(entry)
        opponents.append(Opponent(name=f"rank{entry.rank:02d}", command=cmd))
    return opponents


def _resolve_opponents(args) -> List[Opponent]:
    opponents: List[Opponent] = []
    eval_pool = getattr(args, "eval_pool", None)
    if eval_pool:
        if eval_pool not in EVAL_POOLS:
            raise SystemExit(
                f"--eval-pool {eval_pool!r} unknown; available: "
                f"{sorted(EVAL_POOLS)}")
        opponents = _opponents_from_selectors(list(EVAL_POOLS[eval_pool]))
    else:
        opponents = _opponents_from_selectors(list(args.ladder_opponent or []))
    for spec in args.opponent or []:
        if "=" not in spec:
            raise SystemExit(f"--opponent must be NAME=COMMAND, got {spec!r}")
        name, command = spec.split("=", 1)
        opponents.append(Opponent(name=name, command=command))
    if not opponents:
        raise SystemExit("at least one of --ladder-opponent or --opponent "
                         "(or --eval-pool) is required")
    names = [o.name for o in opponents]
    if len(set(names)) != len(names):
        raise SystemExit("opponent names must be unique")
    return opponents


def _seed_codebase(initial: Path, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for src in initial.rglob("*"):
        if src.is_dir():
            continue
        if any(part in {"__pycache__", ".cache", ".git", ".pytest_cache"}
               for part in src.relative_to(initial).parts):
            continue
        rel = src.relative_to(initial)
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    if not (workspace / "agent.py").exists():
        raise SystemExit(f"--initial-candidate must contain agent.py "
                         f"(checked {workspace})")


def _evaluator_factory(logic_command: str, opponents: List[Opponent],
                       filler_command: str, *, codebase, stage_root, data_root):
    """Build a fresh LostSpaceEvaluator per evaluated version."""
    from agentbench_frame.hl.adapter import candidate_command

    def make(*, version, spec, run_id, opponent_names=None):
        cmd, cwd = candidate_command(
            version, store=codebase.store,
            dest=stage_root / f"eval-{version.version_id}",
        )
        # candidate_command returns (argv, cwd); the evaluator wants a shell
        # string run with cwd at the staged dir.
        import shlex, subprocess
        if sys.platform.startswith("win"):
            cmd_str = subprocess.list2cmdline(cmd)
        else:
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
        # Curriculum (B1): optionally restrict this eval to a subset of the
        # configured opponents (one tier). Empty subset falls back to all.
        opps = opponents
        if opponent_names:
            filtered = [o for o in opponents if o.name in opponent_names]
            opps = filtered or opponents
        return LostSpaceEvaluator(
            logic_command=logic_command,
            # Write runs under the stable run name (args.name), not
            # 'hl-<version_id>'. ContextBuilder/MatchHistoryView read under
            # args.name, so any other value leaves the per-act prompt with an
            # empty match-history table and no replay path (the agent edits
            # blind). version_id stays in run.toml/summary.json for provenance.
            candidate_name=run_id,
            candidate_command=cmd_str,
            opponents=opps,
            filler_command=filler_command,
            pairs=spec.pairs, seats=spec.seats, timeout=spec.timeout,
            data_dir=data_root,
            save_replays=True,
        )
    return make


def _system_prompt() -> str:
    return (
        "You are a heuristic-learning coding agent iterating on a LostSpace "
        "game AI. Each invocation is ONE improvement act. Read the prompt's "
        "match history and at most one replay, diagnose the weakest matchup, "
        "and make a SMALL surgical edit to agent.py via the str_replace tool "
        "(one unique old_string -> new_string change per act; never rewrite "
        "the whole file). Never touch manifest.toml. "
        "Never break the Saiblo stdio protocol (4-byte big-endian length "
        "prefix + UTF-8 JSON). Leave the agent runnable. "
        "DATA-SAFETY RULE: never gate an action you emit (interact/attack/"
        "move/use_tool) on a membership or equality check against data whose "
        "RUNTIME type you have not confirmed from the code that produces it. "
        "In LostSpace `self.view.nodes[i].interprops` is a list of INTEGER "
        "CODES / objects (1=EscapeCapsule, 2=KeyMachine; the client also "
        "appends the string 'Box'), NOT a list of strings. So a guard like "
        "`if 'KeyMachine' in interprops` is ALWAYS False and silently disables "
        "key collection. Prefer the codebase's existing blind-call-then-check "
        "pattern: call `self.interact('KeyMachine')` and branch on "
        "`result['success']` (the server returns success only when the action "
        "is actually valid)."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agentbench_frame.hl",
        description="Drive the Heuristic-Learning iteration loop with a real "
                    "coding agent (Claude Code).",
    )
    p.add_argument("--logic", required=True,
                   help="official logic command; cwd must be gamecode_logic/. "
                        "Use a python interpreter that has "
                        "antlr4-python3-runtime==4.9.*, or pass "
                        "--logic-python to rewrite the python token.")
    p.add_argument("--logic-python", default=None,
                   help="rewrite the bare 'python' in --logic to this "
                        "interpreter (must have antlr4-python3-runtime==4.9.*). "
                        "Example: C:/Users/.../.conda/envs/torchy/python.exe")
    p.add_argument("--initial-candidate", default=None, type=Path,
                   help="dir seeded into the codebase (must contain agent.py). "
                        "Default: lostspace/candidates/v1 (official sample AI).")
    p.add_argument("--name", default=None,
                   help="HL agent / round name (-> runs/25_lostspace/<name>/). "
                        "If omitted, auto-generated as "
                        "hl-v<YY-MM-DD>-round<n> with a monotonic n persisted "
                        "in .hl_codebase/hl_state.json.")
    p.add_argument("--reference", required=True, type=Path,
                   help="path to a frozen ReferenceStateSet JSON (nu)")
    p.add_argument("--ladder-opponent", action="append", default=[],
                   metavar="rank=NAME",
                   help="ranked human algorithm (rank=6, rank=omegafantasy, "
                        "or rank=最终幻想). Repeatable.")
    p.add_argument("--opponent", action="append", default=[],
                   metavar="NAME=COMMAND",
                   help="explicit opponent as NAME=COMMAND. Repeatable.")
    p.add_argument("--filler", default=None,
                   help="command to pad empty seats (default: bundled sample AI)")
    p.add_argument("--acts", type=int, default=5)
    p.add_argument("--min-kl", type=float, default=0.0,
                   help="early-stop: from --stall-after acts on, if every "
                        "trailing policy_kl.ig is below this (or missing), "
                        "stop the loop (0 disables). Default 0.")
    p.add_argument("--stall-after", type=int, default=5,
                   help="act index from which the --min-kl early-stop is "
                        "active. Default 5.")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=3,
                   help="match pairs per eval. 4-player FFA is noisy; 5+ is "
                        "recommended so a real improvement isn't drowned in "
                        "variance over a 2-pair sample.")
    p.add_argument("--curriculum", action="store_true",
                   help="evaluate one opponent tier at a time, weakest first "
                        "(order of --ladder-opponent), promoting to the next "
                        "tier once avg_rank <= --promote-rank. Avoids throwing "
                        "a weak agent straight at the strongest opponent.")
    p.add_argument("--promote-rank", type=float, default=2.0,
                   help="avg_rank threshold to advance to the next curriculum "
                        "tier (default 2.0 = must reach 2nd place or better).")
    p.add_argument("--consolidate-every", type=int, default=4,
                   help="every K-th act is a consolidation pass (compress & "
                        "re-summarize, no new behavior) instead of a piling "
                        "edit (HL std 4). 0 disables. Default 4.")
    p.add_argument("--max-growth-pct", type=float, default=40.0,
                   help="code-growth nudge threshold: surface a 'consider "
                        "consolidation' nudge when agent.py grows more than "
                        "this many percent over the last piling acts. "
                        "Default 40.")
    p.add_argument("--no-experience", action="store_true",
                   help="disable the self-summarized EXPERIENCE.md store "
                        "(HL std 5). By default it is enabled.")
    p.add_argument("--seats", choices=("all", "0", "1", "2", "3"), default="0")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--spec-id", default=None,
                   help="benchmark spec id (default: hl-<name>)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--codebase-root", type=Path, default=None,
                   help="default: ./.hl_codebase/<name>")
    p.add_argument("--model-key", default=None,
                   help="model key from .env MODELS= to run (default: first in MODELS)")
    p.add_argument("--max-turns", type=int, default=6,
                   help="max tool-use turns per act for the API runner (default 6)")
    p.add_argument("--claude-timeout", type=float, default=600.0,
                   help="per-act API runner wall-clock timeout (seconds)")
    p.add_argument("--rules-validation", action="store_true",
                   help="before the edit loop, run one REPLAY_SKILL validation "
                        "act (doc Fix-D): the coding agent parses a real replay "
                        "and reports its score_dic; the harness cross-checks the "
                        "claim against the replay's actual r[-1] and writes a "
                        "rules_validation event + RULES_VALIDATION.md. Needs "
                        "--data-dir/AGENTBENCH_DATA for replay lookup.")
    p.add_argument("--eval-pool", default=None,
                   help="named frozen eval pool (doc Fix-E): a fixed, "
                        "reusable opponent set recorded in the BenchmarkSpec. "
                        f"Available: {sorted(EVAL_POOLS)}. Overrides "
                        "--ladder-opponent/--opponent for the pool ranks.")
    return p


def _kl_stalled(events_path, min_kl: float, window: int = 3) -> bool:
    """True when the last ``window`` policy_kl events are all below ``min_kl``
    (or unmeasurable) — consecutive acts with no valid policy update."""
    from agentbench_frame.hl.events import read_events

    kls = [e for e in read_events(events_path)
           if e.get("event_type") == "policy_kl"]
    if len(kls) < 2:
        return False
    for e in kls[-window:]:
        ig = e.get("ig")
        if ig is None or ig >= min_kl:
            return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    from agentbench_frame.tracking.run import _data_root
    from agentbench_frame.hl.codebase import HLCodebase
    from agentbench_frame.hl.controller import HLIterationController
    from agentbench_frame.hl.context import ContextBuilder
    from agentbench_frame.hl.reference import BenchmarkSpec, ReferenceStateSet
    from agentbench_frame.hl.events import read_events

    args = build_parser().parse_args(argv)
    data_root = Path(args.data_dir) if args.data_dir else Path(_data_root())

    # Resolve the round name + codebase root. ``--name`` overrides (manual,
    # skips the counter); omit -> auto-generate ``hl-v<YY-MM-DD>-round<n>`` with
    # a global monotonic n persisted in <hl_root>/hl_state.json. Each round
    # reseeds from the fixed canonical candidate (candidates/v1 by default).
    from agentbench_frame.hl.naming import round_name, today_date
    from agentbench_frame.hl.round_state import next_round
    if args.codebase_root:
        codebase_root = Path(args.codebase_root)
        hl_root = codebase_root.parent
    else:
        hl_root = Path.cwd() / ".hl_codebase"
    if args.name:
        name = args.name
    else:
        name = round_name(today_date(), next_round(hl_root))
    if not args.codebase_root:
        codebase_root = hl_root / name

    print(f"[hl] name          = {name}", file=sys.stderr)
    workspace = codebase_root / "workspace"
    store = codebase_root / "store"
    stage_root = codebase_root / "stage"
    events_path = codebase_root / "events.jsonl"

    print(f"[hl] data_root      = {data_root}", file=sys.stderr)
    print(f"[hl] codebase_root  = {codebase_root}", file=sys.stderr)
    print(f"[hl] events.jsonl   = {events_path}", file=sys.stderr)

    opponents = _resolve_opponents(args)
    filler = args.filler or _default_filler_command()

    # Resolve the logic interpreter BEFORE seeding/running: rewrite the bare
    # 'python' token to --logic-python if given, then probe antlr4 so a
    # missing dep fails fast instead of producing a silent all-error eval.
    logic_command = _rewrite_logic_python(args.logic, args.logic_python)
    if logic_command != args.logic:
        print(f"[hl] logic_python   = {args.logic_python} (rewrote python token)",
              file=sys.stderr)
    _probe_logic_antlr4(logic_command)
    print(f"[hl] logic antlr4    = OK", file=sys.stderr)

    # Fixed canonical seed: default to the official sample AI (candidates/v1).
    initial = args.initial_candidate
    if initial is None:
        initial = (Path(__file__).resolve().parent.parent
                   / "lostspace" / "candidates" / "v1")
    _seed_codebase(initial, workspace)

    codebase = HLCodebase(root=workspace, store=store)
    spec = BenchmarkSpec(
        spec_id=args.spec_id or f"hl-{args.name}",
        opponents=tuple(o.name for o in opponents),
        pairs=args.pairs, seats=args.seats, timeout=args.timeout,
        notes={"map": "mapconf2.map"},
    )
    if args.eval_pool:
        spec = BenchmarkSpec(
            spec_id=spec.spec_id, opponents=spec.opponents,
            pairs=spec.pairs, seats=spec.seats, timeout=spec.timeout,
            notes={**spec.notes, "eval_pool": args.eval_pool},
        )
    reference = ReferenceStateSet.load(args.reference)

    skill_path = (_PLAYBACK_SKILL if _PLAYBACK_SKILL.exists()
                  else _PLAYBACK_SKILL_FALLBACK)
    # Self-summarized experience store (HL std 5): lives at the round root,
    # outside the snapshot store so it never pollutes the agent.py diff.
    experience = None
    if not args.no_experience:
        from agentbench_frame.hl.experience import ExperienceStore
        experience = ExperienceStore(round_root=codebase_root)
    context_builder = ContextBuilder(
        codebase=codebase, data_root=data_root, game=GAME,
        agent_name=name, spec=spec,
        playback_skill_path=skill_path, timeout=args.claude_timeout,
        experience=experience,
        consolidate_every=args.consolidate_every,
        max_growth_pct=args.max_growth_pct,
    )
    from agentbench_frame.hl.models_config import load_models
    from agentbench_frame.hl.llm import build_client
    from agentbench_frame.hl.runner import ApiCodingRunner
    models = load_models()
    if not models:
        raise SystemExit("[hl] no models configured (check .env / MODELS=)")
    key = args.model_key or next(iter(models))
    if key not in models:
        raise SystemExit(f"[hl] model key {key!r} not in MODELS={list(models)}")
    entry = models[key]
    print(f"[hl] model         = {key} ({entry.provider}/{entry.model})", file=sys.stderr)
    if entry.provider == "openai" and not entry.base_url:
        raise SystemExit(
            f"[hl] model {key!r} (provider=openai) has no {key.upper()}_BASE_URL "
            "in .env; the openai SDK would silently target api.openai.com. "
            "Set the base URL explicitly."
        )
    runner = ApiCodingRunner(
        client=build_client(entry),
        system_prompt=_system_prompt(),
        max_turns=args.max_turns,
        timeout=args.claude_timeout,
    )
    eval_factory = _evaluator_factory(
        logic_command, opponents, filler,
        codebase=codebase, stage_root=stage_root, data_root=data_root,
    )

    run_id = name
    ctrl = HLIterationController(
        codebase=codebase, runner=runner, spec=spec, reference=reference,
        run_id=run_id, events_path=events_path, epsilon=args.epsilon,
        evaluator_factory=eval_factory, stage_root=stage_root,
        context_builder=context_builder.build,
        curriculum=args.curriculum, promote_rank=args.promote_rank,
        experience=experience, data_root=data_root, game=GAME,
    )

    if args.rules_validation:
        print("[hl] rules_validation act ...", file=sys.stderr)
        payload = ctrl.rules_validation_act(version=None)
        print(f"[hl] rules_validation -> {payload.get('validation_status') if payload else None}",
              file=sys.stderr)
        # Re-seed the workspace so a stray tool call by the validation agent
        # cannot pollute the edit loop's first snapshot.
        _seed_codebase(initial, workspace)

    print(f"[hl] running {args.acts} acts against "
          f"{[o.name for o in opponents]}...", file=sys.stderr)
    version = None
    for i in range(args.acts):
        print(f"[hl] act {i+1}/{args.acts} ...", file=sys.stderr, end=" ", flush=True)
        version = ctrl.act(version_before=version)
        print("done" + (f" -> {version.version_id}" if version else " (no version)"),
              file=sys.stderr)
        # Early-stop: from --stall-after acts on, if every trailing policy_kl
        # event is KL<--min-kl (or missing), no valid update is landing — stop
        # and keep the honest stream instead of burning budget on no-ops.
        if (args.min_kl > 0 and (i + 1) >= args.stall_after
                and _kl_stalled(events_path, args.min_kl)):
            print(f"\n[hl] early stop: KL < {args.min_kl} for the last acts "
                  f"(after {args.stall_after}); no valid policy update. "
                  "Keeping the stream as-is.", file=sys.stderr)
            break

    # Auto-render the score/IG iteration curves (doc Fix-E item 1). Best-effort:
    # a missing matplotlib must not fail the run.
    try:
        from agentbench_frame.hl import plot_curves
        data = plot_curves.read_iteration_curves(events_path)
        figures = codebase_root / "figures"
        out = plot_curves.plot(data, figures)
        print(f"[hl] figures     : {', '.join(out.values())}", file=sys.stderr)
    except (Exception, SystemExit) as e:
        print(f"[hl] figures     : skipped ({type(e).__name__})", file=sys.stderr)

    events = read_events(events_path)
    versions = [e for e in events if e["event_type"] == "version"]
    evals = [e for e in events if e["event_type"] == "eval"]
    kls = [e for e in events if e["event_type"] == "policy_kl"]
    final_wr = evals[-1].get("win_rate") if evals else None
    print("\n[hl] === summary ===", file=sys.stderr)
    print(f"  acts        : {len([e for e in events if e['event_type']=='agent_act'])}",
          file=sys.stderr)
    print(f"  versions    : {len(versions)}", file=sys.stderr)
    print(f"  evals       : {len(evals)} (final win_rate={final_wr})",
          file=sys.stderr)
    print(f"  policy_kl   : {len(kls)} measurements", file=sys.stderr)
    print(f"  events.jsonl: {events_path}", file=sys.stderr)
    return 0

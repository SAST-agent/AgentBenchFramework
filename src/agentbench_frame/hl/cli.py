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
from typing import List, Optional, Sequence, Type

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
    from agentbench_frame.lostspace.match import _split_cwd, _to_argv

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
        proc = subprocess.run(
            [py_token, "-c", "import antlr4; print('antlr4 ok')"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
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


def _resolve_opponents(args) -> List[Opponent]:
    opponents: List[Opponent] = []
    for selector in args.ladder_opponent or []:
        entry = ladder.resolve(selector)
        cmd, _cwd = ladder.launch_command(entry)
        opponents.append(Opponent(name=f"rank{entry.rank:02d}", command=cmd))
    for spec in args.opponent or []:
        if "=" not in spec:
            raise SystemExit(f"--opponent must be NAME=COMMAND, got {spec!r}")
        name, command = spec.split("=", 1)
        opponents.append(Opponent(name=name, command=command))
    if not opponents:
        raise SystemExit("at least one of --ladder-opponent or --opponent "
                         "is required")
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

    def make(*, version, spec, run_id):
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
        return LostSpaceEvaluator(
            logic_command=logic_command,
            candidate_name=f"hl-{version.version_id}"[:40],
            candidate_command=cmd_str,
            opponents=opponents,
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
        "and make a small, reasoned edit to agent.py. Never touch manifest.toml. "
        "Never break the Saiblo stdio protocol (4-byte big-endian length "
        "prefix + UTF-8 JSON). Leave the agent runnable."
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
    p.add_argument("--initial-candidate", required=True, type=Path,
                   help="dir seeded into the codebase (must contain agent.py)")
    p.add_argument("--name", required=True,
                   help="HL agent name (-> runs/25_lostspace/<name>/)")
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
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--seats", choices=("all", "0", "1", "2", "3"), default="0")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--spec-id", default=None,
                   help="benchmark spec id (default: hl-<name>)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--codebase-root", type=Path, default=None,
                   help="default: ./.hl_codebase/<name>")
    p.add_argument("--claude-path", default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--permission-mode", default="acceptEdits",
                   choices=("default", "acceptEdits", "bypassPermissions", "plan"))
    p.add_argument("--dangerously-skip-permissions", action="store_true",
                   help="fully autonomous (no permission prompts); use only in "
                        "a trusted sandbox")
    p.add_argument("--claude-timeout", type=float, default=600.0,
                   help="per-act claude CLI timeout (seconds)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from agentbench_frame.tracking.run import _data_root
    from agentbench_frame.hl.codebase import HLCodebase
    from agentbench_frame.hl.controller import HLIterationController
    from agentbench_frame.hl.context import ContextBuilder
    from agentbench_frame.hl.reference import BenchmarkSpec, ReferenceStateSet
    from agentbench_frame.hl.runner import ClaudeCodeRunner
    from agentbench_frame.hl.events import read_events

    args = build_parser().parse_args(argv)
    data_root = Path(args.data_dir) if args.data_dir else Path(_data_root())
    codebase_root = args.codebase_root or (Path.cwd() / ".hl_codebase" / args.name)
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

    _seed_codebase(args.initial_candidate, workspace)

    codebase = HLCodebase(root=workspace, store=store)
    spec = BenchmarkSpec(
        spec_id=args.spec_id or f"hl-{args.name}",
        opponents=tuple(o.name for o in opponents),
        pairs=args.pairs, seats=args.seats, timeout=args.timeout,
        notes={"map": "mapconf2.map"},
    )
    reference = ReferenceStateSet.load(args.reference)

    skill_path = (_PLAYBACK_SKILL if _PLAYBACK_SKILL.exists()
                  else _PLAYBACK_SKILL_FALLBACK)
    context_builder = ContextBuilder(
        codebase=codebase, data_root=data_root, game=GAME,
        agent_name=args.name, spec=spec,
        playback_skill_path=skill_path, timeout=args.claude_timeout,
    )
    runner = ClaudeCodeRunner(
        claude_path=args.claude_path, model=args.model,
        permission_mode=args.permission_mode,
        dangerously_skip_permissions=args.dangerously_skip_permissions,
        system_prompt=_system_prompt(),
    )
    eval_factory = _evaluator_factory(
        logic_command, opponents, filler,
        codebase=codebase, stage_root=stage_root, data_root=data_root,
    )

    run_id = args.name
    ctrl = HLIterationController(
        codebase=codebase, runner=runner, spec=spec, reference=reference,
        run_id=run_id, events_path=events_path, epsilon=args.epsilon,
        evaluator_factory=eval_factory, stage_root=stage_root,
        context_builder=context_builder.build,
    )

    print(f"[hl] running {args.acts} acts against "
          f"{[o.name for o in opponents]}...", file=sys.stderr)
    version = None
    for i in range(args.acts):
        print(f"[hl] act {i+1}/{args.acts} ...", file=sys.stderr, end=" ", flush=True)
        version = ctrl.act(version_before=version)
        print("done" + (f" -> {version.version_id}" if version else " (no version)"),
              file=sys.stderr)

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

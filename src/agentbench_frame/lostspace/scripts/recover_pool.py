#!/usr/bin/env python3
"""Recover usable AIs that the first preflight pass falsely rejected.

Two systematic, Saiblo-faithful fixes recover ~100 submissions:

1. **C++ map fix.** 100 Makefile-built AIs exit ``rc=3`` with
   ``Fail when reading map in sdk`` because their SDK reads ``mapconf2.map``
   from cwd at init (``main.cpp: argc!=2 ? "mapconf2.map" : argv[1]``). The
   manifest does not materialize it (Saiblo's judger supplies the map at run
   time, so submitters omit it from the zip). The harness must do the same:
   copy the logic's canonical ``mapconf2.map`` into each AI's launch cwd.
   This is NOT a cold-spawn false negative — the AI genuinely dies on a missing
   resource; Saiblo runs it because Saiblo provides the map.

2. **antlr4 python fix.** One python AI imports ``antlr4`` (MapGen grammar,
   same dep as the logic backend). The framework ``.venv`` lacks it; run under
   ``D:\\pymol\\python.exe`` which has ``antlr4-python3-runtime 4.9``.

Reads the original ``preflight.tsv``, retries every ``spawn_fail`` sub with the
relevant fix, re-smokes each via a REAL ``match.run_match`` (same filter as
``submissions_pool``), and emits ``recovered_pool.json`` + ``recovered.tsv`` for
merge into the live ``pool.json``.

Usage::

    uv run python -m agentbench_frame.lostspace.scripts.recover_pool
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from agentbench_frame.lostspace.scripts import submissions_pool as sp

_CACHE = sp._CACHE
_MAP_SRC = sp._BACKEND / "src" / "mapconf2.map"
_ANTLR_PY = r"D:\pymol\python.exe"  # has antlr4-python3-runtime 4.9, like the logic
_SAMPLE = sp._SAMPLE


def _retry_targets(pool_root: Path, manifests: Path) -> list[str]:
    """Sub_ids to retry = every manifest NOT already usable.

    Already-usable = union of the original ``pool.json`` (ok) and any prior
    ``recovered_pool.json``. Everything else (spawn_fail / no_launcher /
    build_fail) gets retried with the current (deeper) ``_detect`` + build step.
    """
    usable: set[str] = set()
    for name in ("pool.json", *sorted(p.name for p in pool_root.glob("recovered*_pool.json"))):
        f = pool_root / name
        if f.is_file():
            usable.update(json.loads(f.read_text(encoding="utf-8")).keys())
    all_ids = sorted(p.stem for p in manifests.glob("s_*.json"))
    return [s for s in all_ids if s not in usable]


def _fix_map(root: Path) -> None:
    """Copy the canonical mapconf2.map into the AI launch cwd."""
    dst = root / "mapconf2.map"
    if not dst.exists():
        shutil.copy2(_MAP_SRC, dst)


def _prep_make_deps(root: Path) -> None:
    """Copy build deps the Makefile references but the submission placed
    elsewhere in its tree (Saiblo's build image has them on a fixed path; the
    player's zip nests them under sdk/). Known case: jsoncpp.cpp + json/ headers."""
    makefiles = [p for p in root.glob("*") if p.name.lower() in ("makefile", "gnumakefile")]
    if not makefiles:
        return
    text = "\n".join(m.read_text(encoding="utf-8", errors="ignore") for m in makefiles)
    # find referenced source deps of the form <path>.cpp / <path>.h
    refs = set(re.findall(r"([\w./-]+\.(?:cpp|cc|cxx|h|hpp))", text))
    for rel in refs:
        target = root / rel
        if target.exists():
            continue
        name = Path(rel).name
        # search the whole submission tree for a same-named candidate
        cands = [p for p in root.rglob(name) if p.is_file()]
        if not cands:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cands[0], target)
        except OSError:
            pass


def _find_exe(root: Path) -> str | None:
    """Find the executable make produced. Prefer main.exe, then any .exe (some
    Makefiles output `ai`/`strategy`/etc.), excluding build artifacts."""
    exes = [p for p in root.glob("*.exe") if p.is_file()]
    if not exes:
        return None
    for pref in ("main.exe", "ai.exe"):
        for p in exes:
            if p.name.lower() == pref:
                return p.name
    # largest .exe (the real binary, not a stray); exclude tiny stubs
    exes.sort(key=lambda p: p.stat().st_size, reverse=True)
    return exes[0].name if exes[0].stat().st_size > 50_000 else None


def _build_make_any(root: Path) -> str | None:
    """Build via Makefile; return the produced exe name or None.
    Many Saiblo Makefiles default to a zip/so artifact target rather than `main`,
    and some output a binary named `ai`/`strategy` (not `main`), so try explicit
    targets and detect whatever executable appears."""
    for variant in (None, "main", "all", "main.exe"):
        before = {p.name for p in root.glob("*.exe")}
        target_args = ["make"] + ([variant] if variant else [])
        try:
            subprocess.run(target_args, cwd=root, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue
        exe = _find_exe(root)
        if exe and exe not in before:
            return exe
    # last resort: any exe present at all
    return _find_exe(root)


def _retry(sub_id: str, logic_cmd: str, sample_cmd: str,
           ai_py: str, smoke_timeout: float) -> tuple[str, str | None, str]:
    """Detect -> build -> apply fixes -> re-smoke. Returns (status, command, smoke_msg)."""
    src = _CACHE / sub_id
    det = sp._detect(src)
    if det is None:
        return "no_launcher", None, "detect failed on retry"
    detkind, root, detail = det
    py_text = ""
    for p in src.rglob("*"):
        if p.suffix in (".cpp", ".h", ".cc", ".cxx", ".py"):
            try:
                py_text += p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
    # Build C/C++ entries (build_fail had no binary; no_launcher-make needs one too).
    binname = None
    if detkind == "make":
        _prep_make_deps(root)
        binname = _build_make_any(root)
        if binname is None:
            return "build_fail", None, "no exe after make/main/all"
    elif detkind == "cpp_bare":
        built, bmsg = sp._build_cpp(root, detail)
        if not built:
            return "build_fail", None, bmsg
        binname = sp._binary_name()
    # Map fix: C++ SDK reads mapconf2.map from cwd; Saiblo's judger supplies it.
    if detkind in ("make", "cpp_bare") and (
        "mapconf2.map" in py_text or "_read_map" in py_text
        or "reading map" in py_text
    ):
        _fix_map(root)
    # Antlr4 fix: a python AI that vendor-copies the logic's MapGen needs antlr4.
    use_py = _ANTLR_PY if (detkind == "python" and "antlr4" in py_text) else ai_py
    if detkind in ("make", "cpp_bare") and binname:
        # build command with the ACTUAL produced exe name (may be ai.exe, not main.exe)
        if sys.platform.startswith("win"):
            cmd = f'cd /d "{root}" && {binname}'
        else:
            cmd = f'cd "{root}" && ./{binname}'
    else:
        cmd = sp._command_for(root, detkind, detail, use_py)
    if cmd is None:
        return "no_launcher", None, "no command"
    ok, msg = sp._smoke_real(cmd, logic_cmd, sample_cmd, smoke_timeout)
    if not ok:
        return "smoke_fail", None, msg
    return "ok", cmd, msg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-root", default=str(
        Path("agentbench_data/pools/lostspace/submissions-251")))
    p.add_argument("--manifests", default=str(sp._MANIFESTS))
    p.add_argument("--ai-python", default=sp._AI_PY_DEFAULT)
    p.add_argument("--logic-python", default=sp._DEFAULT_LOGIC_PY)
    p.add_argument("--smoke-timeout", type=float, default=8.0)
    args = p.parse_args(argv)

    pool_root = Path(args.pool_root)
    targets = _retry_targets(pool_root, Path(args.manifests))
    print(f"Retrying {len(targets)} not-yet-usable submissions -> {pool_root}",
          file=sys.stderr)

    logic_cmd = sp._logic_command(args.logic_python)
    sample_cmd = f'"{args.ai_python}" "{_SAMPLE.resolve()}"'

    pool: dict[str, str] = {}
    rows: list[dict] = []
    from collections import Counter
    c = Counter()
    for i, sid in enumerate(targets):
        status, cmd, msg = _retry(sid, logic_cmd, sample_cmd,
                                  args.ai_python, args.smoke_timeout)
        c[status] += 1
        rows.append({"sub_id": sid, "status": status, "smoke": msg[:90]})
        if status == "ok" and cmd:
            pool[sid] = cmd
        if (i + 1) % 25 == 0 or i + 1 == len(targets):
            print(f"  [{i+1}/{len(targets)}] recovered={len(pool)} ({c})",
                  file=sys.stderr)

    out = pool_root / "recovered4_pool.json"
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    with (pool_root / "recovered4.tsv").open("w", newline="",
                                             encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sub_id", "status", "smoke"],
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== recovery summary ===", file=sys.stderr)
    for k, v in c.most_common():
        print(f"  {k:<14} {v}", file=sys.stderr)
    print(f"\nrecovered {len(pool)} new entries -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

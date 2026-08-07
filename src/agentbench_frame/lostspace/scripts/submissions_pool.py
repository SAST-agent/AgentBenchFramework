#!/usr/bin/env python3
"""Materialize + build + preflight the full 251-player lostspace submission pool.

The 16 ranked finalists in ``top_algorithms/corpus/25_lostspace_final_ladder``
are a curated subset. The full anonymized corpus lives under
``AgentBench/player_submissions`` (content-addressed sha256 blobs + per-
submission manifests). This script turns every manifest in
``manifests/25_lostspace/s_*.json`` into a runnable AI:

1. **materialize** the manifest into a per-submission dir (copy mode, so
   building C++ never mutates the shared content-addressed blobs),
2. **detect** language + entry, searching the root *and* immediate subdirs
   (many zips nest their project under ``Strategy/`` / ``src/`` / ``ShowCry_v2/``),
3. **build** C++ entries (``make`` or direct ``g++``),
4. **smoke** each via a REAL ``match.run_match`` (sub + 3 sample fillers, short
   timeout): the AI must run through a game without raising. This is the
   accurate test — many Saiblo C++ AIs segfault on a cold start with no judger
   init frame but run fine once the harness acts as judger, so a cold
   spawn-check is a false negative.

Emits ``pool.json`` = ``{submission_id: command}`` for every usable entry, plus
``preflight.tsv`` and ``meta.json``. The pool feeds
``pool_tournament.py --pool-file``.

Usage::

    uv run python -m agentbench_frame.lostspace.scripts.submissions_pool
    uv run python -m agentbench_frame.lostspace.scripts.submissions_pool --no-smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

from agentbench_frame.lostspace import match

_REPO = Path(r"E:\HL_Agent\AgentBench")
_MANIFESTS = _REPO / "player_submissions" / "manifests" / "25_lostspace"
_DATASET = _REPO / "player_submissions"
_CACHE = Path(r"E:\HL_Agent\AgentBenchFramework\.cache\submissions")
_SAMPLE = (Path(__file__).resolve().parents[1] / "baselines" / "sample_ai" / "main.py")
_BACKEND = _REPO / "backend_sources" / "corpus" / "25_lostspace" / "logic" / "gamecode_logic"
_DEFAULT_LOGIC_PY = r"D:\pymol\python.exe"

_AI_PY_DEFAULT = sys.executable
_MAKE = os.environ.get("MAKE", "make")


def _safe_output_path(root: Path, relpath: str) -> Path:
    posix = PurePosixPath(relpath)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"unsafe path in manifest: {relpath}")
    return root.joinpath(*posix.parts)


def _materialize(manifest: Path, out: Path) -> int:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fi in data.get("files", []):
        src = _DATASET / fi["object_path"]
        if not src.is_file():
            continue
        dst = _safe_output_path(out, fi["relpath"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        n += 1
    return n


def _detect_in(root: Path) -> tuple[str, object] | None:
    """Detect entry within a specific dir. Returns (kind, detail) or None."""
    if (root / "main.py").is_file():
        return "python", root / "main.py"
    root_py = [p for p in root.iterdir() if p.is_file() and p.suffix == ".py"]
    if len(root_py) == 1:
        return "python", root_py[0]
    if (root / "Makefile").is_file() or (root / "makefile").is_file():
        return "make", None
    cpps = [p for p in root.iterdir() if p.is_file() and p.suffix in (".cpp", ".cc", ".cxx")]
    if cpps:
        return "cpp_bare", cpps
    return None


def _detect(src: Path) -> tuple[str, Path, object] | None:
    """Search src then its immediate subdirs. Returns (kind, effective_root, detail)."""
    d = _detect_in(src)
    if d:
        return d[0], src, d[1]
    # recurse one level (zips often nest under Strategy/ src/ ShowCry_v2/ ...)
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        d = _detect_in(sub)
        if d and d[0] in ("python", "make"):
            return d[0], sub, d[1]
    # fall back to bare cpp in a subdir
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        d = _detect_in(sub)
        if d and d[0] == "cpp_bare":
            return d[0], sub, d[1]
    return None


def _binary_name() -> str:
    return "main.exe" if sys.platform.startswith("win") else "main"


def _build_make(root: Path) -> tuple[bool, str]:
    try:
        subprocess.run([_MAKE], cwd=root, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "make timeout"
    b = root / _binary_name()
    return (b.is_file(), "ok" if b.is_file() else "no binary")


def _build_cpp(root: Path, cpps: list[Path]) -> tuple[bool, str]:
    out = root / _binary_name()
    cmd = ["g++", *[str(c) for c in cpps], "-O2", "-std=c++17", "-o", str(out)]
    try:
        r = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, f"compile error: {exc}"
    return (out.is_file(), "ok" if out.is_file() else r.stdout[-200:])


def _command_for(root: Path, kind: str, detail: object, ai_py: str) -> str | None:
    binname = _binary_name()
    if kind == "python":
        return f'"{ai_py}" "{detail}"'
    if kind in ("make", "cpp_bare"):
        if sys.platform.startswith("win"):
            return f'cd /d "{root}" && {binname}'
        return f'cd "{root}" && ./{binname}'
    return None


def _logic_command(py: str) -> str:
    return f'cd /d "{_BACKEND}" && "{py}" main.py'


def _smoke_real(command: str, logic_cmd: str, sample_cmd: str, timeout: float) -> tuple[bool, str]:
    """Run sub+3 fillers in a real match; usable iff it completes without raising."""
    with tempfile.TemporaryDirectory() as d:
        try:
            res = match.run_match(logic_cmd, [command, sample_cmd, sample_cmd, sample_cmd],
                                  timeout, Path(d) / "r.json", trace_path=None,
                                  media_player_seat=None)
            return True, f"ok turns={res['turns']}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {str(exc)[:80]}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", default=str(_MANIFESTS))
    p.add_argument("--cache", default=str(_CACHE))
    p.add_argument("--ai-python", default=_AI_PY_DEFAULT)
    p.add_argument("--logic-python", default=_DEFAULT_LOGIC_PY)
    p.add_argument("--out", default=None)
    p.add_argument("--out-name", default="submissions-251")
    p.add_argument("--no-smoke", action="store_true",
                   help="skip real-match smoke (materialize+build+detect only)")
    p.add_argument("--smoke-timeout", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    env_data = os.environ.get("AGENTBENCH_DATA")
    out_root = Path(args.out).resolve() if args.out else (
        (Path(env_data).resolve() if env_data else Path.cwd() / "agentbench_data")
        / "pools" / "lostspace")
    out_dir = out_root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(Path(args.manifests).glob("s_*.json"))
    if args.limit:
        manifests = manifests[: args.limit]
    sample_cmd = f'"{args.ai_python}" "{_SAMPLE.resolve()}"'
    logic_cmd = _logic_command(args.logic_python)

    print(f"Processing {len(manifests)} submissions -> {out_dir}", file=sys.stderr)
    pool: dict[str, str] = {}
    meta: dict[str, dict] = {}
    rows: list[dict] = []
    t0 = time.time()
    for i, mf in enumerate(manifests):
        sub_id = mf.stem
        row = {"sub_id": sub_id, "status": "skip", "raw_ext": "", "kind": "",
               "files": "", "build": "", "smoke": ""}
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            row["raw_ext"] = data.get("raw_ext", "")
            src = Path(args.cache) / sub_id
            nfiles = _materialize(mf, src)
            row["files"] = nfiles
            det = _detect(src)
            if det is None:
                row["status"] = "no_launcher"; rows.append(row); continue
            kind, root, detail = det
            row["kind"] = kind
            built, msg = True, ""
            if kind == "make":
                built, msg = _build_make(root)
            elif kind == "cpp_bare":
                built, msg = _build_cpp(root, detail)
            row["build"] = msg
            if not built:
                row["status"] = "build_fail"; rows.append(row); continue
            cmd = _command_for(root, kind, detail, args.ai_python)
            if cmd is None:
                row["status"] = "no_launcher"; rows.append(row); continue
            if args.no_smoke:
                ok, smsg = True, "skipped"
            else:
                ok, smsg = _smoke_real(cmd, logic_cmd, sample_cmd, args.smoke_timeout)
            row["smoke"] = smsg
            if not ok:
                row["status"] = "smoke_fail"; rows.append(row); continue
            pool[sub_id] = cmd
            meta[sub_id] = {"kind": kind, "root": str(root), "raw_ext": row["raw_ext"],
                            "files": nfiles}
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"error: {type(exc).__name__}"
            row["smoke"] = str(exc)[:80]
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(manifests):
            print(f"  [{i+1}/{len(manifests)}] usable={len(pool)} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)

    (out_dir / "pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "preflight.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sub_id", "status", "raw_ext", "kind",
                                           "files", "build", "smoke"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in
                        ["sub_id", "status", "raw_ext", "kind", "files", "build", "smoke"]})

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print("\n=== preflight summary ===", file=sys.stderr)
    for k, v in c.most_common():
        print(f"  {k:<14} {v}", file=sys.stderr)
    print(f"\nusable pool: {len(pool)}/{len(manifests)} -> {out_dir/'pool.json'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

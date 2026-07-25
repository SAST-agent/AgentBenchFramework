#!/usr/bin/env python3
"""Build and smoke-test every ranked ladder algorithm.

For each of the 16 algorithms under
``AgentBench/top_algorithms/corpus/25_lostspace_final_ladder``:

* resolve the entry (building C++ into the cache dir on first use),
* smoke-test it by running one short match with three bundled ``sample_ai``
  fillers,
* print a per-rank pass/fail table.

This keeps the corpus tree clean (builds go to ``.cache/ladder``) and
verifies each algorithm launches and speaks the Saiblo wire protocol.

Usage::

    uv run python -m agentbench_frame.lostspace.scripts.build_ladder
    uv run python -m agentbench_frame.lostspace.scripts.build_ladder --rank 1 3 6
    uv run python -m agentbench_frame.lostspace.scripts.build_ladder --build-only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from agentbench_frame.lostspace import ladder, match

_LOGIC = (
    r"cd /d E:\HL_Agent\AgentBench\backend_sources\corpus"
    r"\25_lostspace\logic\gamecode_logic && D:\pymol\python.exe main.py"
)
_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "sample_ai"
    / "main.py"
)


def _sample_command() -> str:
    return f'"{sys.executable}" "{_SAMPLE}"'


def _smoke(entry: ladder.LadderEntry, timeout: float) -> dict:
    cmd, _cwd = ladder.launch_command(entry)
    sample = _sample_command()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        try:
            result = match.run_match(
                _LOGIC,
                [cmd, sample, sample, sample],
                timeout,
                root / "replay.json",
                root / "trace.jsonl",
            )
        except Exception as exc:  # noqa: BLE001 - report any failure
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "turns": result["turns"],
        "ranking": result["ranking"],
        "winner": result["winner"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank",
        type=int,
        nargs="*",
        help="only smoke-test these ranks (default: all 16)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="just build C++ entries, do not run a smoke match",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    ranks = args.rank if args.rank else ladder.all_ranks()
    print(f"{'rank':>4}  {'user':<14} {'lang':<10} {'result':<8} detail")
    print("-" * 70)
    failures = 0
    for rank in ranks:
        entry = ladder.entries()[rank]
        try:
            if args.build_only:
                if entry.language == "make":
                    ladder.build(entry)
                detail = "built" if entry.language == "make" else "n/a"
                print(f"{rank:>4}  {entry.username:<14} {entry.language:<10} {'OK':<8} {detail}")
                continue
            res = _smoke(entry, args.timeout)
            if res["ok"]:
                detail = f"turns={res['turns']} winner=p{res['winner']} ranking={res['ranking']}"
                print(f"{rank:>4}  {entry.username:<14} {entry.language:<10} {'OK':<8} {detail}")
            else:
                failures += 1
                print(f"{rank:>4}  {entry.username:<14} {entry.language:<10} {'FAIL':<8} {res['error']}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{rank:>4}  {entry.username:<14} {entry.language:<10} {'FAIL':<8} {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("-" * 70)
    print(f"{len(ranks) - failures}/{len(ranks)} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run an immutable Python asset after deterministically seeding ``random``.

The launcher keeps the Judge/AI source byte-identical.  It is only one part of
the frozen-case contract: an authoritative run must separately prove that every
participating process either consumes its assigned seed or is deterministic.
"""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path
from typing import Sequence


def run_seeded_script(script: Path, seed: int, argv: Sequence[str] = ()) -> None:
    script = script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    previous_argv = sys.argv
    previous_random_state = random.getstate()
    try:
        random.seed(seed)
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        random.setstate(previous_random_state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    run_seeded_script(args.script, args.seed, args.script_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

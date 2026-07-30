#!/usr/bin/env python3
"""Seed Python and NumPy before executing the frozen Rollman logic."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic-root", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    root = Path(args.logic_root).resolve()
    entrypoint = root / "main.py"
    if not entrypoint.is_file():
        parser.error(f"missing frozen logic entrypoint: {entrypoint}")
    seed = (
        args.seed
        if args.seed is not None
        else int(os.environ["AGENTBENCH_ROLLMAN_SEED"])
    )
    random.seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(root))
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

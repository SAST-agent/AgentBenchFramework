#!/usr/bin/env python3
"""Seed Python agent randomness before running an opaque entrypoint."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    args = parser.parse_args()

    entrypoint = Path(args.entrypoint).resolve()
    if not entrypoint.is_file():
        parser.error(f"missing agent entrypoint: {entrypoint}")
    seed = int(os.environ["AGENTBENCH_ROLLMAN_SEED"])
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    sys.path.insert(0, str(entrypoint.parent))
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

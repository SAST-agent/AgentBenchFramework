"""Seed and launch an unmodified DOTO server tree."""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    server_dir = args.server_dir.resolve()
    random.seed(args.seed)
    sys.path.insert(0, str(server_dir))
    sys.argv = [str(server_dir / "main.py"), str(args.replay.resolve())]
    runpy.run_path(str(server_dir / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

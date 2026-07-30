#!/usr/bin/env python3
"""Print a machine-readable frozen/upstream Rollman source audit."""

from __future__ import annotations

import argparse
import json

from agentbench_frame.games.rollman.contract import audit_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentbench-root", required=True)
    parser.add_argument("--official-logic-root", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_sources(args.agentbench_root, args.official_logic_root),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

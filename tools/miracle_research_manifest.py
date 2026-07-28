#!/usr/bin/env python3
"""Print the pre-registered 24_miracle research manifest as canonical JSON.

This command is read-only: it does not create a session or start any runtime.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.research_protocol import (
    canonical_research_manifest_bytes,
)


def main() -> int:
    sys.stdout.buffer.write(canonical_research_manifest_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

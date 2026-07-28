"""Environment-driven path resolution for the 24_miracle adapter tools/tests.

No machine-specific absolute paths live in source. Callers set:
  AGENTBENCH_ROOT      - AgentBench corpus root (has backend_sources/ + top_algorithms/)
  MIRACLE_IFELSE_DIR   - 高翔 if-else bot directory (ifelse_bot/)
  AGENTBENCH_RESULTS   - AgentBenchResults repo root (optional; pipeline tests skip if unset)
Falls back with a clear error / None rather than embedding any local path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _required(env: str, what: str) -> Path:
    v = os.environ.get(env)
    if not v:
        raise RuntimeError(f"env {env} not set; point it at the {what}")
    return Path(v)


def agentbench_root() -> Path:
    return _required("AGENTBENCH_ROOT", "AgentBench corpus root (backend_sources/ + top_algorithms/)")


def judge_dir() -> Path:
    return agentbench_root() / "backend_sources" / "corpus" / "24_miracle" / "logic" / "judge_dev_logic"


def sample_ai_dir() -> Path:
    return agentbench_root() / "backend_sources" / "corpus" / "24_miracle" / "logic" / "judge_dev_sample_ai"


def extracted_dir() -> Path:
    return agentbench_root() / "top_algorithms" / "corpus" / "24_miracle_final" / "extracted"


def archives_dir() -> Path:
    return agentbench_root() / "top_algorithms" / "corpus" / "24_miracle_final" / "archives"


def ifelse_dir() -> Path:
    return _required("MIRACLE_IFELSE_DIR", "高翔 if-else bot directory (contains main.py)")


def results_repo() -> Optional[Path]:
    v = os.environ.get("AGENTBENCH_RESULTS")
    return Path(v) if v else None

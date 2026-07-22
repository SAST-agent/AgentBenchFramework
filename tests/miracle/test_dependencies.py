"""Dependency-declaration test: psutil is required by the Miracle adapter and a
missing install must produce an actionable error (not a bare ModuleNotFoundError)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def test_missing_psutil_gives_actionable_error():
    code = (
        "import sys\n"
        "sys.modules['psutil'] = None  # force import psutil to fail\n"
        "import agentbench_frame.games.miracle.proctree\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTHONHOME", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    combined = r.stderr + r.stdout
    assert "psutil" in combined
    assert "uv sync --extra miracle" in combined, combined


def test_pyproject_declares_miracle_extra():
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'miracle = ["psutil"]' in text

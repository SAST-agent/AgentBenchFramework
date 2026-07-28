"""Pytest bootstrap: put the src-layout package on sys.path without an editable
install, so `from agentbench_frame... import ...` resolves in tests and so the
`agentbench_frame.*` modules can be imported by helper scripts.

Run tests with the Python 3.11+ interpreter (the framework imports `tomllib`):
    py -3.13 -m pytest tests/
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

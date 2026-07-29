"""Tests for hl/round_state.py — monotonic round counter + recovery."""
from __future__ import annotations

import json
from pathlib import Path

from agentbench_frame.hl.round_state import (
    next_round, STATE_FILENAME, _state_path,
)


def test_next_round_fresh_then_increment(tmp_path: Path):
    hl = tmp_path / ".hl_codebase"
    assert next_round(hl) == 1
    state = json.loads((hl / STATE_FILENAME).read_text(encoding="utf-8"))
    assert state == {"schema": 1, "last_round": 1}
    assert next_round(hl) == 2
    state = json.loads((hl / STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["last_round"] == 2


def test_next_round_recovers_from_existing_dirs(tmp_path: Path):
    """Missing state file -> recover max round from sibling round dirs."""
    hl = tmp_path / ".hl_codebase"
    hl.mkdir()
    (hl / "hl-v99-07-29-round3").mkdir()  # prior round 3 exists, no state file
    assert not _state_path(hl).exists()
    assert next_round(hl) == 4
    state = json.loads((hl / STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["last_round"] == 4


def test_next_round_state_file_wins_over_dirs(tmp_path: Path):
    """If the state file exists, it is authoritative (dir-scan skipped)."""
    hl = tmp_path / ".hl_codebase"
    hl.mkdir()
    (hl / "hl-v99-07-29-round9").mkdir()  # dir says 9
    _state_path(hl).write_text('{"schema": 1, "last_round": 2}', encoding="utf-8")
    assert next_round(hl) == 3  # state (2) wins, not dir (9)

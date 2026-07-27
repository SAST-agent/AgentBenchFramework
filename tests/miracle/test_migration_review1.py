"""Regression coverage for the portable 24_miracle review-1 migrator."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MIGRATOR = REPO / "tools" / "migrate_24_miracle_to_review1.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MIGRATOR), *args], capture_output=True, text=True
    )


def test_migrator_requires_all_explicit_paths():
    result = _run()
    assert result.returncode == 2
    assert "--session" in result.stderr
    assert "--original-run-dir" in result.stderr
    assert "--output-root" in result.stderr


def test_migrator_refuses_legacy_preview_name(tmp_path):
    session = tmp_path / "session"
    original_run = tmp_path / "results" / "runs" / "game" / "agent" / "rid"
    session.mkdir()
    original_run.mkdir(parents=True)
    legacy = tmp_path / "24_miracle_results_migration_preview_v2"
    result = _run(
        "--session", str(session),
        "--original-run-dir", str(original_run),
        "--output-root", str(legacy),
    )
    assert result.returncode == 2
    assert "legacy preview" in result.stderr
    assert not legacy.exists()


def test_migrator_refuses_output_under_authority_input(tmp_path):
    session = tmp_path / "session"
    original_run = tmp_path / "results" / "runs" / "game" / "agent" / "rid"
    session.mkdir()
    original_run.mkdir(parents=True)
    result = _run(
        "--session", str(session),
        "--original-run-dir", str(original_run),
        "--output-root", str(session / "new_preview"),
    )
    assert result.returncode == 2
    assert "must not be inside input paths" in result.stderr


def test_migrator_source_has_no_machine_specific_user_path():
    source = MIGRATOR.read_text(encoding="utf-8")
    windows_path = "C:" + "\\Users" + "\\gongh"
    posix_path = "C:/" + "Users/gongh"
    assert windows_path not in source
    assert posix_path not in source


def test_run_lifecycle_preserves_historical_execution_time(tmp_path):
    from agentbench_frame.tracking.run import Run

    run = Run.start(
        game="24_miracle", agent="migrated", data_dir=str(tmp_path),
        run_id="historical", append=False,
        created="2026-07-21T16:19:30Z", git_commit="",
        started_at=1784650770.0399437,
        config={"migration_started_at": 1785000000.0},
    )
    run.write("game", game_id="g", valid=True, normalized_result="win", steps=3)
    run.recompute_totals_from_events(game_event_type="game")
    summary = run.finish(finished_at=1784651015.0399437)
    assert summary["created"] == "2026-07-21T16:19:30Z"
    assert summary["started_at"] == 1784650770.0399437
    assert summary["finished_at"] == 1784651015.0399437
    assert summary["wall_hours"] == round((245.0 / 3600), 2)
    run_toml = (tmp_path / "runs" / "24_miracle" / "migrated" / "historical" / "run.toml").read_text()
    assert "started_at = 1784650770.0399437" in run_toml
    assert "finished_at = 1784651015.0399437" in run_toml
    assert "migration_started_at = 1785000000.0" in run_toml

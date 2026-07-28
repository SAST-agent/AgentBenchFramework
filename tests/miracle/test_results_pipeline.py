"""End-to-end Results-pipeline tests (SKILL.md 测试门槛 items 14, 15, 16).

Drive the real framework ``Run`` to produce a contract-complete run, then run
the AgentBenchResults scripts that SKILL.md 本地验收 mandates, and assert:

  14. aggregate.py picks up the 24_miracle run into registry.toml
  15. report_builder.py renders a static site with data.json
  16. web win_rate == summary.json win_rate == win_rate recomputed from events
      (three-way consistency; "CI成功不等于数据正确")

UPSTREAM BUG (documented, not hidden):
  The upstream ``aggregate.py`` writes ``path = str(run_dir.relative_to(...))``,
  which on Windows emits backslashes and makes registry.toml invalid TOML, so
  report_builder aborts. We vendor a one-line portability patch
  (vendor/results_local/aggregate.py, as_posix) for local verification and pin
  the upstream bug with a dedicated regression test below. report_builder.py is
  used unmodified (it works once registry.toml is valid).

    py -3.13 -m pytest tests/miracle/test_results_pipeline.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.driver import feed_outcomes_to_run
from agentbench_frame.games.miracle.result import GameOutcome, finalize
from agentbench_frame.tracking.run import Run

_REPO_ROOT = Path(__file__).resolve().parents[2]
from agentbench_frame.games.miracle.paths import results_repo as _results_repo
RESULTS_REPO = _results_repo() or Path("")
UPSTREAM_AGGREGATE = RESULTS_REPO / "scripts" / "aggregate.py"
UPSTREAM_REPORT_BUILDER = RESULTS_REPO / "scripts" / "report_builder.py"
# local portability-patched copy (as_posix); see header in that file.
LOCAL_AGGREGATE = _REPO_ROOT / "vendor" / "results_local" / "aggregate.py"

pytestmark = pytest.mark.skipif(
    not (LOCAL_AGGREGATE.exists() and UPSTREAM_REPORT_BUILDER.exists()),
    reason="local aggregate + upstream report_builder required",
)


# ---- helpers (self-contained) -------------------------------------------- #
def _outcome(gid, eval_camp, raw_winner, steps=40, **kw):
    s0 = 5 if raw_winner == 0 else 2
    s1 = 2 if raw_winner == 0 else 5
    return finalize(
        GameOutcome(
            game_id=gid, evaluated_agent="ifelse", opponent="rank04",
            evaluated_agent_camp=eval_camp, raw_winner=raw_winner,
            score0=s0, score1=s1, steps=steps,
            replay_path=f"/tmp/{gid}.replay", replay_sha256="x" * 64, **kw,
        )
    )


def _build_run(tmp_path: Path, outcomes):
    run = Run.start(
        game="24_miracle", agent="ifelse", run_type="eval",
        data_dir=str(tmp_path),
        config={"opponent_set": "smoke", "n_games": str(len(outcomes))},
    )
    feed_outcomes_to_run(run, outcomes)
    summary = run.finish()
    return summary, tmp_path / "runs" / "24_miracle" / "ifelse" / run.run_id


def _recompute_win_rate_from_events(run_dir: Path) -> float:
    games = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("event") == "game"
    ]
    valid = [g for g in games if g["normalized_result"] in ("win", "loss", "draw")]
    if not valid:
        return 0.0
    wins = sum(1 for g in valid if g["normalized_result"] == "win")
    return wins / len(valid)


def _run_py(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


# ---- 14: aggregate (local patched copy) ---------------------------------- #
def test_aggregate_picks_up_miracle_run(tmp_path):
    _, run_dir = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    out = _run_py(LOCAL_AGGREGATE, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "registry.toml"))
    assert out.returncode == 0, out.stderr
    reg = tomllib.loads((tmp_path / "registry.toml").read_text())  # parses cleanly
    runs = reg.get("runs", [])
    assert len(runs) == 1
    assert runs[0]["game"] == "24_miracle"
    assert runs[0]["agent"] == "ifelse"
    assert runs[0]["path"].startswith("runs/24_miracle/")  # forward slashes


# ---- 15 + 16: render + three-way win_rate consistency -------------------- #
def test_results_pipeline_renders_and_winrate_is_consistent(tmp_path):
    outcomes = [
        _outcome("g1", 0, 0, steps=40),   # win
        _outcome("g2", 1, 1, steps=38),   # win (swapped)
        _outcome("g3", 0, 1, steps=50),   # loss
        _outcome("g4", 0, 0, steps=10, ai_error_player=1),  # error (excluded)
    ]
    summary, run_dir = _build_run(tmp_path, outcomes)

    agg = _run_py(LOCAL_AGGREGATE, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "registry.toml"))
    assert agg.returncode == 0, agg.stderr

    rep = _run_py(UPSTREAM_REPORT_BUILDER, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "_site"))
    assert rep.returncode == 0, rep.stderr
    assert (tmp_path / "_site" / "data.json").exists()

    summary_wr = summary["win_rate"]
    events_wr = _recompute_win_rate_from_events(run_dir)
    site = json.loads((tmp_path / "_site" / "data.json").read_text())
    web_wr = site["runs"][0]["metrics"]["win_rate"]

    assert events_wr == pytest.approx(2 / 3)          # 2 wins / 3 valid (error excluded)
    assert summary_wr == pytest.approx(events_wr)     # summary == events
    assert web_wr == pytest.approx(events_wr)         # web == events


# ---- upstream bug regression (pins the finding, does NOT hide it) -------- #
@pytest.mark.skipif(sys.platform != "win32",
                    reason="upstream path-separator bug only manifests on Windows")
@pytest.mark.skipif(not UPSTREAM_AGGREGATE.exists(), reason="upstream aggregate not present")
def test_upstream_aggregate_emits_invalid_toml_on_windows(tmp_path):
    """Locks in the documented upstream bug: on Windows, upstream aggregate.py
    writes backslash paths into registry.toml, which tomllib rejects. The local
    vendored copy (LOCAL_AGGREGATE) fixes this; this test proves the fix is
    necessary and that we are not masking a contract problem in our adapter."""
    _, _ = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    out = _run_py(UPSTREAM_AGGREGATE, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "registry_up.toml"))
    assert out.returncode == 0  # upstream runs, but its output is invalid TOML
    text = (tmp_path / "registry_up.toml").read_text()
    assert "\\" in text, "expected backslash path separators on Windows"
    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(text)

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
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from test_iteration_acceptance import _evaluation_bundle

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
ACTIVE_AGGREGATE = (
    UPSTREAM_AGGREGATE if UPSTREAM_AGGREGATE.exists() else LOCAL_AGGREGATE
)

pytestmark = pytest.mark.skipif(
    not (ACTIVE_AGGREGATE.exists() and UPSTREAM_REPORT_BUILDER.exists()),
    reason="Results aggregate + report_builder required",
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
    out = _run_py(ACTIVE_AGGREGATE, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "registry.toml"))
    assert out.returncode == 0, out.stderr
    reg = tomllib.loads(
        (tmp_path / "registry.toml").read_text(encoding="utf-8")
    )  # parses cleanly
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

    agg = _run_py(ACTIVE_AGGREGATE, "--data-dir", str(tmp_path),
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
def test_upstream_aggregate_emits_utf8_valid_toml_on_windows(tmp_path):
    """The current Results aggregate fixes both native separators and encoding."""
    _, _ = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    out = _run_py(UPSTREAM_AGGREGATE, "--data-dir", str(tmp_path),
                  "--output", str(tmp_path / "registry_up.toml"))
    assert out.returncode == 0
    text = (tmp_path / "registry_up.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert parsed["runs"][0]["path"].startswith("runs/24_miracle/")
    assert "\\" not in parsed["runs"][0]["path"]


def test_iteration_acceptance_summary_is_losslessly_available_to_results_loader(
    tmp_path, monkeypatch
):
    """Results keeps detailed v3 acceptance identities in raw_summary."""

    data = _evaluation_bundle(tmp_path, monkeypatch, kl_complete=False)
    summary = data["acceptance"].to_results_summary()
    data_dir = tmp_path / "results-data"
    relative = Path("runs/24_miracle/fake-agent/fake-iteration")
    run_dir = data_dir / relative
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    entry = {
        "run_id": "fake-iteration",
        "game": "24_miracle",
        "agent": "fake-agent",
        "type": "eval",
        "path": relative.as_posix(),
        "created": "",
        "git_commit": "",
        "summary": {
            "raw_score": summary["raw_score"],
            "evo_score": summary["evo_score"],
            "gain": summary["gain"],
            "evaluation_status": summary["evaluation_status"],
        },
    }
    spec = importlib.util.spec_from_file_location(
        "results_report_builder_iteration_mapping", UPSTREAM_REPORT_BUILDER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        loaded = module.load_run(data_dir, entry)
    finally:
        sys.modules.pop(spec.name, None)

    for field in (
        "information_gain",
        "incomplete_reasons",
        "iteration_protocol_version",
        "research_manifest_sha256",
        "iteration_manifest_sha256",
        "iteration_acceptance_sha256",
        "candidate_evaluation_plan_sha256",
        "evaluation_evidence_sha256",
        "training_match_plan_sha256",
        "training_replay_evidence_sha256",
        "trajectory_kl_evidence_sha256",
        "baseline_policy_sha256",
        "candidate_policy_sha256",
        "experience_skill_sha256",
    ):
        assert loaded.raw_summary[field] == summary[field]
    assert loaded.research["raw_score"] == summary["raw_score"]
    assert loaded.research["evo_score"] == summary["evo_score"]
    assert loaded.research["gain"] == summary["gain"]
    assert loaded.research["evaluation_status"] == "incomplete"

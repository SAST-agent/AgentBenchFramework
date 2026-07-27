"""Phase 8 wiring tests:
1. agent_version (content_hash) recorded into run.toml so a run is traceable
   to the HL codebase version that produced it.
2. QueryHistoryTool reads the real runs/{game}/{agent}/{run_id}/ layout
   (previously it read flat *.json from ./training_logs).
"""
import json
from pathlib import Path

import pytest

from agentbench_frame.tracking.run import Run


def test_run_toml_records_agent_version(tmp_path):
    """A run can carry an agent_version (content_hash) in its config, written
    to run.toml so the run is traceable to the HL version that produced it."""
    run = Run.start(
        game="25_lostspace", agent="cand-v1", run_type="eval",
        data_dir=str(tmp_path),
        config={"agent_version": "abc123def456",
                "spec_id": "bench-v1"},
    )
    run.finish()
    toml = (tmp_path / "runs" / "25_lostspace" / "cand-v1" /
            run.run_id / "run.toml").read_text()
    assert "agent_version" in toml
    assert "abc123def456" in toml
    assert "spec_id" in toml


def test_run_toml_without_agent_version_still_works(tmp_path):
    """Existing callers that don't pass agent_version are unaffected."""
    run = Run.start(game="g", agent="a", run_type="eval", data_dir=str(tmp_path))
    run.finish()
    toml = (tmp_path / "runs" / "g" / "a" / run.run_id / "run.toml").read_text()
    # no agent_version key, but the run still writes fine
    assert "agent_version" not in toml


def _make_run(data_root, agent, run_id, *, matches, win_rate=0.5):
    d = data_root / "runs" / "25_lostspace" / agent / run_id
    d.mkdir(parents=True)
    summary = {
        "run_id": run_id, "agent": agent, "win_rate": win_rate,
        "lostspace": {"aggregate": {"wins": 1, "losses": 1, "errors": 0,
                                    "valid_games": 2, "win_rate": win_rate}},
    }
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "matches.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in matches))


def _match(opp, result):
    return {"opponent": opp, "candidate_result": result,
            "candidate_rank": 1 if result == "win" else 3,
            "pair": 0, "candidate_seat": 0, "turns": 50}


def test_query_history_reads_real_runs_layout(tmp_path):
    """QueryHistoryTool must read runs/{game}/{agent}/{run_id}/, not flat
    ./training_logs/*.json."""
    from agentbench_frame.mcp.history_tool import QueryHistoryTool

    _make_run(tmp_path, "cand-v1", "r1",
              matches=[_match("alpha", "win"), _match("beta", "loss")])
    _make_run(tmp_path, "cand-v1", "r2",
              matches=[_match("alpha", "loss")])

    tool = QueryHistoryTool(data_dir=str(tmp_path))
    history = tool._load_history()
    # one entry per match, across all runs
    assert len(history) == 3
    agents = {h["agent"] for h in history}
    assert agents == {"cand-v1"}
    opps = {h["opponent"] for h in history}
    assert opps == {"alpha", "beta"}


def test_query_history_head_to_head(tmp_path):
    from agentbench_frame.mcp.history_tool import QueryHistoryTool

    _make_run(tmp_path, "cand-v1", "r1",
              matches=[_match("alpha", "win"), _match("alpha", "win"),
                       _match("alpha", "loss"), _match("beta", "loss")])
    tool = QueryHistoryTool(data_dir=str(tmp_path))
    result = tool.call(query_type="head_to_head", agent_name="cand-v1",
                       opponent_name="alpha")
    assert result["success"] is True
    assert result["wins"] == 2
    assert result["losses"] == 1


def test_query_history_agent_stats(tmp_path):
    from agentbench_frame.mcp.history_tool import QueryHistoryTool

    _make_run(tmp_path, "cand-v1", "r1",
              matches=[_match("alpha", "win"), _match("beta", "loss")])
    tool = QueryHistoryTool(data_dir=str(tmp_path))
    result = tool.call(query_type="agent_stats", agent_name="cand-v1")
    assert result["success"] is True
    assert result["total_games"] == 2
    assert result["wins"] == 1
    assert result["losses"] == 1


def test_query_history_empty_when_no_runs(tmp_path):
    from agentbench_frame.mcp.history_tool import QueryHistoryTool
    tool = QueryHistoryTool(data_dir=str(tmp_path / "nope"))
    assert tool._load_history() == []

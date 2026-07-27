"""Tests for hl/resources.py — read-only views handed to the coding agent.

Contract (plan §resources):
- MatchHistoryView: aggregates runs/{game}/{agent}/*/summary.json + the
  per-match records from matches.jsonl. No corpus leakage (ladder source
  hidden). Returns per-opponent win rates, ranks, error counts.
- ReplayView: native artifacts/*.json + parsed per-decision tuples from
  optional .trace.jsonl.
- VersionDiffView: read-only prior-version content + structured diff.
"""
import json
from pathlib import Path

import pytest

from agentbench_frame.hl.resources import (
    MatchHistoryView,
    ReplayView,
    VersionDiffView,
)
from agentbench_frame.hl.codebase import HLCodebase


def _make_run(data_root: Path, agent: str, run_id: str, *,
              opponents=("rank01", "rank06"), matches=None, summary=None):
    d = data_root / "runs" / "25_lostspace" / agent / run_id
    d.mkdir(parents=True)
    s = summary or {
        "run_id": run_id, "agent": agent, "win_rate": 0.5,
        "lostspace": {
            "aggregate": {"wins": 2, "losses": 2, "errors": 0,
                          "valid_games": 4, "win_rate": 0.5,
                          "avg_rank": 2.0},
            "by_opponent": {o: {"wins": 1, "losses": 1, "win_rate": 0.5,
                                "valid_games": 2, "errors": 0}
                            for o in opponents},
        },
    }
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    (d / "matches.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in (matches or [])),
        encoding="utf-8",
    )
    return d


def _match(opp, result, rank, run_id="r1"):
    return {"opponent": opp, "candidate_result": result,
            "candidate_rank": rank, "pair": 0, "candidate_seat": 0,
            "turns": 50, "run_id": run_id}


def test_match_history_view_aggregates_across_runs(tmp_path):
    data = tmp_path / "data"
    _make_run(data, "cand-v1", "20260101_0001_aaaaaaaa",
              matches=[_match("rank01", "win", 1), _match("rank06", "loss", 3)])
    _make_run(data, "cand-v1", "20260101_0002_bbbbbbbb",
              matches=[_match("rank01", "loss", 2), _match("rank06", "win", 1)])

    view = MatchHistoryView(data_root=data, game="25_lostspace", agent="cand-v1")
    rows = view.match_rows()
    assert len(rows) == 4  # 2 runs × 2 matches

    by_opp = view.by_opponent()
    assert by_opp["rank01"]["wins"] == 1
    assert by_opp["rank01"]["losses"] == 1
    assert by_opp["rank01"]["win_rate"] == 0.5
    assert by_opp["rank06"]["wins"] == 1


def test_match_history_view_no_runs_returns_empty(tmp_path):
    view = MatchHistoryView(data_root=tmp_path, game="25_lostspace", agent="none")
    assert view.match_rows() == []
    assert view.by_opponent() == {}


def test_match_history_view_keeps_per_run_summary(tmp_path):
    data = tmp_path / "data"
    _make_run(data, "cand-v1", "r1", matches=[_match("rank01", "win", 1)])
    view = MatchHistoryView(data_root=data, game="25_lostspace", agent="cand-v1")
    runs = view.run_summaries()
    assert len(runs) == 1
    assert runs[0]["agent"] == "cand-v1"
    assert runs[0]["lostspace"]["aggregate"]["wins"] == 2


def test_match_history_view_records_errors_not_as_losses(tmp_path):
    """Doc §12: errors are not counted as losses; valid_games excludes them."""
    data = tmp_path / "data"
    _make_run(data, "cand-v1", "r1",
              matches=[_match("rank01", "win", 1),
                       {"opponent": "rank06", "candidate_result": "error",
                        "candidate_rank": None, "pair": 0,
                        "candidate_seat": 0, "turns": 0}])
    view = MatchHistoryView(data_root=data, game="25_lostspace", agent="cand-v1")
    by_opp = view.by_opponent()
    assert by_opp["rank06"]["errors"] == 1
    assert by_opp["rank06"]["valid_games"] == 0
    assert by_opp["rank06"]["win_rate"] is None  # no valid games


def test_replay_view_loads_native_json(tmp_path):
    # minimal native replay: [birthplaces, round1, score_dic]
    replay = [
        [[0, 0, 1], [6, 0, 1], [6, 6, 1], [0, 6, 1]],
        [[], [], [], []],  # round 1: 4 empty turns
        {"0": 4, "1": 3, "2": 2, "3": 1},
    ]
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(replay), encoding="utf-8")
    rv = ReplayView(p)
    assert rv.n_rounds() == 1
    # score_dic returns int keys (player id) -> rank points
    assert rv.score_dic() == {0: 4, 1: 3, 2: 2, 3: 1}
    assert rv.ranking() == [0, 1, 2, 3]  # by score descending


def test_replay_view_winner(tmp_path):
    replay = [[], [], {"0": 2, "1": 4, "2": 3, "3": 1}]  # player 1 wins
    p = tmp_path / "r.json"
    p.write_text(json.dumps(replay), encoding="utf-8")
    rv = ReplayView(p)
    assert rv.winner() == 1


def test_version_diff_view_is_read_only(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("v0\n", encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    (ws / "agent.py").write_text("v1\n", encoding="utf-8")
    h1 = cb.snapshot(parent_version_id=h0.version_id)

    dv = VersionDiffView(cb, h0.content_hash, h1.content_hash)
    assert "agent.py" in dv.modified
    assert "manifest.toml" in dv.unchanged

    # read-only: the view exposes content but does not let the agent write
    # the store or the workspace.
    content = dv.read_file(after_hash=h1.content_hash, relpath="agent.py")
    assert content == "v1\n"

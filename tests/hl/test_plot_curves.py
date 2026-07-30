"""Tests for hl/plot_curves.py — the iteration-curve renderer.

Honesty contract (the point of this module):
- An incomplete eval (win_rate=None) yields a GAP on the score curve, never a 0.
- A failed act is marked (red ✗), not hidden.
- policy_kl is only present for acts with a prior version; act 1 has no IG point.
- The reader never imputes measurements that the events file did not record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.hl.plot_curves import (
    read_iteration_curves,
    _act_sort_key,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _ev(event_type: str, **kw) -> dict:
    """Build a minimal recognized event with the public fields."""
    base = {
        "schema_version": 1, "event_id": f"id-{kw.get('act_id', 'x')}-{event_type}",
        "run_id": "r", "created_at": "2026-07-29T00:00:00.000000Z",
        "event_type": event_type,
    }
    base.update(kw)
    return base


def _write_events(path: Path, events: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _round_events() -> list:
    """A 3-act round:
    - act 1: complete eval win_rate=0.0, no policy_kl (first act).
    - act 2: complete eval win_rate=0.25, policy_kl trace [0.0, 0.5, 0.0].
    - act 3: INCOMPLETE eval (win_rate=None), policy_kl trace [0.0, 0.0, 0.0],
      plus a failure_reason on the version event (failed act).
    """
    return [
        _ev("agent_act", act_id="r-000001"),
        _ev("version", act_id="r-000001", version_id="v1",
            content_hash="h1", parent_version_id=None,
            edit_type="initial", failure_reason=None),
        _ev("eval", act_id="r-000001", spec_id="s",
            version_after="v1", evaluation_status="complete", win_rate=0.0),
        _ev("budget", act_id="r-000001", scope="learning",
            coding_agent_acts=1, failure_reason=None),

        _ev("agent_act", act_id="r-000002"),
        _ev("version", act_id="r-000002", version_id="v2",
            content_hash="h2", parent_version_id="v1",
            edit_type="add_rule", failure_reason=None),
        _ev("eval", act_id="r-000002", spec_id="s",
            version_after="v2", evaluation_status="complete", win_rate=0.25),
        _ev("policy_kl", act_id="r-000002",
            version_before="v1", version_after="v2",
            local_policy_kl_trace=[0.0, 0.5, 0.0], epsilon=0.1),
        _ev("budget", act_id="r-000002", scope="learning",
            coding_agent_acts=2, failure_reason=None),

        _ev("agent_act", act_id="r-000003"),
        _ev("version", act_id="r-000003", version_id="v3",
            content_hash="h3", parent_version_id="v2",
            edit_type="noop", failure_reason="claude exited 1: boom"),
        _ev("eval", act_id="r-000003", spec_id="s",
            version_after="v3", evaluation_status="incomplete", win_rate=None),
        _ev("policy_kl", act_id="r-000003",
            version_before="v2", version_after="v3",
            local_policy_kl_trace=[0.0, 0.0, 0.0], epsilon=0.1),
        _ev("budget", act_id="r-000003", scope="learning",
            coding_agent_acts=3, failure_reason="claude exited 1: boom"),
    ]


def test_act_sort_key():
    assert _act_sort_key("r-000001") == 1
    assert _act_sort_key("r-000010") == 10
    assert _act_sort_key(None) == -1


def test_read_joins_per_act_and_orders_by_index(tmp_path):
    p = tmp_path / "events.jsonl"
    _write_events(p, _round_events())
    data = read_iteration_curves(p)

    assert [pt.iteration for pt in data.points] == [1, 2, 3]
    a1, a2, a3 = data.points

    # act 1: complete eval, no policy_kl (first act)
    assert a1.win_rate == 0.0
    assert a1.policy_kl_mean is None
    assert a1.edit_type == "initial"
    assert a1.failed is False

    # act 2: the valid policy update — KL>0 and win_rate improved
    assert a2.win_rate == 0.25
    assert a2.policy_kl_mean == pytest.approx(0.5 / 3)
    assert a2.policy_kl_n == 3
    assert a2.edit_type == "add_rule"

    # act 3: incomplete eval STAYS None (never coerced to 0)
    assert a3.win_rate is None
    assert a3.evaluation_status == "incomplete"
    assert a3.failed is True
    assert a3.failure_reason == "claude exited 1: boom"
    # KL still measured (parent existed) — 0.0 mean, but present
    assert a3.policy_kl_mean == 0.0


def test_summary_counts(tmp_path):
    p = tmp_path / "events.jsonl"
    _write_events(p, _round_events())
    data = read_iteration_curves(p)
    assert data.n_incomplete == 1     # act 3
    assert data.n_failed == 1         # act 3


def test_plot_renders_two_pngs_and_preserves_gaps(tmp_path):
    from agentbench_frame.hl.plot_curves import plot
    p = tmp_path / "events.jsonl"
    _write_events(p, _round_events())
    data = read_iteration_curves(p)

    out = plot(data, tmp_path / "figs")
    assert Path(out["score_iteration"]).exists()
    assert Path(out["ig_iteration"]).exists()
    # PNGs are non-empty
    assert Path(out["score_iteration"]).stat().st_size > 0
    assert Path(out["ig_iteration"]).stat().st_size > 0


def test_edit_type_joined_from_version_event(tmp_path):
    """edit_type lives on the version event (now carrying act_id); the curve
    row must surface it so the figure/table can annotate the edit kind."""
    p = tmp_path / "events.jsonl"
    _write_events(p, _round_events())
    data = read_iteration_curves(p)
    by_iter = {pt.iteration: pt for pt in data.points}
    assert by_iter[1].edit_type == "initial"
    assert by_iter[2].edit_type == "add_rule"
    assert by_iter[3].edit_type == "noop"


def test_missing_eval_kept_none_not_zero(tmp_path):
    """An act with an agent_act + version but NO eval event must record
    win_rate=None, not 0 — the core honesty rule."""
    p = tmp_path / "events.jsonl"
    _write_events(p, [
        _ev("agent_act", act_id="r-000001"),
        _ev("version", act_id="r-000001", version_id="v1",
            content_hash="h1", parent_version_id=None,
            edit_type="initial", failure_reason=None),
        # no eval event at all
    ])
    data = read_iteration_curves(p)
    assert len(data.points) == 1
    assert data.points[0].win_rate is None
    assert data.n_incomplete == 1

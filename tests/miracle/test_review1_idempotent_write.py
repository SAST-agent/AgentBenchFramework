"""review #1 §8: ``write_run_compatible_output`` must be idempotent.

Repeated calls (e.g. matrix resume, re-runs in CI) must not append duplicate
events to ``events.jsonl``; each rebuild yields the same per-game record count
and the same envelope (event_id disallowed to vary across calls — we don't
re-seed; each call produces a fresh set of UUID event_ids but the gameId-
count is what determines reproducibility). For the test we verify:

  - repeated calls do not increase the number of ``game`` events;
  - the run directory stays at one entry (no second run_id);
  - win_rate and per-game lines count are deterministic.

Also covers review #1 §5: framework envelope is present on every emitted
event (``event``, ``event_type``, ``schema_version``, ``event_id``,
``run_id``, ``created_at``, ``timestamp``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_runner(tmp_path):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle.matrix import mark_done, write_progress_atomic, append_event_atomic
    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=lambda **k: None,
        run_id="FIXED_RID_for_idempotency",
    )
    r.prepare_session()
    r.record_manifest(opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
                      build_hashes={i: "b" + str(i) for i in range(1, 17)},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    # write three sample session events
    for gid, valid, norm in [("m_rank01_camp0", True, "win"),
                              ("m_rank01_camp1", True, "loss"),
                              ("m_rank02_camp0", False, "error")]:
        mark_done(r.progress, gid, {"valid": valid, "normalized_result": norm,
                                    "rank": int(gid.split("_")[1][5:]), "camp": int(gid[-1])})
        append_event_atomic(r.events_path, {
            "event": "game", "game_id": gid,
            "valid": valid, "normalized_result": norm,
            "rank": int(gid.split("_")[1][5:]), "camp": int(gid[-1]),
            "judge_exit": 0, "ai0_exit": 0, "ai1_exit": 1,
            "replay_sha256": "x" * 64,
            "process_cleanup": [{"role": "vendor"}, {"role": "judge"}],
        })
    write_progress_atomic(r.progress_path, r.progress)
    return r


def test_write_run_compatible_output_is_idempotent(tmp_path: Path):
    r = _make_runner(tmp_path)
    run_dir = r.write_run_compatible_output()
    ev = Path(run_dir) / "events.jsonl"
    assert ev.exists()
    lines1 = [l for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len([l for l in lines1 if json.loads(l).get("event_type") == "game"]) == 3, \
        "first call should produce exactly 3 game events"

    # second call on the SAME runner+session — must NOT duplicate
    run_dir2 = r.write_run_compatible_output()
    assert Path(run_dir) == Path(run_dir2), "run_dir path must be reused, not duplicated"

    # only ONE run directory under data_dir/runs/24_miracle/...
    data_runs = Path(run_dir).parent
    sib = list(data_runs.iterdir())
    assert len([p for p in sib if p.is_dir()]) == 1, \
        f"second run_id created: {[p.name for p in sib]}"

    # events count must remain exactly 3 (no duplicate append)
    lines2 = [l for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines2) == 3, \
        f"events doubled after re-call: {len(lines2)} (expected 3)"

    # all events carry framework envelope after re-call
    for l in lines2:
        e = json.loads(l)
        assert e.get("event") == "game" and e.get("event_type") == "game"
        assert e.get("schema_version") == "1.0"
        assert e.get("event_id", "").startswith("evt_")
        assert e.get("run_id") == "FIXED_RID_for_idempotency"
        assert "created_at" in e and "timestamp" in e

    # a THIRD re-call must still be exactly 3 (idempotent at N invocations)
    r.write_run_compatible_output()
    lines3 = [l for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines3) == 3, f"3rd call doubled: {len(lines3)}"


def test_envelope_present_on_every_game_event(tmp_path: Path):
    r = _make_runner(tmp_path)
    run_dir = r.write_run_compatible_output()
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert events
    for e in events:
        for k in ("event", "event_type", "schema_version", "event_id", "run_id", "created_at", "timestamp"):
            assert k in e, f"event missing envelope key {k}: {sorted(e)}"
        # Miracle first-hand fields preserved verbatim
        for k in ("game_id", "judge_exit", "ai0_exit", "ai1_exit", "replay_sha256", "process_cleanup"):
            assert k in e, f"first-hand Miracle field {k} missing after envelope: {sorted(e)}"


def test_summary_json_win_rate_matches_aggregate(tmp_path: Path):
    r = _make_runner(tmp_path)
    run_dir = r.write_run_compatible_output()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # 2 valid games / 1 win / 1 loss → win_rate 0.5 (not 0.0 from framework default)
    assert summary["win_rate"] == 0.5
    assert summary["total_episodes"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    # matrix aggregate preserved
    assert summary["matrix_aggregate"]["valid_games"] == 2
    assert summary["matrix_aggregate"]["wins"] == 1


def test_run_toml_round_trips_after_review1_lifecycle(tmp_path: Path):
    import tomllib
    r = _make_runner(tmp_path)
    run_dir = r.write_run_compatible_output()
    meta = tomllib.loads((run_dir / "run.toml").read_text(encoding="utf-8"))
    assert meta["run"]["run_id"] == "FIXED_RID_for_idempotency"
    assert meta["run"]["game"] == "24_miracle"
    assert meta["run"]["type"] == "eval"
    assert meta["config"]["matrix"] == "plan_a_32"
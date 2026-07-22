"""match_runner contract tests (阶段4b-4 spec, 20 required behaviours).

RED LIGHT FIRST: the implementation match_runner.py does not exist yet, so the
whole module fails to import; the failure is saved to
docs/games/evidence/match_runner_redlight.txt, THEN the implementation is
written to turn these green.

Pure-logic behaviours (1, 5-16, 18) drive synthesized result-json / trace /
Replay files through match_runner's reader + cross-validation + classifier.
Integration behaviours (2, 3, 4, 17, 20) drive the harmless _fake_vendor.py
through run_match_attempt. Behaviour 19 checks cwd-independent import.

    py -3.13 -m pytest tests/miracle/test_match_runner.py -v
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from agentbench_frame.games.miracle import match_runner
from agentbench_frame.games.miracle.match_runner import (
    Classification,
    MatchAttempt,
    ReplayInfo,
    TraceStats,
    classify,
    cross_validate,
    load_result_json,
    read_replay_info,
    run_match_attempt,
    stream_trace,
)

FAKE_VENDOR = Path(__file__).parent / "_fake_vendor.py"
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


# ---- synthesized-data builders ------------------------------------------- #
def mk_rj(**kw):
    base = {
        "schema_version": 1, "tag": "g",
        "end_info_received": True, "end_info": {"0": 5, "1": 2},
        "scores": {"0": 5, "1": 2}, "raw_winner": 0,
        "score_tie": False, "judge_tiebreak_applied": False,
        "timeout": {"ai0": False, "ai1": False},
        "ai_error": {"ai0": False, "ai1": False},
        "cleanup_all_succeeded": True, "exception": None,
        "run_match_returncode": 0,
        "judge": {"cleanup_succeeded": True, "natural_exit": True, "termination_requested": False, "final_returncode": 0},
        "ai0": {"cleanup_succeeded": True, "natural_exit": True, "termination_requested": False, "final_returncode": 0},
        "ai1": {"cleanup_succeeded": True, "natural_exit": True, "termination_requested": False, "final_returncode": 0},
    }
    base.update(kw)
    return base


def write_json(p: Path, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def write_trace(p: Path, events):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def write_replay(p: Path, map_type=0, day_time=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(struct.pack(">7i", 0, 0, 0, map_type, day_time, 0, 0))


def _classify_for(rj, ts, ri, *, wrapper_timeout=False, discrepancies=None,
                  evaluated_agent_camp=0, vendor_returncode=0):
    status, data = ("ok", rj) if rj is not None else ("missing", None)
    return classify(rj_status=status, rj=data, ts=ts, ri=ri,
                    discrepancies=discrepancies or [], vendor_returncode=vendor_returncode,
                    wrapper_timeout=wrapper_timeout, evaluated_agent_camp=evaluated_agent_camp,
                    evaluated_agent="ifelse", opponent="rank04")


# ========================================================================== #
# 1. vendor runner returns a complete result-json (integration, fake vendor) #
# ========================================================================== #
def test_01_normal_complete_result_json(tmp_path):
    att = run_match_attempt(
        game_id="g1", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="sampleA", p1_name="sampleB", judge_dir=tmp_path / "judge",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        evaluated_agent="sampleA", opponent="sampleB", evaluated_agent_camp=0,
        extra_vendor_args=["--fake-mode", "normal"],
    )
    assert att.result_json_status == "ok"
    assert att.valid is True
    assert att.normalized_result == "win"     # raw_winner 0, evaluated camp 0
    assert att.raw_winner == 0
    assert att.steps == 3                       # 3 ai_operation events
    assert att.realized_randomization == {"map_type": 0, "day_time": 1}


# ===== 2. result-json missing ===== #
def test_02_result_json_missing(tmp_path):
    att = run_match_attempt(
        game_id="g2", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="a", p1_name="b", judge_dir=tmp_path / "j",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        extra_vendor_args=["--fake-mode", "no_result"],
    )
    assert att.result_json_status == "missing"
    assert att.valid is False
    assert att.error_type == "result_json_missing"


# ===== 3. result-json corrupt / truncated ===== #
def test_03_result_json_corrupt(tmp_path):
    att = run_match_attempt(
        game_id="g3", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="a", p1_name="b", judge_dir=tmp_path / "j",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        extra_vendor_args=["--fake-mode", "corrupt"],
    )
    assert att.result_json_status == "corrupt"
    assert att.valid is False
    assert att.error_type == "result_json_corrupt"


# ===== 4. vendor runner nonzero exit ===== #
def test_04_vendor_nonzero_exit(tmp_path):
    att = run_match_attempt(
        game_id="g4", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="a", p1_name="b", judge_dir=tmp_path / "j",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        extra_vendor_args=["--fake-mode", "nonzero"],
    )
    assert att.vendor_returncode != 0
    assert att.valid is False
    assert att.error_type == "result_json_missing"   # no result-json produced


# ===== 5. result-json & trace end_info consistent ===== #
def test_05_end_info_consistent(tmp_path):
    rj = mk_rj()
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [
        {"kind": "ai_operation", "player": 0},
        {"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})},
    ])
    ts = stream_trace(trace)
    ri = read_replay_info(tmp_path / "absent.replay")
    assert cross_validate(rj, ts, ri) == []


# ===== 6. result-json & trace scores conflict ===== #
def test_06_scores_conflict(tmp_path):
    rj = mk_rj(end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2}, raw_winner=0)
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [{"kind": "match_end", "end_info": json.dumps({"0": 2, "1": 5})}])
    ts = stream_trace(trace)
    ri = read_replay_info(tmp_path / "absent.replay")
    discs = cross_validate(rj, ts, ri)
    assert any("score" in d.lower() or "end_info" in d.lower() for d in discs)
    c = _classify_for(rj, ts, ri, discrepancies=discs)
    assert c.error_type == "evidence_mismatch" and not c.valid


# ===== 7. derived raw_winner conflicts with score rule ===== #
def test_07_raw_winner_conflicts(tmp_path):
    rj = mk_rj(end_info={"0": 2, "1": 5}, scores={"0": 2, "1": 5}, raw_winner=0)  # 0 says p0 won but s1>s0
    ri = read_replay_info(tmp_path / "absent.replay")
    ts = TraceStats()  # empty
    discs = cross_validate(rj, ts, ri)
    assert any("raw_winner" in d.lower() or "winner" in d.lower() for d in discs)
    c = _classify_for(rj, ts, ri, discrepancies=discs)
    assert c.error_type == "evidence_mismatch" and not c.valid


# ===== 8. trace has ai_error ===== #
def test_08_trace_ai_error(tmp_path):
    rj = mk_rj()
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [
        {"kind": "ai_operation", "player": 0},
        {"kind": "ai_error", "player": 1, "state": 3},
        {"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})},
    ])
    ts = stream_trace(trace)
    assert 1 in ts.ai_error_players
    ri = read_replay_info(tmp_path / "absent.replay")
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "ai_crash" and c.ai_crash_player == 1 and not c.valid


# ===== 9. trace has ai_timeout ===== #
def test_09_trace_ai_timeout(tmp_path):
    rj = mk_rj()
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [
        {"kind": "ai_timeout", "player": 0, "state": 2},
        {"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})},
    ])
    ts = stream_trace(trace)
    assert 0 in ts.ai_timeout_players
    ri = read_replay_info(tmp_path / "absent.replay")
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "ai_timeout" and c.ai_timeout_player == 0 and not c.valid


# ===== 10. AI natural exit before end_info ===== #
def test_10_ai_natural_exit_before_end_info(tmp_path):
    rj = mk_rj(end_info_received=False, end_info=None, scores=None, raw_winner=None,
               ai0={"cleanup_succeeded": True, "natural_exit": True,
                    "termination_requested": False, "final_returncode": 1})
    ri = read_replay_info(tmp_path / "absent.replay")
    ts = TraceStats()
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "ai_crash" and c.ai_crash_player == 0 and not c.valid


# ===== 11. AI cleanup-killed after end_info, nonzero returncode -> NOT crash #
def test_11_ai_cleanup_nonzero_after_end_info_is_normal(tmp_path):
    rj = mk_rj(ai0={"cleanup_succeeded": True, "natural_exit": False,
                    "termination_requested": True, "final_returncode": 1},
               ai1={"cleanup_succeeded": True, "natural_exit": False,
                    "termination_requested": True, "final_returncode": 1})
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [{"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})}])
    ts = stream_trace(trace)
    replay = tmp_path / "r.replay"; write_replay(replay)
    ri = read_replay_info(replay)
    c = _classify_for(rj, ts, ri)
    assert c.valid is True
    assert c.normal_cleanup_nonzero is True
    assert c.error_type is None
    assert c.normalized_result == "win"


# ===== 12. Judge exits before end_info ===== #
def test_12_judge_exit_before_end_info(tmp_path):
    rj = mk_rj(end_info_received=False, end_info=None, scores=None, raw_winner=None,
               judge={"cleanup_succeeded": True, "natural_exit": True,
                      "termination_requested": False, "final_returncode": 1},
               ai0={"cleanup_succeeded": True, "natural_exit": False,
                    "termination_requested": False, "final_returncode": None},
               ai1={"cleanup_succeeded": True, "natural_exit": False,
                    "termination_requested": False, "final_returncode": None})
    ri = read_replay_info(tmp_path / "absent.replay")
    ts = TraceStats()
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "judge_crash" and c.judge_crash and not c.valid


# ===== 13. Replay missing ===== #
def test_13_replay_missing(tmp_path):
    rj = mk_rj()
    ts = TraceStats()
    ri = read_replay_info(tmp_path / "absent.replay")
    assert ri.exists is False
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "replay_missing" and not c.valid


# ===== 14. Replay corrupt / header too short ===== #
def test_14_replay_short_header(tmp_path):
    rj = mk_rj()
    ts = TraceStats()
    replay = tmp_path / "r.replay"
    replay.write_bytes(b"\x00" * 10)  # too short for 7 int32
    ri = read_replay_info(replay)
    assert ri.exists is True and ri.header_valid is False
    c = _classify_for(rj, ts, ri)
    assert c.error_type == "replay_corrupt" and not c.valid


# ===== 15. Replay map_type/day_time parseable ===== #
def test_15_replay_header_parseable(tmp_path):
    replay = tmp_path / "r.replay"
    write_replay(replay, map_type=1, day_time=0)
    ri = read_replay_info(replay)
    assert ri.header_valid is True
    assert ri.map_type == 1 and ri.day_time == 0
    assert ri.sha256 and len(ri.sha256) == 64


# ===== 16. ai_operation streaming count ===== #
def test_16_ai_operation_streaming_count(tmp_path):
    trace = tmp_path / "t.jsonl"
    write_trace(trace, [
        {"kind": "ai_operation", "player": 0},
        {"kind": "ai_operation", "player": 1},
        {"kind": "ai_operation", "player": 0},
        {"kind": "ai_operation", "player": 1},
        {"kind": "ai_operation", "player": 0},
        {"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})},
    ])
    ts = stream_trace(trace)
    assert ts.n_ai_operation == 5
    assert ts.end_info_seen is True


# ===== 17. wrapper timeout cleans vendor + child tree ===== #
def test_17_wrapper_timeout_kills_vendor_tree(tmp_path):
    import os as _os
    att = run_match_attempt(
        game_id="g17", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="a", p1_name="b", judge_dir=tmp_path / "j",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        wrapper_timeout_s=2.0,
        extra_vendor_args=["--fake-mode", "hang"],
    )
    assert att.wrapper_timeout is True
    assert att.valid is False
    assert att.error_type == "wrapper_timeout"
    # the fake vendor printed "CHILD <pid>" to its stdout (captured to a file)
    stdout_file = Path(att.evidence_paths["stdout"])
    child_pid = None
    for line in stdout_file.read_text(errors="replace").splitlines():
        if line.startswith("CHILD "):
            child_pid = int(line.split()[1])
    assert child_pid is not None, "fake vendor did not report a child PID"
    # grace for the OS to reap
    deadline = time.time() + 5
    while time.time() < deadline and psutil.pid_exists(child_pid):
        time.sleep(0.1)
    assert not psutil.pid_exists(child_pid), "orphan child survived wrapper-timeout cleanup"


# ===== 18. duplicate game_id / output-dir collision ===== #
def test_18_duplicate_game_id_collision(tmp_path):
    common = dict(
        p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1", p0_name="a", p1_name="b",
        judge_dir=tmp_path / "j", work_dir=tmp_path / "work",
        vendor_script=FAKE_VENDOR, framework_src=SRC,
        extra_vendor_args=["--fake-mode", "normal"],
    )
    first = run_match_attempt(game_id="dup", evaluated_agent_camp=0, **common)
    assert first.valid is True
    first_json = Path(first.evidence_paths["result_json"])
    first_content = first_json.read_text()
    second = run_match_attempt(game_id="dup", evaluated_agent_camp=0, **common)
    assert second.collision_detected is True
    # the first attempt's result-json must NOT have been clobbered
    assert first_json.read_text() == first_content


# ===== 19. import works from any cwd ===== #
def test_19_import_from_any_cwd(tmp_path):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from agentbench_frame.games.miracle import match_runner\n"
        "print('IMPORT_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "IMPORT_OK" in r.stdout


# ===== 20. large stdout/stderr does not deadlock ===== #
def test_20_large_stdout_stderr_no_deadlock(tmp_path):
    t0 = time.time()
    att = run_match_attempt(
        game_id="g20", p0_dir=tmp_path / "p0", p1_dir=tmp_path / "p1",
        p0_name="a", p1_name="b", judge_dir=tmp_path / "j",
        work_dir=tmp_path / "work", vendor_script=FAKE_VENDOR, framework_src=SRC,
        extra_vendor_args=["--fake-mode", "bigio"],
    )
    elapsed = time.time() - t0
    assert elapsed < 30, f"wrapper deadlocked or hung: {elapsed:.1f}s"
    stdout_size = Path(att.evidence_paths["stdout"]).stat().st_size
    stderr_size = Path(att.evidence_paths["stderr"]).stat().st_size
    assert stdout_size >= 3 * 1024 * 1024 - 1024
    assert stderr_size >= 3 * 1024 * 1024 - 1024

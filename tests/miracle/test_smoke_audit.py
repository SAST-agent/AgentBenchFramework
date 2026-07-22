"""Tests for the smoke evidence-safety helpers (阶段4b smoke driver fixes)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from agentbench_frame.games.miracle.smoke_audit import (
    ManagedProc,
    build_manifest,
    check_residual_procs,
    ensure_fresh_session,
    group1_strict_clean,
    load_managed_procs_from_result_json,
    make_session_id,
    session_exists,
    should_run_group2,
    write_manifest_atomic,
)


# ---- 1. existing session is rejected, never deleted ---- #
def test_existing_session_rejected_and_not_deleted(tmp_path):
    sid = make_session_id()
    sd = ensure_fresh_session(tmp_path, sid)
    (sd / "marker").write_text("prior evidence")
    # a second ensure with the SAME id must refuse
    with pytest.raises(FileExistsError):
        ensure_fresh_session(tmp_path, sid)
    # and the prior evidence must still be there
    assert (sd / "marker").read_text() == "prior evidence"


def test_session_ids_are_unique():
    ids = {make_session_id() for _ in range(50)}
    assert len(ids) == 50


# ---- a fake MatchAttempt-shaped object for gate tests ---- #
class FakeAtt:
    def __init__(self, *, game_id="g1_00_camp0", valid=True, normalized_result="win",
                 error_type=None, wrapper_timeout=False, result_json_status="ok",
                 discrepancies=None, realized_randomization=None, raw_winner=0,
                 reason="", evidence_paths=None):
        self.game_id = game_id
        self.valid = valid
        self.normalized_result = normalized_result
        self.error_type = error_type
        self.wrapper_timeout = wrapper_timeout
        self.result_json_status = result_json_status
        self.discrepancies = discrepancies or []
        self.realized_randomization = realized_randomization or {"map_type": 0, "day_time": 1}
        self.raw_winner = raw_winner
        self.reason = reason
        self.evidence_paths = evidence_paths or {"result_json": ""}


def _good_summary():
    return {"attempted_games": 2, "valid_games": 2, "invalid_games": 0,
            "evaluation_status": "COMPLETE"}


def _two_good_attempts_with_procs(tmp_path):
    # give each attempt a result-json with judge/ai0/ai1 identity for the residual check
    rj = tmp_path / "rj.json"
    rj.write_text(json.dumps({
        "judge": {"pid": 999999, "started_at": 1.0, "role": "judge"},
        "ai0": {"pid": 999998, "started_at": 1.0, "role": "ai0"},
        "ai1": {"pid": 999997, "started_at": 1.0, "role": "ai1"},
    }))
    a0 = FakeAtt(game_id="g1_00_camp0", raw_winner=0,
                 evidence_paths={"result_json": str(rj)})
    a1 = FakeAtt(game_id="g1_01_camp1", raw_winner=1,
                 evidence_paths={"result_json": str(rj)})
    return [a0, a1]


# ---- 2. invalid attempt blocks Group 2 ---- #
def test_invalid_attempt_blocks_group2(tmp_path):
    attempts = _two_good_attempts_with_procs(tmp_path)
    attempts[0].valid = False
    attempts[0].error_type = "ai_crash"
    ok, reasons = group1_strict_clean(attempts, _good_summary())
    assert ok is False
    assert should_run_group2(ok) is False


# ---- 3. attempt count < or > 2 blocks ---- #
def test_attempt_count_not_two_blocks(tmp_path):
    one = _two_good_attempts_with_procs(tmp_path)[:1]
    ok, reasons = group1_strict_clean(one, _good_summary())
    assert ok is False and any("attempt_count" in r for r in reasons)
    three = _two_good_attempts_with_procs(tmp_path) + [FakeAtt()]
    ok3, _ = group1_strict_clean(three, _good_summary())
    assert ok3 is False


# ---- 4. summary count mismatch blocks ---- #
def test_summary_count_mismatch_blocks(tmp_path):
    attempts = _two_good_attempts_with_procs(tmp_path)
    bad = {"attempted_games": 2, "valid_games": 1, "invalid_games": 1,
           "evaluation_status": "COMPLETE"}
    ok, reasons = group1_strict_clean(attempts, bad)
    assert ok is False
    assert any("valid_games" in r for r in reasons)


# ---- 5. PID alive + same create_time -> residual ---- #
def test_residual_detected_for_live_pid_same_createtime():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        ct = psutil.Process(p.pid).create_time()
        res = check_residual_procs([ManagedProc(pid=p.pid, started_at=ct, role="judge")])
        assert res["residual"] and res["residual"][0].pid == p.pid
        assert res["clean"] == [] and res["reused"] == []
    finally:
        p.terminate(); p.wait(timeout=5)


# ---- 6. PID reuse (wrong create_time) -> NOT residual, NOT killed ---- #
def test_pid_reuse_not_flagged_as_residual():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        res = check_residual_procs([ManagedProc(pid=p.pid, started_at=1.0, role="ai0")])
        assert res["residual"] == []            # not flagged
        assert res["reused"] and res["reused"][0].pid == p.pid
        assert psutil.pid_exists(p.pid)         # still alive — we did NOT kill it
    finally:
        p.terminate(); p.wait(timeout=5)


# ---- 7. missing process identity -> cannot pass via empty set ---- #
def test_missing_proc_identity_blocks_gate(tmp_path):
    a0 = FakeAtt(game_id="g1_00_camp0", evidence_paths={"result_json": ""})  # no identity
    a1 = FakeAtt(game_id="g1_01_camp1", evidence_paths={"result_json": ""})
    ok, reasons = group1_strict_clean([a0, a1], _good_summary())
    assert ok is False
    assert any("no managed-proc identity" in r for r in reasons)


def test_load_managed_procs_handles_missing_and_identity():
    assert load_managed_procs_from_result_json("/no/such/file.json") == []
    import tempfile
    f = Path(tempfile.mkdtemp()) / "r.json"
    f.write_text(json.dumps({"judge": {"pid": 123, "started_at": 9.0}}))
    procs = load_managed_procs_from_result_json(f)
    assert len(procs) == 1 and procs[0].pid == 123 and procs[0].role == "judge"


# ---- 8. full manifest + log file generated ---- #
def test_manifest_written_atomically(tmp_path):
    code = tmp_path / "code.py"; code.write_text("print(1)")
    asset = tmp_path / "asset.bin"; asset.write_bytes(b"xyz")
    m = build_manifest(session_id="SID", auth_cap=4, python_executable="py",
                       python_version="3.13.5", code_files=[code], asset_files=[asset],
                       groups_planned=[{"group": "GROUP1", "n_games": 2}],
                       notes=["prior 429 was a tool rejection, not a game result"])
    p = write_manifest_atomic(tmp_path, m)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["session_id"] == "SID"
    assert loaded["auth_cap_games"] == 4
    assert loaded["code_hashes"][str(code)]
    assert loaded["notes"][0].startswith("prior 429")
    # atomic temp must be gone
    assert not (tmp_path / "manifest.json.tmp").exists()


# ---- 9. Group 2 never called when Group 1 not fully passing ---- #
def test_group2_only_when_group1_clean():
    assert should_run_group2(True) is True
    assert should_run_group2(False) is False


def test_strict_clean_passes_for_two_clean_attempts(tmp_path):
    attempts = _two_good_attempts_with_procs(tmp_path)
    ok, reasons = group1_strict_clean(attempts, _good_summary())
    assert ok is True, reasons

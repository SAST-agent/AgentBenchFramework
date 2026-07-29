"""Section 4 端到端贯通测试（MatchAttempt → GameOutcome → to_event_record → JSON）。

约束（与事前提上的契约一致）：

1. 不只构造最终 GameOutcome，必须经过包含完整 fake result-json 的
   ``_build_attempt_from_files`` 入口逐值验证。
2. 验证以下字段在「result-json→MatchAttempt→GameOutcome→to_event_record→JSON 落盘重读」
   全链逐字一致：
     * vendor exception 原文（``vendor_exception``，逐字 result-json 的 ``exception``）
     * wrapper exception 原文（``wrapper_exception``，外层 wrapper Popen/包装异常）
     * 有 reason 但无真实 exception 时 ``exception`` MUST 为 null（不得用 reason 填）
     * ``timeout``（result-json 原始 ``timeout`` 形态）与 ``timeout_s``（实际传入的逐步超时）
     * Judge/ai0/ai1 不同 final_returncode
     * ``process_cleanup`` 包含四个 role，且来源明确
       （vendor = ProcessTreeManager 外层；judge/ai0/ai1 = result-json 内层）
     * cleanup failure 走分类 ``cleanup_failure``
     * 内部 ``run_match_returncode`` 与外部 ``vendor_returncode`` 各自落事件且不混淆
     * Replay SHA-256
     * ``evidence_paths`` 全套保留
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


# --------------------------------------------------------------------------- #
# fake result-json shape (mirrors vendor/miracle_local/run_match.py output)
# --------------------------------------------------------------------------- #
def _inner_status(*, role, pid, final_returncode, **over):
    """Per-role status block the vendor writes into result-json."""
    s = {
        "role": role, "pid": pid, "started_at": 1000.0,
        "natural_exit": False, "natural_returncode": None,
        "termination_requested": False, "termination_reason": None,
        "final_returncode": final_returncode,
        "forced_kill": False, "cleanup_succeeded": True, "identity_confirmed": True,
    }
    s.update(over)
    return s


def _fake_result_json(tmp_path, tag, *, vendor_exception=None,
                      cleanup_all_succeeded=True, run_match_returncode=0,
                      end_info_received=True, end_info=None,
                      raw_winner=0, timeout_flag=None, ai_error=None,
                      ai0_final_returncode=0, ai1_final_returncode=0,
                      judge_final_returncode=0, scores=None,
                      score_tie=False):
    if timeout_flag is None:
        timeout_flag = {"ai0": False, "ai1": False}
    if ai_error is None:
        ai_error = {"ai0": False, "ai1": False}
    if end_info is None:
        end_info = {"0": 5, "1": 2}
    if scores is None:
        scores = {"0": 5, "1": 2}
    return {
        "schema_version": 1, "tag": str(tag),
        "started_at": 1000.0, "finished_at": 1005.0, "duration_s": 5.0,
        "judge_dir_resolved": str(tmp_path / "fake_judge"),
        "p0": {"name": "ifelse", "dir": str(tmp_path / "p0")},
        "p1": {"name": "rank01", "dir": str(tmp_path / "p1")},
        "judge": _inner_status(role="judge", pid=111, final_returncode=judge_final_returncode,
                               natural_exit=True, natural_returncode=judge_final_returncode),
        "ai0": _inner_status(role="ai0", pid=112, final_returncode=ai0_final_returncode),
        "ai1": _inner_status(role="ai1", pid=113, final_returncode=ai1_final_returncode),
        "end_info_received": end_info_received,
        "end_info": end_info,
        "scores": scores,
        "raw_winner": raw_winner,
        "score_tie": score_tie,
        "judge_tiebreak_applied": score_tie,
        "timeout": timeout_flag,
        "ai_error": ai_error,
        "trace_path": str(tmp_path / "work" / f"{tag}.jsonl"),
        "replay_path": str(tmp_path / "work" / f"{tag}.replay"),
        "cleanup_all_succeeded": cleanup_all_succeeded,
        "exception": vendor_exception,
        "run_match_returncode": run_match_returncode,
    }


def _vendor_manager_status(*, vendor_final_rc=-1, cleanup_succeeded=True,
                           forced_kill=False, termination_requested=True,
                           termination_reason="wrapper-timeout"):
    return [{
        "role": "vendor", "pid": 999, "started_at": 1000.0,
        "natural_exit": False, "natural_returncode": None,
        "termination_requested": termination_requested,
        "termination_reason": termination_reason,
        "final_returncode": vendor_final_rc,
        "forced_kill": forced_kill,
        "cleanup_succeeded": cleanup_succeeded,
        "identity_confirmed": True,
    }]





# =========================================================================== #
# 1. Main full-chain propagation test (vendor exception + wrapper exception +
#    timeout/timeout_s + per-role exits + 4-role cleanup + internal/external rc
#    + replay SHA + evidence_paths + reason without exception)
# =========================================================================== #
def test_full_chain_field_propagation_via_fake_result_json(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    from agentbench_frame.games.miracle.result import sha256_file

    tag = "m_rank01_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    vendor_exception_text = "RuntimeError('vendor AI read EOF before end')"
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=vendor_exception_text,
        cleanup_all_succeeded=False,
        run_match_returncode=7,
        ai0_final_returncode=1, ai1_final_returncode=0, judge_final_returncode=0,
        timeout_flag={"ai0": False, "ai1": True},
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    pop = payload.pop  # local alias
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    replay_bytes = struct.pack(">7i", 0, 0, 0, 1, 0, 0, 0) + b"\x00" * 16
    replay_path = work_dir / f"{tag}.replay"
    replay_path.write_bytes(replay_bytes)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    wrapper_exception_text = "OSError('wrapper Popen failed: phantom')"
    mgr_status = _vendor_manager_status(vendor_final_rc=-1, cleanup_succeeded=True,
                                        forced_kill=True,
                                        termination_reason="wrapper-timeout")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank01",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=-1, wrapper_timeout=True,
        wrapper_exception=wrapper_exception_text,
        vendor_manager_status=mgr_status,
        timeout_s=8.0, started=999.0, finished=1006.0,
        collision_detected=False,
    )

    # === MatchAttempt 级断言 ===
    assert att.vendor_exception == vendor_exception_text, "vendor_exception must be verbatim from result-json"
    assert att.wrapper_exception == wrapper_exception_text, "wrapper_exception must be verbatim outer Popen exc"
    assert att.timeout == {"ai0": False, "ai1": True}, "timeout must be the raw result-json timeout dict"
    assert att.timeout_s == 8.0, "timeout_s must be the run_match_attempt timeout kwarg"
    assert att.wrapper_timeout is True
    assert att.judge_exit == 0
    assert att.ai0_exit == 1
    assert att.ai1_exit == 0
    assert att.run_match_returncode == 7
    assert att.vendor_returncode == -1
    assert att.replay_sha256 == sha256_file(replay_path)
    assert att.reason and isinstance(att.reason, str)
    # exception compat rule: wrapper_exception 拥有最高优先级，无 reason 填充
    assert att.exception == wrapper_exception_text, \
        "exception compat field must equal wrapper_exception (NOT reason)"
    # evidence_paths 5 项保留
    assert set(att.evidence_paths.keys()) == {"stdout", "stderr", "trace", "replay", "result_json"}

    # process_cleanup 必须含四个 role + source 标记
    roles = sorted(r["role"] for r in att.process_cleanup)
    assert roles == ["ai0", "ai1", "judge", "vendor"], \
        f"process_cleanup must contain 4 roles, got {roles}"
    sources = {(r["role"], r["source"]) for r in att.process_cleanup}
    assert ("vendor", "process_tree_manager") in sources, \
        "vendor must be tagged source=process_tree_manager"
    assert ("judge", "result_json") in sources
    assert ("ai0", "result_json") in sources
    assert ("ai1", "result_json") in sources

    # === GameOutcome 级断言 ===
    o = attempt_to_outcome(att)
    assert o.vendor_exception == vendor_exception_text
    assert o.wrapper_exception == wrapper_exception_text
    assert o.timeout == {"ai0": False, "ai1": True}
    assert o.timeout_s == 8.0
    assert o.run_match_returncode == 7
    assert o.judge_exit == 0 and o.ai0_exit == 1 and o.ai1_exit == 0
    assert o.vendor_returncode == -1
    assert o.replay_sha256 == att.replay_sha256
    assert o.exception == wrapper_exception_text
    assert o.reason and o.reason == att.reason

    # === to_event_record + JSON 落盘重读 ===
    rec = to_event_record(o)
    rec_path = tmp_path / "sample_event.jsonl"
    rec_path.write_text(json.dumps(rec, default=str, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    loaded = json.loads(rec_path.read_text(encoding="utf-8").strip())

    assert loaded["timeout"] == {"ai0": False, "ai1": True}
    assert loaded["timeout_s"] == 8.0
    assert loaded["wrapper_timeout"] is True
    assert loaded["wrapper_exception"] == wrapper_exception_text
    assert loaded["vendor_exception"] == vendor_exception_text
    assert loaded["run_match_returncode"] == 7
    assert loaded["vendor_returncode"] == -1
    assert loaded["judge_exit"] == 0
    assert loaded["ai0_exit"] == 1
    assert loaded["ai1_exit"] == 0
    assert loaded["result_json_status"] == "ok"
    assert loaded["reason"] == att.reason
    # exception is null when no real exception — but here we have a real exception,
    # so it must equal wrapper_exception (the highest priority real exception source)
    assert loaded["exception"] == wrapper_exception_text
    # internal rc NOT silently feeding reason / exception:
    assert "RuntimeError" not in loaded.get("reason", "")  # reason doesn't echo exception text
    # replay sha same as the MatchAttempt's hash:
    assert loaded["replay_sha256"] == att.replay_sha256
    # evidence_paths preserved on the event layer too:
    assert loaded["evidence_paths"]["replay"].endswith(".replay")
    assert loaded["evidence_paths"]["result_json"].endswith(".result.json")


# =========================================================================== #
# 2. reason populated but no real exception → exception MUST be null
# =========================================================================== #
def test_reason_without_real_exception_yields_null_exception(tmp_path):
    """result-json MISSING case: classify produces a populated reason, but
    there is no real exception (no wrapper exception, no vendor exception),
    so the exception (compat field) MUST be null — never filled by reason."""
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record

    tag = "m_rank02_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    # do NOT write a result-json → load_result_json returns ("missing", None)
    (work_dir / f"{tag}.jsonl").write_text("", encoding="utf-8")          # empty trace
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")
    # no replay file

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank02",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False,
        wrapper_exception=None,           # no wrapper exception
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=True,
            termination_requested=False, termination_reason=None,
        ),
        timeout_s=12.0, started=1000.0, finished=1001.0, collision_detected=False,
    )

    # classify flags result_json_missing with a populated reason
    assert att.error_type == "result_json_missing"
    assert att.reason, "classify must produce a reason for the missing case"
    assert att.reason and att.reason is not None
    # but there is NO real exception anywhere → compat exception MUST be null
    assert att.exception is None, "no real exception → compat exception MUST be None"
    assert att.wrapper_exception is None
    assert att.vendor_exception is None
    assert att.valid is False

    o = attempt_to_outcome(att)
    assert o.exception is None
    assert o.reason and o.reason == att.reason

    rec = to_event_record(o)
    rec_path = tmp_path / "ev2.jsonl"
    rec_path.write_text(json.dumps(rec, default=str, ensure_ascii=False),
                        encoding="utf-8")
    loaded = json.loads(rec_path.read_text(encoding="utf-8"))
    assert loaded["exception"] is None, "exception sentinel must be null in JSON"
    # reason is still preserved verbatim — proves exception wasn't filled by reason
    assert loaded["reason"] == att.reason
    assert loaded["error_type"] == "result_json_missing"


# =========================================================================== #
# 3. vendor_exception only (no wrapper exception) — compat exception = vendor_exception
# =========================================================================== #
def test_vendor_exception_only_path(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record

    tag = "m_rank03_camp1"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    vendor_exc = "ValueError('vendor side bad JSON')"
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=vendor_exc,
        cleanup_all_succeeded=True, run_match_returncode=1,
        ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 1, 1, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank03",
        evaluated_agent_camp=1,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False,
        wrapper_exception=None,
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=True,
            termination_requested=False, termination_reason=None,
        ),
        timeout_s=8.0, started=1000.0, finished=1002.0, collision_detected=False,
    )
    assert att.vendor_exception == vendor_exc
    assert att.wrapper_exception is None
    # compat exception rule: wrapper_exception (None) OR vendor_exception → vendor_exception
    assert att.exception == vendor_exc, "compat exception must promote vendor_exception when no wrapper exception"
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    p = tmp_path / "ev3.jsonl"
    p.write_text(json.dumps(rec, default=str, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["vendor_exception"] == vendor_exc
    assert loaded["wrapper_exception"] is None
    assert loaded["exception"] == vendor_exc


# =========================================================================== #
# 4. cleanup failure classification path (no wrapper_timeout, both rcs 0, no exc)
# =========================================================================== #
def test_cleanup_failure_classification_path(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record

    tag = "m_rank04_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=None,
        cleanup_all_succeeded=False,
        run_match_returncode=0,
        ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 0, 0, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank04",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False,
        wrapper_exception=None,
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=False,
            termination_requested=True, termination_reason="post_end_info",
        ),
        timeout_s=8.0, started=1000.0, finished=1005.0, collision_detected=False,
    )
    assert att.error_type == "cleanup_failure", f"cleanup_failure not classified, got {att.error_type}: {att.reason}"
    assert att.valid is False
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    p = tmp_path / "ev4.jsonl"
    p.write_text(json.dumps(rec, default=str, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["error_type"] == "cleanup_failure"
    assert loaded["valid"] is False
    # cleanup failed role vendor still in process_cleanup, sourced from mgr
    assert any(r.get("role") == "vendor" and r.get("cleanup_succeeded") is False
               for r in loaded["process_cleanup"])


# =========================================================================== #
# 5. internal run_match_returncode lands separately from external vendor_returncode
# =========================================================================== #
def test_internal_run_match_returncode_lands_on_event_separately(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record
    from agentbench_frame.games.miracle.atomicio import atomic_write_json

    tag = "m_rank05_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=None,
        cleanup_all_succeeded=True, run_match_returncode=3,
        ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 1, 1, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank05",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0,            # outer wrapper returned 0 cleanly
        wrapper_timeout=False, wrapper_exception=None,
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=True,
            termination_requested=False, termination_reason=None,
        ),
        timeout_s=8.0, started=1000.0, finished=1005.0, collision_detected=False,
    )
    # classify must flag vendor_exception (internal rc != 0) and stop the matrix
    assert att.error_type == "vendor_exception", att.reason
    # the external rc stays 0, internal stays 3 (must not be conflated)
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    p = tmp_path / "ev5.jsonl"
    p.write_text(json.dumps(rec, default=str, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["run_match_returncode"] == 3
    assert loaded["vendor_returncode"] == 0    # both keys present and distinct:
    assert "run_match_returncode" in loaded
    assert "vendor_returncode" in loaded


# =========================================================================== #
# 6. evidence_paths 落事件且 result_json 路径以 .result.json 结尾
# =========================================================================== #
def test_evidence_paths_lands_on_event(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record
    from agentbench_frame.games.miracle.atomicio import atomic_write_json

    tag = "m_rank06_camp1"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=None, cleanup_all_succeeded=True, run_match_returncode=0,
        ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
        raw_winner=1, end_info={"0": 2, "1": 5}, scores={"0": 2, "1": 5},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 2, "1": 5})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 1, 0, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank06",
        evaluated_agent_camp=1,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False, wrapper_exception=None,
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=True,
            termination_requested=False, termination_reason=None,
        ),
        timeout_s=8.0, started=1000.0, finished=1001.0, collision_detected=False,
    )
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    assert all(k in rec["evidence_paths"] for k in
               ("stdout", "stderr", "trace", "replay", "result_json"))
    assert rec["evidence_paths"]["result_json"].endswith(f"{tag}.result.json")
    assert rec["evidence_paths"]["replay"].endswith(f"{tag}.replay")


# =========================================================================== #
# 7. timeout field shape — even when both ai flags are False, raw result-json
#    timeout dict must be preserved verbatim on the event
# =========================================================================== #
def test_timeout_field_raw_dict_preserved(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.runner import attempt_to_outcome
    from agentbench_frame.games.miracle.result import to_event_record
    from agentbench_frame.games.miracle.atomicio import atomic_write_json

    tag = "m_rank07_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    custom_timeout = {"ai0": True, "ai1": False}
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=None, cleanup_all_succeeded=True, run_match_returncode=0,
        ai0_final_returncode=2, ai1_final_returncode=0, judge_final_returncode=0,
        timeout_flag=custom_timeout,
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "ai_timeout", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 0, 1, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank07",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False, wrapper_exception=None,
        vendor_manager_status=_vendor_manager_status(
            vendor_final_rc=0, cleanup_succeeded=True,
            termination_requested=False, termination_reason=None,
        ),
        timeout_s=11.0, started=1000.0, finished=1002.0, collision_detected=False,
    )
    assert att.timeout == custom_timeout  # raw verbatim from result-json
    assert att.timeout_s == 11.0
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    p = tmp_path / "ev7.jsonl"
    p.write_text(json.dumps(rec, default=str, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["timeout"] == custom_timeout
    assert loaded["timeout_s"] == 11.0


# =========================================================================== #
# 8. boundary: _build_attempt_from_files must work even with vendor_manager_status=[]
#    (no vendor process; only inner judge/ai0/ai1 from result-json land in process_cleanup)
# =========================================================================== #
def test_process_cleanup_handles_empty_vendor_manager_status(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.atomicio import atomic_write_json

    tag = "m_rank08_camp0"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = _fake_result_json(
        tmp_path, tag,
        vendor_exception=None, cleanup_all_succeeded=True, run_match_returncode=0,
        ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
        raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
    )
    atomic_write_json(work_dir / f"{tag}.result.json", payload)
    ei = json.dumps({"0": 5, "1": 2})
    (work_dir / f"{tag}.jsonl").write_text(
        json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
        json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
        encoding="utf-8",
    )
    (work_dir / f"{tag}.replay").write_bytes(
        struct.pack(">7i", 0, 0, 0, 0, 0, 0, 0) + b"\x00" * 16)
    (work_dir / f"{tag}.stdout").write_text("out", encoding="utf-8")
    (work_dir / f"{tag}.stderr").write_text("err", encoding="utf-8")

    att = _build_attempt_from_files(
        game_id=tag, evaluated_agent="ifelse", opponent="rank08",
        evaluated_agent_camp=0,
        work_dir=work_dir, tag=tag,
        vendor_returncode=0, wrapper_timeout=False, wrapper_exception=None,
        vendor_manager_status=[],  # empty — only result-json inner procs come in
        timeout_s=8.0, started=1000.0, finished=1001.0, collision_detected=False,
    )
    roles = sorted(r["role"] for r in att.process_cleanup)
    assert roles == ["ai0", "ai1", "judge"]  # no vendor, only inner 3
    # each comes from result-json
    assert all(r["source"] == "result_json" for r in att.process_cleanup)


# =========================================================================== #
# 9. compat exception rule is deterministic & test-covered:
#    wrapper_exception + vendor_exception both present → wrapper wins (no overwrite)
#    wrapper=None, vendor=None  → None
#    wrapper set, vendor None → wrapper_exception
#    wrapper None, vendor set  → vendor_exception
# =========================================================================== #
def test_compat_exception_rule_deterministic(tmp_path):
    from agentbench_frame.games.miracle.match_runner import _build_attempt_from_files
    from agentbench_frame.games.miracle.atomicio import atomic_write_json

    def _build_for(*, wrapper_exception, vendor_exception):
        tag = f"test_{abs(hash((wrapper_exception, vendor_exception))) % 100000}"
        # use unique tag per call within the same tmp_path dir
        work_dir = tmp_path / f"wd_{tag}"
        work_dir.mkdir(parents=True, exist_ok=True)
        payload = _fake_result_json(
            tmp_path, tag,
            vendor_exception=vendor_exception,
            cleanup_all_succeeded=(vendor_exception is None),
            run_match_returncode=0 if vendor_exception is None else 1,
            ai0_final_returncode=0, ai1_final_returncode=0, judge_final_returncode=0,
            timeout_flag={"ai0": False, "ai1": False},
            raw_winner=0, end_info={"0": 5, "1": 2}, scores={"0": 5, "1": 2},
        )
        atomic_write_json(work_dir / f"{tag}.result.json", payload)
        ei = json.dumps({"0": 5, "1": 2})
        (work_dir / f"{tag}.jsonl").write_text(
            json.dumps({"kind": "ai_operation", "player": 0}) + "\n" +
            json.dumps({"kind": "match_end", "end_info": ei}) + "\n",
            encoding="utf-8",
        )
        (work_dir / f"{tag}.replay").write_bytes(
            struct.pack(">7i", 0, 0, 0, 0, 0, 0, 0) + b"\x00" * 16)
        (work_dir / f"{tag}.stdout").write_text("o", encoding="utf-8")
        (work_dir / f"{tag}.stderr").write_text("e", encoding="utf-8")
        att = _build_attempt_from_files(
            game_id=tag, evaluated_agent="ifelse", opponent="opp",
            evaluated_agent_camp=0,
            work_dir=work_dir, tag=tag,
            vendor_returncode=(0 if wrapper_exception is None else -1),
            wrapper_timeout=(wrapper_exception is not None),
            wrapper_exception=wrapper_exception,
            vendor_manager_status=_vendor_manager_status(
                vendor_final_rc=0, cleanup_succeeded=True,
                termination_requested=(wrapper_exception is not None),
                termination_reason="wrapper-timeout" if wrapper_exception else None,
            ),
            timeout_s=8.0, started=1000.0, finished=1002.0, collision_detected=False,
        )
        return att

    # case 1: wrapper + vendor → wrapper wins
    att = _build_for(wrapper_exception="WrapperErr('w')", vendor_exception="VendorErr('v')")
    assert att.wrapper_exception == "WrapperErr('w')" and att.vendor_exception == "VendorErr('v')"
    assert att.exception == "WrapperErr('w')"
    # case 2: both None → None
    att = _build_for(wrapper_exception=None, vendor_exception=None)
    assert att.exception is None
    # case 3: wrapper alone → wrapper exception
    att = _build_for(wrapper_exception="WrapperErr('w2')", vendor_exception=None)
    assert att.exception == "WrapperErr('w2')"
    # case 4: vendor alone → vendor_exception
    att = _build_for(wrapper_exception=None, vendor_exception="VendorErr('v2')")
    assert att.exception == "VendorErr('v2')"
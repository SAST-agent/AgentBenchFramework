"""Section 2 余下严格性缺口针对性红灯测试（24_miracle resume 验证链）。

覆盖上一位 Agent 完成的 verify_session_for_resume 与 resume() 之外的剩余缺口：

  1. progress 中未知 game_id 必须明确拒绝
  2. progress 中非法 state（非 not_started/running/done）必须明确拒绝
  3. opponent / build / code_hashes 字典必须精确键集合匹配（拒绝额外键）
  4. manifest 缺 session_id 必须拒绝
  5. plan_count 必须与真实 plan 长度 AND 32 严格一致
  6. complete-rank audit 必须解析、rank 一致、ok=True、两 game_id/camp 与 progress 一致
  7. partial rank（仅一 camp done）提前出现声称完整的 audit 文件必须拒绝
  8. CLI 级 fake 恢复必须经过 tools/miracle_matrix._run_resume 的 verify 链
  9. resume() 后不应再走宽松 load_progress（必须严格解析同一文件）
  10. events.jsonl 非 JSON 对象行（int/str/list 等顶层非 dict）必须保守拒绝

只读：所有测试均使用 tmp_path 临时夹具，不触碰权威 session；不启动 Judge/AI；
无 git 操作。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.matrix_runner import verify_session_for_resume


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_PLAN32 = [{"rank": r, "camp": c, "game_id": f"m_rank{r:02d}_camp{c}"}
           for r in range(1, 17) for c in (0, 1)]
_KNOWN_GIDS = {a["game_id"] for a in _PLAN32}


def _wm(tmp_path, *, progress=None, **over):
    """Write a full manifest with all identity fields. session_id 默认等于 dirname."""
    m = {
        "protocol_sha256": "a" * 64,
        "code_hashes": {"matrix": "abc", "matrix_runner": "def",
                        "match_runner": "ghi", "vendor_run_match": "jkl"},
        "plan_count": 32, "plan": _PLAN32, "run_id": "RID",
        "timeout": 8.0, "wrapper_timeout_s": 180.0,
        "ifelse_sha256": "if_sha", "judge_sha256": "j_sha",
        "opponent_archive_sha256": {f"rank{r:02d}": f"opp_{r}" for r in range(1, 17)},
        "cpp_build_sha256": {f"rank{r:02d}": f"bld_{r}" for r in range(1, 17)},
        "session_id": tmp_path.name, "python_version": "3.13.5", "platform": "test_plat",
    }
    m.update(over)
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    pr = progress if progress is not None else {"attempts": {}}
    (tmp_path / "progress.json").write_text(json.dumps(pr, ensure_ascii=False), encoding="utf-8")
    return m


def _done_entry(*, valid=True, norm="win", rank=1, camp=0):
    return {"state": "done", "valid": valid, "normalized_result": norm,
            "rank": rank, "camp": camp}


# =========================================================================== #
# 1. progress 未知 game_id 拒绝
# =========================================================================== #
def test_progress_unknown_game_id_rejected(tmp_path):
    proc = {"attempts": {"m_rank01_camp0": _done_entry(rank=1, camp=0),
                         "m_rank99_camp5": _done_entry(rank=99, camp=5)}}
    _wm(tmp_path, progress=proc)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("unknown" in e.lower() or "m_rank99" in e for e in errs), errs


# =========================================================================== #
# 2. progress 非法 state 拒绝
# =========================================================================== #
def test_progress_illegal_state_rejected(tmp_path):
    proc = {"attempts": {"m_rank01_camp0": {"state": "tilted", "rank": 1, "camp": 0}}}
    _wm(tmp_path, progress=proc)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("state" in e.lower() and ("illegal" in e.lower() or "invalid" in e.lower())
               for e in errs), errs


def test_progress_not_started_explicit_state_accepted(tmp_path):
    """state=not_started 显式出现也应被接受（合法，无事件对应）。"""
    proc = {"attempts": {"m_rank01_camp0": {"state": "not_started", "rank": 1, "camp": 0}}}
    _wm(tmp_path, progress=proc)
    # without events / running it should be all clean — but we don't pass expected_* IDs
    # so the unknown check must NOT flag a known id with legal not_started state.
    ok, errs = verify_session_for_resume(tmp_path)
    assert not any("m_rank01_camp0" in e and "unknown" in e.lower() for e in errs), errs


# =========================================================================== #
# 3. opponent/build/code_hashes 精确键集合匹配（拒绝额外键）
# =========================================================================== #
def test_manifest_opponent_extra_key_rejected(tmp_path):
    m = _wm(tmp_path)
    m["opponent_archive_sha256"]["rank99"] = "should_not_be_here"
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(
        tmp_path,
        expected_opponent_shas={r: f"opp_{r}" for r in range(1, 17)},
    )
    assert not ok
    assert any("opponent" in e.lower() and ("extra" in e.lower() or "unexpected" in e.lower())
               for e in errs), errs


def test_manifest_build_extra_key_rejected(tmp_path):
    m = _wm(tmp_path)
    m["cpp_build_sha256"]["rank99"] = "should_not_be_here"
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(
        tmp_path,
        expected_build_shas={r: f"bld_{r}" for r in range(1, 17)},
    )
    assert not ok
    assert any("build" in e.lower() and ("extra" in e.lower() or "unexpected" in e.lower())
               for e in errs), errs


def test_manifest_code_hash_extra_key_rejected(tmp_path):
    m = _wm(tmp_path)
    m["code_hashes"]["rogue_module"] = "should_not_be_here"
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(
        tmp_path,
        code_files={"matrix": str(tmp_path / "manifest.json"),  # any existing file
                    "matrix_runner": str(tmp_path / "manifest.json"),
                    "match_runner": str(tmp_path / "manifest.json")},
    )
    assert not ok
    assert any("code" in e.lower() and ("extra" in e.lower() or "unexpected" in e.lower())
               for e in errs), errs


def test_resume_rejects_control_input_hash_change(tmp_path):
    m = _wm(tmp_path)
    m["control_inputs"] = {
        "protocol": {"path": "/p.json", "sha256": "old"},
        "roster": {"path": "/r.json", "sha256": "same"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errors = verify_session_for_resume(
        tmp_path,
        expected_control_inputs={"protocol": "new", "roster": "same"},
    )
    assert not ok
    assert any("control input hash mismatch" in error for error in errors), errors


def test_manifest_opponent_missing_key_rejected(tmp_path):
    m = _wm(tmp_path)
    del m["opponent_archive_sha256"]["rank16"]
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(
        tmp_path,
        expected_opponent_shas={r: f"opp_{r}" for r in range(1, 17)},
    )
    assert not ok
    assert any("opponent" in e.lower() for e in errs), errs


# =========================================================================== #
# 4. manifest 缺 session_id 必须拒绝
# =========================================================================== #
def test_manifest_missing_session_id_rejected(tmp_path):
    m = _wm(tmp_path)
    del m["session_id"]
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("session_id" in e.lower() and ("missing" in e.lower() or "absent" in e.lower())
               for e in errs), errs


# =========================================================================== #
# 5. plan_count 与 plan 长度 AND 32 严格一致
# =========================================================================== #
def test_manifest_plan_count_mismatch_len_rejected(tmp_path):
    """plan_count 写 5 但 plan 还是 32 项 -> 拒绝。"""
    _wm(tmp_path, plan_count=5)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("plan_count" in e.lower() for e in errs), errs


def test_manifest_plan_count_field_missing_rejected(tmp_path):
    m = _wm(tmp_path)
    del m["plan_count"]
    (tmp_path / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("plan_count" in e.lower() for e in errs), errs


def test_manifest_plan_count_correct_passes(tmp_path):
    """plan_count=32 且 plan 长 32 时不报错（正反向安抚）。"""
    _wm(tmp_path, plan_count=32)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not any("plan_count" in e.lower() for e in errs), errs


# =========================================================================== #
# 6. complete-rank audit 必须解析、rank 一致、ok=True、game_id/camp 与 progress 一致
# =========================================================================== #
def _mk_audit_dir(tmp_path, rank):
    d = tmp_path / "audit"
    d.mkdir(exist_ok=True)
    return d / f"rank{rank:02d}.json"


def test_complete_rank_audit_corrupt_rejected(tmp_path):
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text("NOT JSON {{{", encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e and ("audit" in e.lower() and "corrupt" in e.lower())
               for e in errs), errs


def test_complete_rank_audit_rank_wrong_rejected(tmp_path):
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 99, "ok": True, "games": []}), encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e and "audit" in e.lower() for e in errs), errs


def test_complete_rank_audit_not_ok_rejected(tmp_path):
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 1, "ok": False, "reasons": ["x"],
                              "games": [{"game_id": "m_rank01_camp0", "camp": 0},
                                        {"game_id": "m_rank01_camp1", "camp": 1}]}),
                   encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e and "audit" in e.lower() and "ok" in e.lower() for e in errs), errs


def test_complete_rank_audit_game_ids_mismatch_rejected(tmp_path):
    """audit JSON 声称的 games 与 progress 的 done 状态 game_id 不一致 -> 拒绝。"""
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 1, "ok": True,
                              "games": [{"game_id": "m_rank01_camp0", "camp": 0},
                                        {"game_id": "m_rank01_camp0", "camp": 0}]}),
                   encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e and "audit" in e.lower()
               and ("game_id" in e.lower() or "mismatch" in e.lower()) for e in errs), errs


def test_complete_rank_audit_camps_mismatch_rejected(tmp_path):
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 1, "ok": True,
                              "games": [{"game_id": "m_rank01_camp0", "camp": 1},
                                        {"game_id": "m_rank01_camp1", "camp": 0}]}),
                   encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e and "audit" in e.lower() for e in errs), errs


def test_complete_rank_audit_valid_passes(tmp_path):
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        "m_rank01_camp1": _done_entry(rank=1, camp=1, norm="loss"),
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 1, "ok": True, "reasons": [],
                              "games": [{"game_id": "m_rank01_camp0", "camp": 0},
                                        {"game_id": "m_rank01_camp1", "camp": 1}]}),
                   encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    # the only errs possibly left must NOT be about audit
    assert not any("rank01" in e and "audit" in e.lower() for e in errs), errs


# =========================================================================== #
# 7. partial rank 提前出现声称完整的 audit 文件必须拒绝
# =========================================================================== #
def test_partial_rank_premature_complete_audit_rejected(tmp_path):
    """只有 camp0 done（part），却已存在一份标准 audit —— 必须拒绝。"""
    proc = {"attempts": {
        "m_rank01_camp0": _done_entry(rank=1, camp=0),
        # camp1 not_started (absent from attempts)
    }}
    _wm(tmp_path, progress=proc)
    af = _mk_audit_dir(tmp_path, 1)
    af.write_text(json.dumps({"rank": 1, "ok": True, "reasons": [],
                              "games": [{"game_id": "m_rank01_camp0", "camp": 0},
                                        {"game_id": "m_rank01_camp1", "camp": 1}]}),
                   encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("rank01" in e.lower() and
               ("premature" in e.lower() or "partial" in e.lower() or "pre" in e.lower())
               for e in errs), errs


# =========================================================================== #
# 8. CLI 级 fake 恢复必须经过 tools/miracle_matrix._run_resume 的 verify 链
# =========================================================================== #
def _import_cli_module(monkeypatch):
    """Import tools/miracle_matrix as a module. The module top-level calls
    paths.judge_dir() / ifelse_dir() which require AGENTBENCH_ROOT and
    MIRACLE_IFELSE_DIR env vars to be set (they only BUILD paths, never read
    them, so dummy values are safe — no Judge/AI is touched)."""
    repo = Path(__file__).resolve().parents[2]
    tools_dir = str(repo / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    # tools/miracle_matrix.py only ever uses these to construct Path objects;
    # it never reads the file system at import time. Dummy values are safe and
    # do NOT touch the authoritative Judge/AI matrix (see SKILL.md boundary).
    monkeypatch.setenv("AGENTBENCH_ROOT", str(repo))
    monkeypatch.setenv("MIRACLE_IFELSE_DIR", str(repo))
    # fresh import: drop any cached broken partial module so env stubs take effect
    sys.modules.pop("miracle_matrix", None)
    import importlib
    return importlib.import_module("miracle_matrix")


def test_cli_run_resume_invokes_verify_and_proceeds(tmp_path, monkeypatch):
    """verify 通过 -> resume() 被调用 + matrix.full.log 被写；execute() 不跑真实比赛。"""
    mm = _import_cli_module(monkeypatch)
    sid = "fake_sid_via_cli"
    sd = tmp_path / sid
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text(
        json.dumps({"run_id": "RID", "plan_count": 32, "plan": _PLAN32,
                    "timeout": 8.0, "wrapper_timeout_s": 180.0,
                    "session_id": sid}, ensure_ascii=False), encoding="utf-8")
    (sd / "progress.json").write_text('{"attempts":{}}', encoding="utf-8")
    monkeypatch.setattr(mm, "SESSION_ROOT", tmp_path)

    verified = []
    def fake_verify(sdp, **kw):
        verified.append(str(sdp))
        return (True, [])
    monkeypatch.setattr(mm, "verify_session_for_resume", fake_verify)

    resumed = {"count": 0}
    executed = {"count": 0}

    class FakeRunner:
        def __init__(self):
            self.session_dir = sd
            self.run_id = "RID"
            self.session_id = sid

        def resume(self, s):
            resumed["count"] += 1
            self.session_id = s

        def execute(self):
            executed["count"] += 1
            return {"completed": True, "halted": False}

        def write_run_compatible_output(self):
            return sd

        def aggregate_from_events(self):
            return {"total_attempts": 0, "valid_games": 0,
                    "invalid_games": 0, "win_rate": None}

    roots = {name: tmp_path / name for name in (
        "judge_dir", "ifelse_dir", "extracted_root", "archives_root",
        "precheck_root", "rank16_build_root",
    )}
    monkeypatch.setattr(mm, "verify_hashes", lambda *_a, **_k: [])
    rc = mm._run_resume(FakeRunner(), sid, **roots)
    assert rc == 0
    assert len(verified) == 1, "verify_session_for_resume must be called exactly once"
    assert resumed["count"] == 1, "r.resume() must be called after verify OK"
    assert executed["count"] == 1
    # matrix.full.log appended (resume path proceeds past verify)
    log_path = sd / "matrix.full.log"
    assert log_path.exists(), "matrix.full.log not written after resume"


def test_cli_run_resume_no_resume_when_verify_fails(tmp_path, monkeypatch):
    """verify 失败 -> 不调用 r.resume()/execute()/write，返回 2。"""
    mm = _import_cli_module(monkeypatch)
    sid = "fake_sid_fail"
    sd = tmp_path / sid
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text(
        json.dumps({"run_id": "RID", "session_id": sid}, ensure_ascii=False), encoding="utf-8")
    (sd / "progress.json").write_text('{"attempts":{}}', encoding="utf-8")
    monkeypatch.setattr(mm, "SESSION_ROOT", tmp_path)
    monkeypatch.setattr(mm, "verify_session_for_resume",
                        lambda *a, **k: (False, ["synthetic mismatch"]))

    resumed = {"count": 0}
    executed = {"count": 0}

    class FakeRunner:
        def __init__(self):
            self.session_dir = sd
            self.run_id = "RID"
            self.session_id = sid

        def resume(self, s):
            resumed["count"] += 1

        def execute(self):
            executed["count"] += 1
            return {"completed": True}

        def write_run_compatible_output(self):
            return sd

        def aggregate_from_events(self):
            return {}

    roots = {name: tmp_path / name for name in (
        "judge_dir", "ifelse_dir", "extracted_root", "archives_root",
        "precheck_root", "rank16_build_root",
    )}
    rc = mm._run_resume(FakeRunner(), sid, **roots)
    assert rc == 2, "verify-fail must return 2"
    assert resumed["count"] == 0, "must NOT resume when verify fails"
    assert executed["count"] == 0
    assert not (sd / "matrix.full.log").exists(), "log must NOT be opened when verify fails"


# =========================================================================== #
# 9. resume() 后不再走宽松 load_progress（同一文件严格解析一致）
# =========================================================================== #
def test_resume_strict_rejects_corrupt_progress(tmp_path):
    """验证通过后，manifest 没变、progress 立即被人为破坏成 NOT JSON —— 此时
    resume() 必须严格解析并拒绝，而不是宽松 load_progress 把它当成空 progress
    默默通过。"""
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner

    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=lambda **k: None,
        run_id="RID",
    )
    r.prepare_session()
    sid = r.session_id
    # write a valid manifest (so a fresh verify_session_for_resume passes by itself)
    r.record_manifest(opponent_hashes={i: "h" for i in range(1, 17)},
                      build_hashes={i: "b" for i in range(1, 17)},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    # corrupt progress (NOT JSON)
    (r.progress_path).write_text("NOT JSON {{{", encoding="utf-8")
    # resume() MUST raise on the strict parse — never silently return empty
    with pytest.raises((ValueError, json.JSONDecodeError, RuntimeError)):
        r2 = MatrixRunner(
            session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
            opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
            framework_src=tmp_path / "src", attempt_fn=lambda **k: None,
            run_id="RID2",
        )
        r2.resume(sid)


def test_resume_strict_rejects_non_dict_progress(tmp_path):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=lambda **k: None, run_id="RID",
    )
    r.prepare_session()
    sid = r.session_id
    r.record_manifest(opponent_hashes={i: "h" for i in range(1, 17)},
                      build_hashes={i: "b" for i in range(1, 17)},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    # progress is a JSON array, not an object
    (r.progress_path).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        r2 = MatrixRunner(
            session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
            opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
            framework_src=tmp_path / "src", attempt_fn=lambda **k: None, run_id="RID2",
        )
        r2.resume(sid)


# =========================================================================== #
# 10. events.jsonl 非 JSON 对象行（int / str / 数组顶层）必须保守拒绝
# =========================================================================== #
def test_events_non_json_object_line_rejected(tmp_path):
    """events.jsonl 含一行 `42`（合法 JSON 但顶层不是对象）—— 必须报错，不能崩。"""
    proc = {"attempts": {}}
    _wm(tmp_path, progress=proc)
    (tmp_path / "events.jsonl").write_text("42\n", encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("events" in e.lower() and
               ("corrupt" in e.lower() or "object" in e.lower() or "invalid" in e.lower())
               for e in errs), errs


def test_events_string_top_level_rejected(tmp_path):
    proc = {"attempts": {}}
    _wm(tmp_path, progress=proc)
    (tmp_path / "events.jsonl").write_text('"hello"\n', encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("events" in e.lower() for e in errs), errs


def test_events_array_top_level_rejected(tmp_path):
    proc = {"attempts": {}}
    _wm(tmp_path, progress=proc)
    (tmp_path / "events.jsonl").write_text('[1, 2, 3]\n', encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("events" in e.lower() for e in errs), errs


def test_events_no_keys_object_with_no_game_id_is_acceptable(tmp_path):
    """一个合法 JSON 对象但没有 game_id —— 不是 corrupt，不应崩，但因为无 game_id
    所以不影响 done/not_started 比对。只是确保 isinstance(e, dict) 保护生效，不报 corrupt。"""
    proc = {"attempts": {}}
    _wm(tmp_path, progress=proc)
    (tmp_path / "events.jsonl").write_text('{"event":"meta","note":"x"}\n', encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    # 不应出现 events corrupt 错误
    assert not any("events" in e.lower() and "corrupt" in e.lower() for e in errs), errs


# =========================================================================== #
# 额外: done 局 attempt_fn 调用次数严格为 0；不生成第二个 run 目录
# =========================================================================== #
def test_resume_done_games_call_attempt_fn_zero_times(tmp_path):
    """session 中 rank01 两 camp done；resume 后 execute() 必须不调用 attempt_fn。"""
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle import matrix as mx

    calls = []
    def fake_fn(**kw):
        calls.append(kw["game_id"])
        return None  # should not be reached

    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=fake_fn, run_id="ORIGINAL_RID",
    )
    r.prepare_session()
    r.record_manifest(opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
                      build_hashes={i: "b" + str(i) for i in range(1, 17)},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    mx.mark_done(r.progress, "m_rank01_camp0", _done_entry(rank=1, camp=0))
    mx.mark_done(r.progress, "m_rank01_camp1", _done_entry(rank=1, camp=1, norm="loss"))
    mx.write_progress_atomic(r.progress_path, r.progress)
    # events for the done games (audit requires 1 event each? no - audit checks
    # events equivalence; let's write minimal events)
    mx.append_event_atomic(r.events_path,
                          {"event": "game", "game_id": "m_rank01_camp0",
                           "valid": True, "normalized_result": "win"})
    mx.append_event_atomic(r.events_path,
                          {"event": "game", "game_id": "m_rank01_camp1",
                           "valid": True, "normalized_result": "loss"})
    # rank audit should exist for the done pair (so we don't trigger
    # "complete rank missing audit" — we write a VALID audit)
    audit_dir = r.session_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "rank01.json").write_text(json.dumps({
        "rank": 1, "ok": True, "reasons": [],
        "games": [{"game_id": "m_rank01_camp0", "camp": 0},
                  {"game_id": "m_rank01_camp1", "camp": 1}],
    }, ensure_ascii=False), encoding="utf-8")

    # new runner with DIFFERENT run_id -- resume must keep the original run_id
    r2 = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=fake_fn, run_id="WRONG",
    )
    r2.resume(r.session_id)
    assert r2.run_id == "ORIGINAL_RID"
    # only execute rank 1 (both camps done) — should not call attempt_fn
    res = r2.execute_rank(rank=1)
    assert res.get("halted") is False
    assert calls == [], f"attempt_fn called {len(calls)} times for done games: {calls}"
    # run_dir baked from ORIGINAL_RID — no second run dir created
    assert "WRONG" not in str(r2.run_dir)
    # session inventory unchanged (only ONE session dir under tmp_path)
    sessions = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(sessions) == 1, f"resume created extra session dirs: {sessions}"

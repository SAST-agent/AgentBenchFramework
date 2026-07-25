"""review #1 收口第 2 项：不可破坏重建测试。

确保 ``MatrixRunner.write_run_compatible_output`` 不会因中途异常丢弃既有
完整 run。如果重建在第 N 条事件或 summary 阶段失败，原 run 三文件（events
+ summary + run.toml）必须逐字节不变，且没有任何「半截 events」或
events/summary 不一致状态被暴露给后续读者。

策略：
- 在同文件系统临时目录生成完整候选 run；
- 对候选 run 执行事件质量、32 unique game_id、summary 重算、run_id 一致性检查；
- 全部通过后原子或带回滚地 promotion；promotion 失败必须能恢复旧版本；
- 正常连续调用仍保持一个 run_id、无重复事件。

注入点（覆盖 3 种失败窗口）：
- 写入候选 events.jsonl 第 7 条后异常；
- 写候选 summary.json 阶段异常；
- promotion 阶段（os.replace）失败 → 回滚。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _runner_with_three_games(tmp_path: Path):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle.matrix import (
        append_event_atomic, mark_done, write_progress_atomic,
    )
    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=lambda **k: None,
        run_id="PROMO_SAFE_RID",
    )
    r.prepare_session()
    r.record_manifest(opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
                      build_hashes={i: "b" + str(i) for i in range(1, 17)},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    for gid, valid, norm, rank, camp in [("m_rank01_camp0", True, "win", 1, 0),
                                          ("m_rank01_camp1", True, "loss", 1, 1),
                                          ("m_rank02_camp0", False, "error", 2, 0)]:
        mark_done(r.progress, gid, {"valid": valid, "normalized_result": norm,
                                    "rank": rank, "camp": camp,
                                    "steps": 100 if valid else 0})
        append_event_atomic(r.events_path, {
            "event": "game", "game_id": gid,
            "valid": valid, "normalized_result": norm,
            "rank": rank, "camp": camp,
            "steps": 100 if valid else 0,
            "judge_exit": 0, "ai0_exit": 0, "ai1_exit": 1,
            "replay_sha256": "x" * 64,
            "process_cleanup": [{"role": "vendor"}, {"role": "judge"}],
        })
    write_progress_atomic(r.progress_path, r.progress)
    return r


def _sha(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(run_dir: Path) -> dict:
    return {f: _sha(run_dir / f) for f in ("events.jsonl", "summary.json", "run.toml")}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_normal_consecutive_calls_keep_single_run_id_and_no_duplicates(tmp_path):
    """正常连续调用：保持一个 run_id、events 不重复。"""
    r = _runner_with_three_games(tmp_path)
    rd1 = r.write_run_compatible_output()
    rd2 = r.write_run_compatible_output()
    assert Path(rd1) == Path(rd2)
    # only ONE run directory
    sib = list(Path(rd1).parent.iterdir())
    assert len([p for p in sib if p.is_dir()]) == 1
    # events not duplicated
    lines = [l for l in (Path(rd1) / "events.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert len(lines) == 3


def test_failure_during_event_emit_keeps_existing_run_byte_for_byte(tmp_path):
    """已存在完整 run 第 1 次成功；第 2 次重建写入 candidate 时在第 2 条事件
    抛异常——原 run 三文件必须逐字节不变。

    注：第 2 次调用会写入 3 个事件（plus internal write 用于 events、budget 等）。
    第 1 次调用成功后 ``state['n']`` 已经历 N 次写入；第 2 次调用进入 build
    阶段时，我们从那里写入事件开始计时。
    """
    from agentbench_frame.games.miracle import matrix_runner as mr_mod
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    before = _snapshot(Path(rd))
    from agentbench_frame.tracking import run as run_mod
    orig_write = run_mod.Run.write

    # we want to fire on the SECOND call's 2nd game event; the bomb state
    # counts only writes AFTER we install it.
    state = {"n": 0, "see_re": re.compile(r"^game$")}

    def bomb(self, event_type=None, **kw):
        state["n"] += 1
        # only raise on the 2nd *game* event during this rebuild to target
        # a mid-stream failure window that doesn't deadlock the writer.
        if event_type == "game":
            state.setdefault("game_n", 0)
            state["game_n"] += 1
            if state["game_n"] == 2:
                raise RuntimeError("injected during event emit")
        return orig_write(self, event_type, **kw)
    run_mod.Run.write = bomb
    try:
        with pytest.raises(RuntimeError):
            r.write_run_compatible_output()
    finally:
        run_mod.Run.write = orig_write
    after = _snapshot(Path(rd))
    assert before == after, f"existing run mutated by aborted rebuild: {before} vs {after}"


def test_failure_during_summary_keeps_existing_run_byte_for_byte(tmp_path):
    """已存在完整 run；第 2 次重建在 run.finish 阶段失败——原 run 不变。"""
    from agentbench_frame.tracking import run as run_mod
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    before = _snapshot(Path(rd))
    orig_finish = run_mod.Run.finish

    def bomb(self, extra_summary=None):
        raise RuntimeError("injected during finish() summary build")
    run_mod.Run.finish = bomb
    try:
        with pytest.raises(RuntimeError):
            r.write_run_compatible_output()
    finally:
        run_mod.Run.finish = orig_finish
    after = _snapshot(Path(rd))
    assert before == after


def test_failure_during_promotion_keeps_existing_run_byte_for_byte(tmp_path):
    """Directory-level promotion failure (candidate→live rename) — must rollback,
    old live run byte-for-byte restored."""
    import os
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    before = _snapshot(Path(rd))
    # New directory-level _atomic_promote uses os.rename (not os.replace).
    # It does: rename(live→backup), rename(candidate→live).
    # Bomb only when the SOURCE is a staging/candidate dir (so the rollback
    # rename(backup→live) is NOT affected).
    from agentbench_frame.games.miracle import matrix_runner as mr_mod
    orig_rename = os.rename

    def bomb(src, dst):
        if ".staging" in str(src) or "staging" in str(src):
            raise OSError("injected promotion failure")
        return orig_rename(src, dst)

    orig = mr_mod.os.rename
    mr_mod.os.rename = bomb
    try:
        with pytest.raises(OSError, match="injected promotion"):
            r.write_run_compatible_output()
    finally:
        mr_mod.os.rename = orig
    after = _snapshot(Path(rd))
    assert before == after


def test_initial_build_when_no_existing_run(tmp_path):
    """当目标 run 目录不存在时，正常写成功；无回滚需求。"""
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    assert Path(rd).exists()
    assert (Path(rd) / "events.jsonl").exists()
    assert (Path(rd) / "summary.json").exists()
    assert (Path(rd) / "run.toml").exists()


def test_candidate_run_dir_is_temporary_not_under_live(tmp_path):
    """候选 run 不应暴露在最终 `runs/<game>/<agent>/` 下。一次成功调用
    结束后，没有 leftover candidate/staging sibling."""
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    sib = [p.name for p in Path(rd).parent.iterdir() if p.is_dir()]
    # exactly one: the live run_id directory
    assert "PROMO_SAFE_RID" in sib
    # nothing ending in .tmp / candidate / staging
    assert all(not (n.endswith(".tmp") or ".candidate" in n or ".staging" in n)
               for n in sib), f"leftover staging dirs: {sib}"


def test_candidate_validation_runs_event_quality_32_unique(tmp_path, monkeypatch):
    """强制最终 live events 包含 exactly3 events（此处用 3-cases 测试）；并且
    event_id 全局唯一（framework 的 inspect_event_file 自动断言）。"""
    from agentbench_frame.tracking.quality import inspect_event_file
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    rep = inspect_event_file(Path(rd) / "events.jsonl")
    assert rep.total_lines == 3
    assert rep.duplicate_event_ids == 0
    assert rep.missing_event_ids == 0
    assert rep.unknown_event_types == 0


def test_summary_recompute_matches_events_independent(tmp_path):
    """events↔summary 独立重算一致（保持 3 valid / 1 win / 1 loss / win_rate 0.5）。"""
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    evs = [json.loads(l) for l in (Path(rd) / "events.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    valid = [e for e in evs if e.get("valid")]
    wins = [e for e in valid if e.get("normalized_result") == "win"]
    losses = [e for e in valid if e.get("normalized_result") == "loss"]
    expected_wr = len(wins) / len(valid)
    s = json.loads((Path(rd) / "summary.json").read_text(encoding="utf-8"))
    assert abs(s["win_rate"] - expected_wr) < 1e-9
    assert s["total_episodes"] == len(valid)
    assert s["wins"] == len(wins)
    assert s["losses"] == len(losses)


def test_run_toml_totals_match_events(tmp_path):
    """run.toml 的 total_steps/total_episodes 应等于 events 有效游戏重算。"""
    import tomllib
    r = _runner_with_three_games(tmp_path)
    rd = r.write_run_compatible_output()
    evs = [json.loads(l) for l in (Path(rd) / "events.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    valid = [e for e in evs if e.get("valid")]
    expected_steps = sum(int(e.get("steps", 0)) for e in valid)
    meta = tomllib.loads((Path(rd) / "run.toml").read_text(encoding="utf-8"))
    assert meta["run"]["total_episodes"] == len(valid)
    assert meta["run"]["total_steps"] == expected_steps


def test_real_child_kill_after_live_rename_recovers_original_bytes(tmp_path):
    """A terminated writer must restore all three old live files byte-for-byte.

    This is intentionally a real subprocess and ``os._exit``, not an
    ``os.rename`` mock.  The child exits in the window after live->backup and
    before candidate->live; the recovery entry must prefer the old backup.
    """
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner

    live = tmp_path / "live"
    candidate = tmp_path / "candidate"
    live.mkdir()
    candidate.mkdir()
    original = {
        "events.jsonl": b'{"old": "events"}\n',
        "summary.json": b'{"old": "summary"}\n',
        "run.toml": b'[run]\nold = "toml"\n',
    }
    replacement = {
        "events.jsonl": b'{"new": "events"}\n',
        "summary.json": b'{"new": "summary"}\n',
        "run.toml": b'[run]\nnew = "toml"\n',
    }
    for name, data in original.items():
        (live / name).write_bytes(data)
    for name, data in replacement.items():
        (candidate / name).write_bytes(data)

    child = (
        "from pathlib import Path; import sys; "
        "from agentbench_frame.games.miracle.matrix_runner import MatrixRunner; "
        "MatrixRunner._atomic_promote(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    env = os.environ.copy()
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    env["MIRACLE_TEST_KILL_AFTER_LIVE_RENAME"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", child, str(candidate), str(live)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 86, result.stderr
    backup = live.with_name(live.name + ".backup_promote")
    marker = live.with_name(live.name + ".promote_marker.json")
    assert not live.exists() and backup.exists() and marker.exists()

    MatrixRunner._recover_promotion_transaction(live, backup, marker)
    assert all((live / name).read_bytes() == data for name, data in original.items())
    assert not backup.exists() and not marker.exists()

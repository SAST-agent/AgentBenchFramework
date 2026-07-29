"""Regression: data check must handle UTF-8 Chinese content on Windows (GBK-safe read)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_data_check_handles_utf8_chinese(tmp_path):
    """summary.json with Chinese config must not cause GBK decode failure."""
    rd = tmp_path / "runs" / "24_miracle" / "test" / "r1"
    rd.mkdir(parents=True)
    (rd / "run.toml").write_text(
        '[run]\nrun_id = "r1"\ngame = "24_miracle"\nagent = "test"\n'
        'type = "eval"\ncreated = "2026-01-01T00:00:00Z"\n', encoding="utf-8")
    (rd / "summary.json").write_text(json.dumps({
        "run_id": "r1", "game": "24_miracle", "agent": "test",
        "wall_hours": 1.0, "total_steps": 100, "win_rate": 0.5,
        "config": {"note": "中文测试字符 — 小样本声明"},
    }, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "agentbench_frame.cli", "data", "check",
         "--data-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=10, env=env)
    assert r.returncode == 0, f"data check failed: {r.stderr}"
    assert "1 valid" in r.stdout

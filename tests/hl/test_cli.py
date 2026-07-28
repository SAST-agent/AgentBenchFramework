"""Tests for hl/cli.py — the run driver.

No real ``claude`` or logic subprocess is spawned. ``ClaudeCodeRunner.run`` is
monkeypatched to a scripted edit (so the controller's act loop runs), and the
LostSpaceEvaluator is swapped for a stub via the lostspace cli's
``evaluator_class`` injection seam — here we exercise the HL cli's
``_evaluator_factory`` directly instead, plus an end-to-end smoke via a
monkeypatched runner + a stub evaluator module path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.hl import cli as hl_cli
from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.reference import ReferenceStateSet
from agentbench_frame.hl.reference_seed import build_seed


def _make_initial_candidate(tmp_path: Path) -> Path:
    d = tmp_path / "cand"
    d.mkdir()
    (d / "agent.py").write_text(
        "THRESHOLD = 10\nimport json,sys,struct\n"
        "# minimal stub agent\n", encoding="utf-8")
    (d / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return d


def _write_reference(tmp_path: Path) -> Path:
    p = tmp_path / "nu.json"
    build_seed("hl-test").save(p)
    return p


def test_resolve_opponents_requires_at_least_one():
    args = type("A", (), {"ladder_opponent": [], "opponent": []})()
    with pytest.raises(SystemExit):
        hl_cli._resolve_opponents(args)


def test_seed_codebase_copies_agent_and_manifest(tmp_path):
    src = _make_initial_candidate(tmp_path)
    ws = tmp_path / "ws"
    hl_cli._seed_codebase(src, ws)
    assert (ws / "agent.py").exists()
    assert (ws / "manifest.toml").exists()


def test_seed_codebase_rejects_dir_without_agent(tmp_path):
    src = tmp_path / "cand"
    src.mkdir()
    (src / "README.md").write_text("no agent here", encoding="utf-8")
    with pytest.raises(SystemExit):
        hl_cli._seed_codebase(src, tmp_path / "ws")


def test_main_runs_acts_with_stubbed_runner_and_eval(monkeypatch, tmp_path):
    """End-to-end: main() wires ClaudeCodeRunner + LostSpaceEvaluator factory
    and runs N acts. Both heavy pieces are stubbed."""
    ref = _write_reference(tmp_path)
    cand = _make_initial_candidate(tmp_path)
    data_root = tmp_path / "data"

    # Stub the runner: each act appends a line so the version changes.
    calls = {"n": 0}

    class _StubRunner:
        def __init__(self, **kw):
            self.kw = kw
        def run(self, *, workspace, context):
            calls["n"] += 1
            (workspace / "agent.py").write_text(
                f"THRESHOLD = {10 + calls['n']}\n", encoding="utf-8")
            from agentbench_frame.hl.runner import AgentRunResult
            return AgentRunResult(edit_type="parametrize",
                                  files_touched=["agent.py"],
                                  prompt_tokens=100,
                                  completion_tokens=20,
                                  total_tokens=120, time_s=0.1)

    # Stub the evaluator factory: returns an object whose .evaluate() gives a
    # fixed summary, ignoring the real LostSpaceEvaluator entirely.
    class _StubEval:
        def __init__(self, *a, **k): pass
        def evaluate(self):
            class R:
                summary = {"win_rate": 0.5, "evaluation_status": "complete",
                           "lostspace": {"aggregate": {"win_rate": 0.5}}}
                matches = []
                error_count = 0
                run_dir = tmp_path
            return R()

    monkeypatch.setattr("agentbench_frame.hl.runner.ClaudeCodeRunner", _StubRunner)
    monkeypatch.setattr(hl_cli, "LostSpaceEvaluator", lambda *a, **k: _StubEval())
    monkeypatch.setattr(hl_cli, "_resolve_opponents",
                        lambda args: [hl_cli.Opponent(name="rank06", command="echo")])
    monkeypatch.setattr(hl_cli, "_default_filler_command", lambda: "echo")
    # stub the probe so _probe_version doesn't spawn a real subprocess
    from agentbench_frame.hl import controller as ctrl_mod

    class _StubProbe:
        def __init__(self, *a, **k): pass
        def probe_set(self, samples):
            from agentbench_frame.hl.probe import EmittedAction
            return [EmittedAction(primitive=("finish",), out_of_support=False,
                                  sample_index=i) for i in range(len(samples))]
        def close(self): pass
    monkeypatch.setattr(ctrl_mod, "ReferenceProbe", _StubProbe)

    code = hl_cli.main([
        "--logic", "echo",
        "--initial-candidate", str(cand),
        "--name", "hl-test",
        "--reference", str(ref),
        "--ladder-opponent", "rank=6",
        "--acts", "2", "--pairs", "1", "--seats", "0", "--timeout", "5",
        "--data-dir", str(data_root),
        "--codebase-root", str(tmp_path / "cb"),
        "--dangerously-skip-permissions",
    ])
    assert code == 0
    assert calls["n"] == 2
    events_path = tmp_path / "cb" / "events.jsonl"
    assert events_path.exists()
    lines = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    types = [e["event_type"] for e in lines]
    assert types.count("agent_act") == 2
    assert types.count("version") == 2
    assert types.count("eval") == 2
    assert "policy_kl" in types  # second act compares against first


# ---- logic interpreter probe (D1) ----

def test_rewrite_logic_python_swaps_bare_python():
    from agentbench_frame.hl.cli import _rewrite_logic_python
    cmd = 'cd /d "E:/x/gamecode_logic" && python main.py'
    out = _rewrite_logic_python(cmd, "C:/torch/python.exe")
    assert "C:/torch/python.exe" in out
    assert "python main.py" not in out  # bare python replaced


def test_rewrite_logic_python_noop_when_unset():
    from agentbench_frame.hl.cli import _rewrite_logic_python
    cmd = 'cd /d X && python main.py'
    assert _rewrite_logic_python(cmd, None) == cmd


def test_probe_logic_antlr4_skips_when_no_python_token():
    """A logic command with no python token (e.g. 'echo') is not probed."""
    from agentbench_frame.hl.cli import _probe_logic_antlr4
    # Should not raise (no python token to probe).
    _probe_logic_antlr4("echo hello")


def test_probe_logic_antlr4_fails_fast_on_missing_dep(tmp_path):
    """An interpreter that can't import antlr4 -> SystemExit with a named msg.

    We simulate the interpreter with a batch file named ``python.bat`` (Windows
    PATHEXT resolves a bare ``python`` token to it) that exits non-zero with
    the antlr4 message — exactly what a real interpreter without
    antlr4-python3-runtime would do.
    """
    from agentbench_frame.hl.cli import _probe_logic_antlr4
    bat = tmp_path / "python_interpreter.bat"
    bat.write_text(
        "@echo off\r\necho No module named antlr4 1>&2\r\nexit /b 1\r\n",
        encoding="utf-8")
    cmd = f'cd /d "{tmp_path}" && "{bat}" main.py'
    with pytest.raises(SystemExit) as ei:
        _probe_logic_antlr4(cmd)
    msg = str(ei.value)
    assert "antlr4" in msg.lower()
    assert "antlr4-python3-runtime" in msg or "--logic-python" in msg


def test_probe_logic_antlr4_passes_when_importable(tmp_path):
    """An interpreter that imports antlr4 (we stub the probe target) passes.

    Uses an absolute python.exe-named batch so the probe runs *our* stub,
    not the system python.
    """
    from agentbench_frame.hl.cli import _probe_logic_antlr4
    bat = tmp_path / "python.exe"
    # A .exe-named batch won't run as a batch; instead use an absolute path
    # to a batch named python_interpreter.bat whose basename starts with 'python'.
    bat = tmp_path / "python_interpreter.bat"
    bat.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    cmd = f'cd /d "{tmp_path}" && "{bat}" main.py'
    # Should not raise.
    _probe_logic_antlr4(cmd)

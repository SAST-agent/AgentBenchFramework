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

    # Stub model loading
    from agentbench_frame.hl.models_config import ModelEntry
    def _stub_load_models(*a, **k):
        return {"stub": ModelEntry(label="stub", provider="stub", model="stub-model",
                                   api_key="stub", base_url="http://stub")}
    monkeypatch.setattr("agentbench_frame.hl.models_config.load_models", _stub_load_models)

    # Stub build_client to return a dummy client
    class _StubClient:
        def __init__(self, entry): pass
        def generate(self, **kw): raise AssertionError("should not be called")
    monkeypatch.setattr("agentbench_frame.hl.llm.build_client", _StubClient)

    monkeypatch.setattr("agentbench_frame.hl.runner.ApiCodingRunner", _StubRunner)
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


# ---- A1: evaluator must write under the stable run name ----

def test_evaluator_factory_uses_stable_run_name_as_candidate(tmp_path, monkeypatch):
    """A1 regression: the evaluator must write runs under the stable run name
    (args.name), NOT 'hl-<version_id>', so MatchHistoryView(agent=name) and the
    prompt's replay pointer resolve to real data.

    The original code used ``f"hl-{version.version_id}"[:40]`` as candidate_name,
    while ContextBuilder reads under ``args.name`` — the two never matched, so
    every act's prompt got an empty match-history table and no replay path.
    """
    from agentbench_frame.hl.reference import BenchmarkSpec
    from agentbench_frame.hl.codebase import HLCodebase

    cb = HLCodebase(root=tmp_path / "ws", store=tmp_path / "store")
    captured = {}

    class _SpyEvaluator:
        def __init__(self, *a, **kw):
            captured.update(kw)

    monkeypatch.setattr(hl_cli, "LostSpaceEvaluator", _SpyEvaluator)
    # Skip real snapshot staging — we only care what candidate_name is passed.
    monkeypatch.setattr(
        "agentbench_frame.hl.adapter.candidate_command",
        lambda version, *, store, dest, python=None: (["python", str(dest)], dest),
    )

    factory = hl_cli._evaluator_factory(
        "echo", [hl_cli.Opponent(name="rank06", command="echo")], "echo",
        codebase=cb, stage_root=tmp_path / "stage", data_root=tmp_path / "data",
    )

    class _V:
        version_id = "v2026-07-29T052332974022Z_000001"
        content_hash = "abc123"
        parent_version_id = None
        edit_type = "parametrize"

    spec = BenchmarkSpec(spec_id="bench-v1", opponents=("rank06",),
                         pairs=1, seats="0", timeout=5.0)
    factory(version=_V(), spec=spec, run_id="hl-v1")

    assert captured["candidate_name"] == "hl-v1"  # stable run name, not hl-<version_id>


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


# ---- auto-naming: --name optional, monotonic round counter, fixed seed ----

def _stub_for_main(monkeypatch, tmp_path):
    """Stub the heavy pieces (runner, evaluator, opponents, probe) so main()
    runs end-to-end without spawning claude or the logic subprocess."""
    class _StubRunner:
        def __init__(self, **kw):
            self.kw = kw
        def run(self, *, workspace, context):
            (workspace / "agent.py").write_text("# stub edit\n", encoding="utf-8")
            from agentbench_frame.hl.runner import AgentRunResult
            return AgentRunResult(edit_type="parametrize",
                                  files_touched=["agent.py"],
                                  prompt_tokens=10, completion_tokens=5,
                                  total_tokens=15, time_s=0.1)

    class _StubEval:
        def __init__(self, *a, **k):
            pass
        def evaluate(self):
            class R:
                summary = {"win_rate": 0.5, "evaluation_status": "complete",
                           "lostspace": {"aggregate": {"win_rate": 0.5}}}
                matches = []
                error_count = 0
                run_dir = tmp_path
            return R()

    # Stub model loading
    from agentbench_frame.hl.models_config import ModelEntry
    def _stub_load_models(*a, **k):
        return {"stub": ModelEntry(label="stub", provider="stub", model="stub-model",
                                   api_key="stub", base_url="http://stub")}
    monkeypatch.setattr("agentbench_frame.hl.models_config.load_models", _stub_load_models)

    # Stub build_client to return a dummy client
    class _StubClient:
        def __init__(self, entry): pass
        def generate(self, **kw): raise AssertionError("should not be called")
    monkeypatch.setattr("agentbench_frame.hl.llm.build_client", _StubClient)

    # Stub the new ApiCodingRunner
    monkeypatch.setattr("agentbench_frame.hl.runner.ApiCodingRunner", _StubRunner)
    monkeypatch.setattr(hl_cli, "LostSpaceEvaluator", lambda *a, **k: _StubEval())
    monkeypatch.setattr(hl_cli, "_resolve_opponents",
                        lambda args: [hl_cli.Opponent(name="rank06", command="echo")])
    monkeypatch.setattr(hl_cli, "_default_filler_command", lambda: "echo")
    from agentbench_frame.hl import controller as ctrl_mod

    class _StubProbe:
        def __init__(self, *a, **k):
            pass
        def probe_set(self, samples):
            from agentbench_frame.hl.probe import EmittedAction
            return [EmittedAction(primitive=("finish",), out_of_support=False,
                                  sample_index=i) for i in range(len(samples))]
        def close(self):
            pass
    monkeypatch.setattr(ctrl_mod, "ReferenceProbe", _StubProbe)


def test_auto_name_round_progresses(monkeypatch, tmp_path):
    """Omitting --name auto-generates hl-v<date>-round<n>; two runs -> round1
    then round2, distinct codebases, counter persisted at 2."""
    _stub_for_main(monkeypatch, tmp_path)
    ref = _write_reference(tmp_path)
    monkeypatch.chdir(tmp_path)

    common = ["--logic", "echo", "--reference", str(ref),
              "--ladder-opponent", "rank=6",
              "--acts", "1", "--pairs", "1", "--seats", "0", "--timeout", "5"]
    assert hl_cli.main(common) == 0
    assert hl_cli.main(common) == 0

    from agentbench_frame.hl.naming import round_name, today_date
    r1 = round_name(today_date(), 1)
    r2 = round_name(today_date(), 2)
    assert (tmp_path / ".hl_codebase" / r1 / "events.jsonl").exists()
    assert (tmp_path / ".hl_codebase" / r2 / "events.jsonl").exists()
    state = json.loads(
        (tmp_path / ".hl_codebase" / "hl_state.json").read_text(encoding="utf-8"))
    assert state["last_round"] == 2


def test_name_override_skips_counter(monkeypatch, tmp_path):
    """--name uses the name verbatim and does NOT touch the round counter."""
    _stub_for_main(monkeypatch, tmp_path)
    ref = _write_reference(tmp_path)
    monkeypatch.chdir(tmp_path)

    code = hl_cli.main([
        "--logic", "echo", "--reference", str(ref),
        "--name", "my-exp", "--ladder-opponent", "rank=6",
        "--acts", "1", "--pairs", "1", "--seats", "0", "--timeout", "5",
    ])
    assert code == 0
    assert (tmp_path / ".hl_codebase" / "my-exp" / "events.jsonl").exists()
    assert not (tmp_path / ".hl_codebase" / "hl_state.json").exists()
    lines = [json.loads(l) for l in
             (tmp_path / ".hl_codebase" / "my-exp" / "events.jsonl")
             .read_text().splitlines() if l.strip()]
    act_ids = {e["act_id"] for e in lines if "act_id" in e}
    assert act_ids == {"my-exp-000001"}


def test_default_seed_is_v1(monkeypatch, tmp_path):
    """Omitting --initial-candidate seeds the workspace from candidates/v1."""
    _stub_for_main(monkeypatch, tmp_path)
    ref = _write_reference(tmp_path)
    monkeypatch.chdir(tmp_path)

    code = hl_cli.main([
        "--logic", "echo", "--reference", str(ref),
        "--name", "seedtest", "--ladder-opponent", "rank=6",
        "--acts", "0", "--pairs", "1", "--seats", "0", "--timeout", "5",
    ])
    assert code == 0
    ws_agent = (tmp_path / ".hl_codebase" / "seedtest" / "workspace" / "agent.py")
    v1_agent = (Path(hl_cli.__file__).resolve().parent.parent
                / "lostspace" / "candidates" / "v1" / "agent.py")
    assert ws_agent.read_bytes() == v1_agent.read_bytes()


def test_system_prompt_states_interprops_schema_and_safe_pattern():
    """The durable system prompt must warn that interprops are int/object-coded
    (so 'X' in interprops is always False) and point to the blind-call-then-
    check-success pattern. This is the regression that zeroed key collection
    in hl-curriculum-0731."""
    from agentbench_frame.hl.cli import _system_prompt

    s = _system_prompt()
    assert "interprops" in s.lower()
    assert "always false" in s.lower()          # 'X' in interprops is always False
    assert "success" in s.lower()               # branch on result['success']
    assert "KeyMachine" in s

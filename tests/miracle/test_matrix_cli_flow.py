from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _import_cli_module(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    tools_dir = str(repo / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    monkeypatch.setenv("AGENTBENCH_ROOT", str(repo))
    monkeypatch.setenv("MIRACLE_IFELSE_DIR", str(repo))
    sys.modules.pop("miracle_matrix", None)
    return importlib.import_module("miracle_matrix")


def test_main_missing_protocol_returns_preflight_error_without_session(
    tmp_path, monkeypatch, capsys
):
    mm = _import_cli_module(monkeypatch)
    rc = mm.main([
        "--dry-run",
        "--protocol", str(tmp_path / "missing.json"),
        "--roster", str(tmp_path / "roster.json"),
        "--session-root", str(tmp_path / "sessions"),
    ])

    assert rc == 2
    assert "protocol file missing" in capsys.readouterr().err
    assert not (tmp_path / "sessions").exists()


def test_cli_resume_verifier_receives_control_hashes(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    sid = "sid"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    monkeypatch.setattr(mm, "SESSION_ROOT", tmp_path)
    observed = {}

    def fake_verify(_session_dir, **kwargs):
        observed.update(kwargs)
        return False, ["stop"]

    monkeypatch.setattr(mm, "verify_session_for_resume", fake_verify)

    class FakeRunner:
        def __init__(self):
            self.session_dir = session_dir

    control_inputs = mm.ControlInputs(
        protocol=json.loads(mm.PROTOCOL.read_text(encoding="utf-8")),
        roster=json.loads(mm.ROSTER.read_text(encoding="utf-8")),
        hashes={"protocol": "p", "roster": "r"},
    )
    rc = mm._run_resume(
        FakeRunner(), sid,
        control_inputs=control_inputs,
    )

    assert rc == 2
    assert observed["expected_control_inputs"] == {"protocol": "p", "roster": "r"}

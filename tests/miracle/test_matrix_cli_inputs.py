from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def _import_cli_module(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    tools_dir = str(repo / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    monkeypatch.setenv("AGENTBENCH_ROOT", str(repo))
    monkeypatch.setenv("MIRACLE_IFELSE_DIR", str(repo))
    sys.modules.pop("miracle_matrix", None)
    return importlib.import_module("miracle_matrix")


def test_load_control_inputs_reports_missing_file(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)

    with pytest.raises(mm.PreflightError, match="protocol file missing"):
        mm.load_control_inputs(tmp_path / "missing.json", tmp_path / "roster.json")


def test_parse_args_accepts_explicit_control_paths(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    args = mm.parse_args([
        "--dry-run",
        "--protocol", str(tmp_path / "p.json"),
        "--roster", str(tmp_path / "r.json"),
    ])

    assert args.dry_run is True
    assert args.protocol == tmp_path / "p.json"
    assert args.roster == tmp_path / "r.json"


def test_load_control_inputs_validates_protocol_and_roster(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    protocol = {"protocol_version": "test"}
    roster = {"strategies": [{"rank": rank} for rank in range(1, 17)]}
    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    roster_path.write_text(json.dumps(roster), encoding="utf-8")

    inputs = mm.load_control_inputs(protocol_path, roster_path)

    assert inputs.protocol == protocol
    assert inputs.roster == roster
    assert set(inputs.hashes) == {"protocol", "roster"}
    assert len(inputs.hashes["protocol"]) == 64
    assert len(inputs.hashes["roster"]) == 64


def test_load_control_inputs_rejects_non_contiguous_roster(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    protocol_path.write_text("{}", encoding="utf-8")
    roster_path.write_text(
        json.dumps({"strategies": [{"rank": 1}, {"rank": 3}]}),
        encoding="utf-8",
    )

    with pytest.raises(mm.PreflightError, match="roster ranks"):
        mm.load_control_inputs(protocol_path, roster_path)

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def _valid_protocol(mm):
    return {
        "frozen_identities": {
            "evaluated_agent": {"sha256": "a" * 64},
            "judge": {"main_py_sha256": "b" * 64},
            "build_artifacts_win64_mingw": {
                f"rank{rank:02d}": "c" * 64 for rank in mm.CPP_RANKS
            },
        }
    }


def _valid_roster(mm):
    strategies = []
    for rank in range(1, 17):
        strategy = {"rank": rank, "archive_sha256": "d" * 64}
        if rank in mm.PYTHON_RANKS:
            strategy.update({"entry": "main.py", "runnable_sha256": "e" * 64})
        strategies.append(strategy)
    return {"strategies": strategies}


def _write_controls(protocol_path, roster_path, protocol, roster):
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    roster_path.write_text(json.dumps(roster), encoding="utf-8")


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
    protocol = _valid_protocol(mm)
    roster = _valid_roster(mm)
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


@pytest.mark.parametrize(
    "missing_field",
    [
        "frozen_identities",
        "evaluated_agent_sha",
        "judge_sha",
        *[f"cpp_sha:{rank}" for rank in (1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16)],
        *[f"archive_sha:{rank}" for rank in range(1, 17)],
        *[f"python_entry:{rank}" for rank in (4, 5, 7)],
        *[f"python_runnable_sha:{rank}" for rank in (4, 5, 7)],
    ],
)
def test_main_rejects_incomplete_control_schema_before_session_write(
    tmp_path, monkeypatch, capsys, missing_field
):
    mm = _import_cli_module(monkeypatch)
    protocol = _valid_protocol(mm)
    roster = _valid_roster(mm)
    identities = protocol["frozen_identities"]
    if missing_field == "frozen_identities":
        protocol.pop("frozen_identities")
    elif missing_field == "evaluated_agent_sha":
        identities["evaluated_agent"].pop("sha256")
    elif missing_field == "judge_sha":
        identities["judge"].pop("main_py_sha256")
    elif missing_field.startswith("cpp_sha:"):
        rank = int(missing_field.split(":", 1)[1])
        identities["build_artifacts_win64_mingw"].pop(f"rank{rank:02d}")
    elif missing_field.startswith("archive_sha:"):
        rank = int(missing_field.split(":", 1)[1])
        roster["strategies"][rank - 1].pop("archive_sha256")
    elif missing_field.startswith("python_entry:"):
        rank = int(missing_field.split(":", 1)[1])
        roster["strategies"][rank - 1].pop("entry")
    else:
        rank = int(missing_field.split(":", 1)[1])
        roster["strategies"][rank - 1].pop("runnable_sha256")

    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    session_root = tmp_path / "sessions"
    _write_controls(protocol_path, roster_path, protocol, roster)
    constructed = []
    prepared = []

    class FakeRunner:
        def __init__(self, **_kwargs):
            constructed.append(True)
            self.session_dir = session_root / "unexpected"
            self.session_id = "unexpected"
            self.run_id = "unexpected"

        def prepare_session(self):
            prepared.append(True)
            self.session_dir.mkdir(parents=True)

    monkeypatch.setattr(mm, "validate_runtime_paths", lambda **_kwargs: None)
    monkeypatch.setattr(mm, "MatrixRunner", FakeRunner)

    rc = mm.main([
        "--dry-run", "--protocol", str(protocol_path), "--roster", str(roster_path),
        "--session-root", str(session_root),
    ])

    assert rc == 2
    assert "FATAL:" in capsys.readouterr().err
    assert constructed == []
    assert prepared == []
    assert not session_root.exists()
    assert not (session_root / "unexpected" / "matrix.full.log").exists()


@pytest.mark.parametrize("field", [
    "evaluated_agent", "judge", "cpp_build", "archive", "python_runnable",
])
@pytest.mark.parametrize("malformed_sha", [
    "a" * 63, "a" * 65, "g" * 64, "A" * 64, " " + "a" * 64 + " ", None,
])
def test_main_rejects_malformed_control_sha_before_session_write(
    tmp_path, monkeypatch, capsys, field, malformed_sha
):
    mm = _import_cli_module(monkeypatch)
    protocol = _valid_protocol(mm)
    roster = _valid_roster(mm)
    if field == "evaluated_agent":
        protocol["frozen_identities"]["evaluated_agent"]["sha256"] = malformed_sha
    elif field == "judge":
        protocol["frozen_identities"]["judge"]["main_py_sha256"] = malformed_sha
    elif field == "cpp_build":
        protocol["frozen_identities"]["build_artifacts_win64_mingw"]["rank01"] = malformed_sha
    elif field == "archive":
        roster["strategies"][0]["archive_sha256"] = malformed_sha
    else:
        roster["strategies"][3]["runnable_sha256"] = malformed_sha

    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    session_root = tmp_path / "sessions"
    _write_controls(protocol_path, roster_path, protocol, roster)
    constructed = []
    prepared = []

    class FakeRunner:
        def __init__(self, **_kwargs):
            constructed.append(True)

        def prepare_session(self):
            prepared.append(True)

    monkeypatch.setattr(mm, "validate_runtime_paths", lambda **_kwargs: None)
    monkeypatch.setattr(mm, "MatrixRunner", FakeRunner)

    rc = mm.main([
        "--dry-run", "--protocol", str(protocol_path), "--roster", str(roster_path),
        "--session-root", str(session_root),
    ])

    assert rc == 2
    assert "FATAL: control schema" in capsys.readouterr().err
    assert constructed == []
    assert prepared == []
    assert not session_root.exists()


def test_control_text_sha_is_identical_for_lf_and_crlf(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "protocol_version": "test"\n}\n')
    crlf.write_bytes(b'{\r\n  "protocol_version": "test"\r\n}\r\n')
    assert mm.control_text_sha(lf) == mm.control_text_sha(crlf)


def test_windows_crlf_protocol_checkout_passes_expected_hash(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    protocol = tmp_path / "protocol.json"
    roster = tmp_path / "roster.json"
    protocol.write_bytes(json.dumps(_valid_protocol(mm), indent=2).replace("\n", "\r\n").encode("utf-8"))
    roster.write_text(json.dumps(_valid_roster(mm)), encoding="utf-8")
    mm.load_control_inputs(protocol, roster, mm.control_text_sha(protocol))


def test_tracked_control_files_have_no_machine_paths(monkeypatch):
    mm = _import_cli_module(monkeypatch)
    framework_marker = "AgentBench" + "Framework.framework"
    markers = ("C:" + "/Users/", "C:" + "\\\\Users\\\\", "/home/", "/Users/", framework_marker)
    for path in (mm.PROTOCOL, mm.ROSTER, mm.REPO / "vendor" / "miracle_local" / "run_match.py"):
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker not in text


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


def test_load_control_inputs_rejects_non_object_strategy(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    protocol_path.write_text("{}", encoding="utf-8")
    roster_path.write_text(
        json.dumps([{"rank": rank} for rank in range(1, 17)] + ["not-an-object"]),
        encoding="utf-8",
    )

    with pytest.raises(mm.PreflightError, match="roster JSON must be an object"):
        mm.load_control_inputs(protocol_path, roster_path)


def test_cli_module_import_does_not_require_external_env(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("AGENTBENCH_ROOT", raising=False)
    monkeypatch.delenv("MIRACLE_IFELSE_DIR", raising=False)
    sys.modules.pop("miracle_matrix", None)
    tools_dir = str(repo / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    mm = importlib.import_module("miracle_matrix")

    assert mm.PROTOCOL.name == "24_miracle_evaluation_protocol.v0.3.json"

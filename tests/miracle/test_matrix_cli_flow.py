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


def test_main_missing_explicit_asset_stops_before_runner_or_session(tmp_path, monkeypatch, capsys):
    mm = _import_cli_module(monkeypatch)
    invoked = []
    monkeypatch.setattr(mm, "MatrixRunner", lambda **_kwargs: invoked.append(True))
    rc = mm.main(["--dry-run", "--judge-dir", str(tmp_path / "missing-judge"),
                  "--session-root", str(tmp_path / "sessions")])
    assert rc == 2
    assert "judge directory missing" in capsys.readouterr().err
    assert invoked == []
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
    roots = {name: tmp_path / name for name in (
        "judge_dir", "ifelse_dir", "extracted_root", "archives_root",
        "precheck_root", "rank16_build_root",
    )}
    rc = mm._run_resume(FakeRunner(), sid, control_inputs=control_inputs, **roots)

    assert rc == 2
    assert observed["expected_control_inputs"] == {"protocol": "p", "roster": "r"}


def test_main_manifest_hashes_explicit_judge_and_ifelse_dirs(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    explicit_ifelse = tmp_path / "ifelse"; explicit_ifelse.mkdir()
    explicit_judge = tmp_path / "judge"; explicit_judge.mkdir()
    (explicit_ifelse / "main.py").write_text("ifelse", encoding="utf-8")
    (explicit_judge / "main.py").write_text("judge", encoding="utf-8")
    missing = tmp_path / "missing-global"
    monkeypatch.setattr(mm, "IFELSE", missing)
    monkeypatch.setattr(mm, "JUDGE", missing)
    monkeypatch.setattr(mm, "validate_runtime_paths", lambda **_: None)
    def verified_hashes(*_args, asset_digests, **_kwargs):
        asset_digests.update({
            "ifelse": mm.sha(explicit_ifelse / "main.py"),
            "judge": mm.sha(explicit_judge / "main.py"),
        })
        return []
    monkeypatch.setattr(mm, "verify_hashes", verified_hashes)
    seen = {}

    class FakeRunner:
        def __init__(self, **_kwargs):
            self.session_id = "fake"; self.run_id = "run"; self.session_dir = tmp_path / "session"
            self.timeout = 8.0; self.wrapper_timeout_s = 180.0
        def prepare_session(self): self.session_dir.mkdir()
        def record_manifest(self, **kwargs): seen.update(kwargs)
        def dry_run(self): return {"plan_count": 1, "plan": [{"game_id": "first"}]}

    monkeypatch.setattr(mm, "MatrixRunner", FakeRunner)
    assert mm.main(["--dry-run", "--ifelse-dir", str(explicit_ifelse),
                    "--judge-dir", str(explicit_judge)]) == 0
    assert seen["ifelse_sha"] == mm.sha(explicit_ifelse / "main.py")
    assert seen["judge_sha"] == mm.sha(explicit_judge / "main.py")


def _make_main_assets(tmp_path, mm):
    roots = {name: tmp_path / name for name in (
        "judge", "ifelse", "extracted", "archives", "precheck", "rank16",
    )}
    for root in roots.values():
        root.mkdir()
    (roots["judge"] / "main.py").write_text("judge\n", encoding="utf-8")
    (roots["ifelse"] / "main.py").write_text("ifelse\n", encoding="utf-8")
    strategies = []
    builds = {}
    archives = {}
    runnables = {}
    for rank in range(1, 17):
        archive = roots["archives"] / f"rank{rank:02d}__fixture.zip"
        archive.write_bytes(f"archive-{rank}".encode("ascii"))
        archives[rank] = archive
        strategy = {"rank": rank, "archive_sha256": mm.sha(archive)}
        if rank in mm.PYTHON_RANKS:
            directory = roots["extracted"] / f"rank{rank:02d}__fixture"
            directory.mkdir()
            runnable = directory / "main.py"
            runnable.write_text(f"python-{rank}\n", encoding="utf-8")
            runnables[rank] = runnable
            strategy.update({"entry": "main.py", "runnable_sha256": mm.sha(runnable)})
        else:
            directory = (roots["rank16"] / "rank16_copy" if rank == 16
                         else roots["precheck"] / "strategies" / f"rank{rank:02d}")
            directory.mkdir(parents=True)
            executable = directory / "main.exe"
            executable.write_bytes(f"exe-{rank}".encode("ascii"))
            runnables[rank] = executable
            builds[f"rank{rank:02d}"] = mm.sha(executable)
        strategies.append(strategy)
    protocol = {
        "frozen_identities": {
            "evaluated_agent": {"sha256": mm.sha(roots["ifelse"] / "main.py")},
            "judge": {"main_py_sha256": mm.sha(roots["judge"] / "main.py")},
            "build_artifacts_win64_mingw": builds,
        }
    }
    protocol_path = tmp_path / "protocol.json"
    roster_path = tmp_path / "roster.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    roster_path.write_text(json.dumps({"strategies": strategies}), encoding="utf-8")
    return roots, archives, runnables, protocol_path, roster_path


@pytest.mark.parametrize("asset", ["judge", "ifelse", "archive", "python", "cpp"])
def test_main_unreadable_asset_stops_before_manifest_or_execution(tmp_path, monkeypatch, capsys, asset):
    mm = _import_cli_module(monkeypatch)
    roots, archives, runnables, protocol_path, roster_path = _make_main_assets(tmp_path, mm)
    unreadable = {
        "judge": roots["judge"] / "main.py",
        "ifelse": roots["ifelse"] / "main.py",
        "archive": archives[1],
        "python": runnables[4],
        "cpp": runnables[1],
    }[asset]
    original_sha = mm.sha

    def injected_sha(path):
        if Path(path) == unreadable:
            raise OSError("injected unreadable asset")
        return original_sha(path)

    monkeypatch.setattr(mm, "sha", injected_sha)
    recorded = []
    dry_runs = []
    executes = []
    session_root = tmp_path / "sessions"

    class FakeRunner:
        def __init__(self, **_kwargs):
            self.session_dir = session_root / "fake"
            self.session_id = "fake"
            self.run_id = "run"

        def prepare_session(self):
            self.session_dir.mkdir(parents=True)

        def record_manifest(self, **_kwargs):
            recorded.append(True)

        def dry_run(self):
            dry_runs.append(True)

        def execute(self):
            executes.append(True)

    monkeypatch.setattr(mm, "MatrixRunner", FakeRunner)
    rc = mm.main([
        "--dry-run", "--protocol", str(protocol_path), "--roster", str(roster_path),
        "--session-root", str(session_root), "--judge-dir", str(roots["judge"]),
        "--ifelse-dir", str(roots["ifelse"]), "--extracted-root", str(roots["extracted"]),
        "--archives-root", str(roots["archives"]), "--precheck-root", str(roots["precheck"]),
        "--rank16-build-root", str(roots["rank16"]),
    ])

    assert rc == 2
    assert "FATAL: hash mismatches" in capsys.readouterr().out
    assert recorded == []
    assert dry_runs == []
    assert executes == []


def test_resume_current_asset_mismatch_stops_before_log_or_resume(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    sid = "sid"; session = tmp_path / sid; session.mkdir()
    monkeypatch.setattr(mm, "verify_session_for_resume", lambda *_a, **_k: (True, []))
    roots = {name: tmp_path / name for name in ("judge", "ifelse", "extracted", "archives", "precheck", "rank16")}
    seen = {}
    def mismatch(*_args, **kwargs):
        seen.update(kwargs); return ["judge sha mismatch"]
    monkeypatch.setattr(mm, "verify_hashes", mismatch)
    class FakeRunner:
        session_dir = session
        def resume(self, _sid): raise AssertionError("resume must not run")
    control = mm.ControlInputs(
        json.loads(mm.PROTOCOL.read_text(encoding="utf-8")),
        json.loads(mm.ROSTER.read_text(encoding="utf-8")),
        {"protocol": "p", "roster": "r"},
    )
    before = {p.name: p.read_bytes() for p in session.iterdir()}
    rc = mm._run_resume(FakeRunner(), sid, control_inputs=control, session_root=tmp_path,
                        judge_dir=roots["judge"], ifelse_dir=roots["ifelse"],
                        extracted_root=roots["extracted"], archives_root=roots["archives"],
                        precheck_root=roots["precheck"], rank16_build_root=roots["rank16"])
    assert rc == 2
    assert seen["judge_root"] == roots["judge"]
    assert not (session / "matrix.full.log").exists()
    assert before == {p.name: p.read_bytes() for p in session.iterdir()}

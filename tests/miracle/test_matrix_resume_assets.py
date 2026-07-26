from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


PYTHON_RANKS = {4, 5, 7}


def _import_cli_module(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    tools_dir = str(repo / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    monkeypatch.setenv("AGENTBENCH_ROOT", str(repo))
    monkeypatch.setenv("MIRACLE_IFELSE_DIR", str(repo))
    sys.modules.pop("miracle_matrix", None)
    return importlib.import_module("miracle_matrix")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _make_assets(tmp_path: Path, mm):
    roots = {name: tmp_path / name for name in (
        "judge", "ifelse", "extracted", "archives", "precheck", "rank16",
    )}
    for root in roots.values():
        root.mkdir()
    (roots["judge"] / "main.py").write_text("judge\n", encoding="utf-8")
    (roots["ifelse"] / "main.py").write_text("ifelse\n", encoding="utf-8")

    strategies = []
    build_hashes = {}
    archive_paths = {}
    executable_paths = {}
    for rank in range(1, 17):
        archive = roots["archives"] / f"rank{rank:02d}__fixture.zip"
        archive.write_bytes(f"archive-{rank}".encode("ascii"))
        archive_paths[rank] = archive
        strategy = {"rank": rank, "archive_sha256": mm.sha(archive)}
        if rank in PYTHON_RANKS:
            directory = roots["extracted"] / f"rank{rank:02d}__fixture"
            directory.mkdir()
            entry = directory / "main.py"
            entry.write_text(f"python-{rank}\n", encoding="utf-8")
            strategy.update({"entry": "main.py", "runnable_sha256": mm.sha(entry)})
            executable_paths[rank] = entry
        else:
            if rank == 16:
                directory = roots["rank16"] / "rank16_copy"
            else:
                directory = roots["precheck"] / "strategies" / f"rank{rank:02d}"
            directory.mkdir(parents=True)
            executable = directory / "main.exe"
            executable.write_bytes(f"exe-{rank}".encode("ascii"))
            executable_paths[rank] = executable
            build_hashes[f"rank{rank:02d}"] = mm.sha(executable)
        strategies.append(strategy)

    protocol = {
        "frozen_identities": {
            "build_artifacts_win64_mingw": build_hashes,
            "evaluated_agent": {"sha256": mm.sha(roots["ifelse"] / "main.py")},
            "judge": {"main_py_sha256": mm.sha(roots["judge"] / "main.py")},
        }
    }
    control = mm.ControlInputs(protocol, {"strategies": strategies}, {"protocol": "p", "roster": "r"})
    return roots, control, archive_paths, executable_paths


class _FakeRunner:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.resume_count = 0
        self.execute_count = 0

    def resume(self, _sid):
        self.resume_count += 1

    def execute(self):
        self.execute_count += 1
        return {"completed": True, "halted": False}

    def write_run_compatible_output(self):
        return self.session_dir

    def aggregate_from_events(self):
        return {"total_attempts": 0, "valid_games": 0, "invalid_games": 0, "win_rate": None}


def _run_with_assets(tmp_path, monkeypatch, mutate=None, missing_root=None):
    mm = _import_cli_module(monkeypatch)
    roots, control, archives, executables = _make_assets(tmp_path, mm)
    session = tmp_path / "session"; session.mkdir()
    (session / "manifest.json").write_text(json.dumps({"fixture": True}), encoding="utf-8")
    (session / "progress.json").write_text('{"attempts": {}}', encoding="utf-8")
    monkeypatch.setattr(mm, "verify_session_for_resume", lambda *_a, **_k: (True, []))
    if mutate is not None:
        mutate(roots, archives, executables)
    runner = _FakeRunner(session)
    before = _tree_bytes(session)
    kwargs = {
        "judge_dir": roots["judge"], "ifelse_dir": roots["ifelse"],
        "extracted_root": roots["extracted"], "archives_root": roots["archives"],
        "precheck_root": roots["precheck"], "rank16_build_root": roots["rank16"],
    }
    if missing_root is not None:
        kwargs[missing_root] = None
    rc = mm._run_resume(runner, "session", control_inputs=control, session_root=tmp_path, **kwargs)
    return rc, runner, session, before


@pytest.mark.parametrize("name,mutate", [
    ("judge", lambda roots, _archives, _executables: (roots["judge"] / "main.py").write_text("changed\n", encoding="utf-8")),
    ("ifelse", lambda roots, _archives, _executables: (roots["ifelse"] / "main.py").write_text("changed\n", encoding="utf-8")),
    ("python runnable", lambda _roots, _archives, executables: executables[4].write_text("changed\n", encoding="utf-8")),
    ("python archive", lambda _roots, archives, _executables: archives[4].write_bytes(b"changed")),
    ("c++ archive", lambda _roots, archives, _executables: archives[1].write_bytes(b"changed")),
    ("c++ executable", lambda _roots, _archives, executables: executables[1].write_bytes(b"changed")),
    ("c++ executable is directory", lambda _roots, _archives, executables: (executables[1].unlink(), executables[1].mkdir())),
    ("archive missing", lambda _roots, archives, _executables: archives[1].unlink()),
    ("archive duplicate", lambda roots, _archives, _executables: (roots["archives"] / "rank01__duplicate.zip").write_bytes(b"duplicate")),
])
def test_resume_rejects_changed_real_runtime_assets_before_session_write(tmp_path, monkeypatch, name, mutate):
    rc, runner, session, before = _run_with_assets(tmp_path, monkeypatch, mutate=mutate)

    assert rc == 2, name
    assert runner.resume_count == 0
    assert runner.execute_count == 0
    assert _tree_bytes(session) == before
    assert not (session / "matrix.full.log").exists()


@pytest.mark.parametrize("rank", range(1, 17))
def test_resume_verifies_the_unique_archive_for_every_rank(tmp_path, monkeypatch, rank):
    rc, runner, session, before = _run_with_assets(
        tmp_path, monkeypatch,
        mutate=lambda _roots, archives, _executables: archives[rank].write_bytes(b"changed"),
    )

    assert rc == 2
    assert runner.resume_count == 0
    assert runner.execute_count == 0
    assert _tree_bytes(session) == before
    assert not (session / "matrix.full.log").exists()


@pytest.mark.parametrize("missing_root", [
    "judge_dir", "ifelse_dir", "extracted_root", "archives_root", "precheck_root", "rank16_build_root",
])
def test_resume_rejects_any_missing_runtime_root_before_session_write(tmp_path, monkeypatch, capsys, missing_root):
    rc, runner, session, before = _run_with_assets(tmp_path, monkeypatch, missing_root=missing_root)

    assert rc == 2
    assert f"runtime root missing: {missing_root}" in capsys.readouterr().err
    assert runner.resume_count == 0
    assert runner.execute_count == 0
    assert _tree_bytes(session) == before
    assert not (session / "matrix.full.log").exists()


def test_resume_with_all_real_runtime_assets_calls_resume_once(tmp_path, monkeypatch):
    rc, runner, _session, _before = _run_with_assets(tmp_path, monkeypatch)

    assert rc == 0
    assert runner.resume_count == 1
    assert runner.execute_count == 1

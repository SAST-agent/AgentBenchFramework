from __future__ import annotations

import hashlib
import importlib
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_unique_dir_rejects_ambiguous_matches(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    (tmp_path / "rank04__a").mkdir()
    (tmp_path / "rank04__b").mkdir()

    with pytest.raises(RuntimeError, match="multiple opponent directories"):
        mm.resolve_unique_dir(tmp_path, "rank04__*")


def test_verify_python_hashes_detects_modified_entry(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    python_dir = tmp_path / "rank04__fixture"
    python_dir.mkdir()
    (python_dir / "main.py").write_text("modified\n", encoding="utf-8")
    strategy = {
        "rank": 4,
        "type": "python_script",
        "entry": "main.py",
        "runnable_sha256": "expected",
    }

    errors = mm.verify_python_strategy_hashes(strategy, tmp_path)

    assert any("rank04" in error and "runnable sha" in error for error in errors)


def test_verify_python_hashes_checks_entry(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    extracted = tmp_path / "extracted"
    archives = tmp_path / "archives"
    extracted.mkdir()
    archives.mkdir()
    python_dir = extracted / "rank04__fixture"
    python_dir.mkdir()
    entry = python_dir / "main.py"
    entry.write_text("print('ok')\n", encoding="utf-8")
    archive = archives / "rank04__fixture.zip"
    archive.write_bytes(b"archive-bytes")
    strategy = {
        "rank": 4,
        "type": "python_script",
        "entry": "main.py",
        "runnable_sha256": _sha(entry),
        "archive_sha256": _sha(archive),
    }

    errors = mm.verify_python_strategy_hashes(strategy, extracted)

    assert errors == []


def test_verify_archive_hash_rejects_duplicate_archives(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    extracted = tmp_path / "extracted"
    archives = tmp_path / "archives"
    extracted.mkdir()
    archives.mkdir()
    python_dir = extracted / "rank04__fixture"
    python_dir.mkdir()
    (python_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (archives / "rank04__a.zip").write_bytes(b"a")
    (archives / "rank04__b.zip").write_bytes(b"b")
    strategy = {
        "rank": 4,
        "type": "python_script",
        "entry": "main.py",
        "runnable_sha256": _sha(python_dir / "main.py"),
        "archive_sha256": _sha(archives / "rank04__a.zip"),
    }

    errors = mm.verify_archive_hash(strategy, archives)

    assert any("multiple archive files" in error for error in errors)


def test_verify_archive_hash_requires_archive_identity(tmp_path, monkeypatch):
    mm = _import_cli_module(monkeypatch)
    extracted = tmp_path / "extracted"
    archives = tmp_path / "archives"
    extracted.mkdir()
    archives.mkdir()
    python_dir = extracted / "rank04__fixture"
    python_dir.mkdir()
    entry = python_dir / "main.py"
    entry.write_text("print('ok')\n", encoding="utf-8")
    (archives / "rank04__fixture.zip").write_bytes(b"archive-bytes")
    strategy = {
        "rank": 4,
        "type": "python_script",
        "entry": "main.py",
        "runnable_sha256": _sha(entry),
    }

    errors = mm.verify_archive_hash(strategy, archives)

    assert any("archive sha missing" in error for error in errors)

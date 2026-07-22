"""Tests for the cross-platform AI entry resolver (阶段9A §2).

The executable is returned as an ABSOLUTE path (Windows CreateProcess does not
search the cwd= argument for a bare name), so Popen(cmd, cwd=ai_dir) works on
every platform.

Covers: Windows main.exe / POSIX main / main.py; explicit priority; paths with
spaces not split; main.exe+main coexistence (deterministic); missing entry;
resolver does not modify the dir; returned exe path is absolute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.entry import resolve_ai_command


def _set_platform(monkeypatch, name):
    monkeypatch.setattr("agentbench_frame.games.miracle.entry._IS_NT", name == "nt")


def _abs(tmp_path, name):
    return str(Path(tmp_path).resolve() / name)


def test_windows_recognises_main_exe(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    assert resolve_ai_command(tmp_path) == [_abs(tmp_path, "main.exe")]


def test_posix_recognises_main(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "posix")
    (tmp_path / "main").write_bytes(b"\x7fELF")
    assert resolve_ai_command(tmp_path) == [_abs(tmp_path, "main")]


def test_python_main_py(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.py").write_text("print('ai')")
    assert resolve_ai_command(tmp_path) == [sys.executable, _abs(tmp_path, "main.py")]


def test_python_main_py_posix(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "posix")
    (tmp_path / "main.py").write_text("print('ai')")
    assert resolve_ai_command(tmp_path) == [sys.executable, _abs(tmp_path, "main.py")]


def test_explicit_list_takes_priority(tmp_path):
    (tmp_path / "main.exe").write_bytes(b"MZ")
    assert resolve_ai_command(tmp_path, explicit=["my", "args"]) == ["my", "args"]


def test_explicit_str_with_spaces_not_split(tmp_path):
    explicit = r"C:\Program Files\Some Dir\main.exe"
    assert resolve_ai_command(tmp_path, explicit=explicit) == [explicit]
    assert len(resolve_ai_command(tmp_path, explicit=explicit)) == 1


def test_explicit_empty_falls_through_to_autodetect(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    assert resolve_ai_command(tmp_path, explicit="") == [_abs(tmp_path, "main.exe")]
    assert resolve_ai_command(tmp_path, explicit=None) == [_abs(tmp_path, "main.exe")]


def test_coexistence_windows_picks_main_exe(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    (tmp_path / "main").write_bytes(b"MZ")
    assert resolve_ai_command(tmp_path) == [_abs(tmp_path, "main.exe")]


def test_coexistence_posix_picks_main(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "posix")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    (tmp_path / "main").write_bytes(b"\x7fELF")
    assert resolve_ai_command(tmp_path) == [_abs(tmp_path, "main")]


def test_missing_entry_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        resolve_ai_command(tmp_path)
    assert str(Path(tmp_path).resolve()) in str(ei.value)


def test_resolver_does_not_modify_strategy_dir(tmp_path, monkeypatch):
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    resolve_ai_command(tmp_path)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert before == after


def test_returned_exe_path_is_absolute(tmp_path, monkeypatch):
    # Windows CreateProcess does not search cwd= for a bare name -> must be absolute
    _set_platform(monkeypatch, "nt")
    (tmp_path / "main.exe").write_bytes(b"MZ")
    cmd = resolve_ai_command(tmp_path)
    assert Path(cmd[0]).is_absolute()

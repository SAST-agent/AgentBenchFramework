import hashlib, json
from pathlib import Path
import pytest

def _winerror():
    e = PermissionError(5, "sharing violation")
    e.winerror = 5
    return e

def test_unique_temp_and_transient_windows_retry(tmp_path, monkeypatch):
    from agentbench_frame.games.miracle import atomicio as aio
    target = tmp_path / "progress.json"; calls=[]; real=aio.os.replace
    monkeypatch.setattr(aio, "_is_windows", lambda: True)
    def flaky(src, dst):
        calls.append(Path(src).name)
        if len(calls) < 3: raise _winerror()
        return real(src, dst)
    monkeypatch.setattr(aio.os, "replace", flaky)
    aio.atomic_write_json(target, {"ok": True})
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert calls[0] != "progress.json.tmp"
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}

def test_persistent_windows_permission_preserves_old_target(tmp_path, monkeypatch):
    from agentbench_frame.games.miracle import atomicio as aio
    target = tmp_path / "progress.json"; target.write_text('{"old":true}', encoding="utf-8")
    before=hashlib.sha256(target.read_bytes()).hexdigest(); monkeypatch.setattr(aio, "_is_windows", lambda: True)
    monkeypatch.setattr(aio.os, "replace", lambda *a: (_ for _ in ()).throw(_winerror()))
    with pytest.raises(PermissionError): aio.atomic_write_json(target, {"new": True})
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".progress.json.*.tmp"))

def test_progress_uses_shared_unique_writer(tmp_path):
    from agentbench_frame.games.miracle.matrix import write_progress_atomic
    a=tmp_path/"a.json"; b=tmp_path/"b.json"
    write_progress_atomic(a,{"attempts":{"a":1}}); write_progress_atomic(b,{"attempts":{"b":2}})
    assert json.loads(a.read_text()) != json.loads(b.read_text())
    assert not (tmp_path/"a.json.tmp").exists() and not (tmp_path/"b.json.tmp").exists()

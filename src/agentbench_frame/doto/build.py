"""Build one immutable DOTO playerAI.cpp against the fixed SDK."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .assets import official_server_dir, sdk_dir, verify_assets


MAX_SOURCE_BYTES = 2 * 1024 * 1024


class DotoBuildError(ValueError):
    pass


@dataclass(frozen=True)
class BuildResult:
    source_hash: str
    executable: Path | None
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    seconds: float

    def to_json(self) -> dict:
        result = asdict(self)
        result["executable"] = str(self.executable) if self.executable else None
        result["command"] = list(self.command)
        return result


def _validate_source(path: Path) -> bytes:
    data = path.read_bytes()
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise DotoBuildError("playerAI.cpp must be nonempty and at most 2 MiB")
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DotoBuildError("playerAI.cpp must be UTF-8") from exc
    if "void playerAI(" not in source:
        raise DotoBuildError("playerAI.cpp must define void playerAI()")
    return data


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_candidate(
    player_ai: Path,
    output_dir: Path,
    compiler: str = "g++",
) -> BuildResult:
    source = _validate_source(Path(player_ai))
    verify_assets()
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    started = time.monotonic()
    command = ("make", f"CXX={compiler}")
    try:
        shutil.copytree(sdk_dir(), temporary, dirs_exist_ok=True)
        shutil.copytree(official_server_dir() / "Maps", temporary / "Maps")
        (temporary / "playerAI.cpp").write_bytes(source)
        completed = subprocess.run(
            command,
            cwd=temporary,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        executable = temporary / "main.out"
        if completed.returncode != 0 or not executable.is_file():
            executable.unlink(missing_ok=True)
            executable_path = None
        else:
            executable_path = output_dir / "main.out"
        result = BuildResult(
            source_hash=hashlib.sha256(source).hexdigest(),
            executable=executable_path,
            command=command,
            exit_code=completed.returncode if executable.is_file() else (completed.returncode or 1),
            stdout=completed.stdout,
            stderr=completed.stderr,
            seconds=time.monotonic() - started,
        )
        _write_json(temporary / "build.json", result.to_json())
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

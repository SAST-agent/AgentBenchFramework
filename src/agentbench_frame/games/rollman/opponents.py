"""Prepare opaque ranked Ghost archives for isolated execution."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentbench_frame.games.rollman.evaluator import Opponent
from agentbench_frame.games.rollman.match import ProcessSpec


DEFAULT_PROFILES = Path(__file__).with_name("assets") / "opponent_profiles.json"


class OpponentBuildError(RuntimeError):
    pass


def load_profiles(path: str | Path = DEFAULT_PROFILES) -> dict[str, dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OpponentBuildError("opponent profiles must be an object")
    return {str(key): dict(profile) for key, profile in value.items()}


def _archive_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or stat.S_ISLNK(mode)
            ):
                raise OpponentBuildError(
                    f"unsafe archive member in {archive.name}: {member.filename}"
                )
            resolved = (destination / relative).resolve()
            if destination.resolve() not in (resolved, *resolved.parents):
                raise OpponentBuildError(
                    f"archive member escapes build root: {member.filename}"
                )
        package.extractall(destination)


def _run_build(argv: tuple[str, ...], *, cwd: Path, label: str) -> None:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout)[-4000:]
        raise OpponentBuildError(f"{label} failed: {diagnostic}")


def _tool_version(executable: str) -> str | None:
    try:
        completed = subprocess.run(
            (executable, "--version"),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = completed.stdout or completed.stderr
    return text.splitlines()[0] if text else None


def prepare_opponent(
    opponent: Opponent,
    *,
    build_root: str | Path,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    python_executable: str = sys.executable,
    cpp_compiler: str = "c++",
    cargo_executable: str = "cargo",
) -> Opponent:
    profile_map = profiles if profiles is not None else load_profiles()
    if opponent.opponent_id not in profile_map:
        raise OpponentBuildError(f"missing profile for {opponent.opponent_id}")
    profile = profile_map[opponent.opponent_id]
    content_hash = _archive_hash(opponent.archive)
    extracted = (
        Path(build_root).resolve()
        / opponent.opponent_id
        / content_hash
        / "source"
    )
    marker = extracted.parent / "extracted.ok"
    if not marker.is_file():
        _safe_extract(opponent.archive, extracted)
        marker.write_text(content_hash + "\n", encoding="utf-8")

    entrypoint = (extracted / str(profile["entrypoint"])).resolve()
    if extracted not in (entrypoint, *entrypoint.parents) or not entrypoint.is_file():
        raise OpponentBuildError(
            f"{opponent.opponent_id} entrypoint is missing: {entrypoint}"
        )
    kind = str(profile["kind"])
    build_command: tuple[str, ...] | None = None
    if kind == "python":
        process = ProcessSpec(
            argv=(python_executable, str(entrypoint)),
            cwd=entrypoint.parent,
        )
    elif kind == "cpp":
        binary = extracted.parent / "bin" / opponent.opponent_id
        if not binary.is_file():
            binary.parent.mkdir(parents=True, exist_ok=True)
            compiler_flags = tuple(
                str(value) for value in profile.get("compiler_flags", ())
            )
            build_command = (
                cpp_compiler,
                "-std=c++17",
                "-O2",
                *compiler_flags,
                str(entrypoint),
                "-o",
                str(binary),
            )
            _run_build(
                build_command,
                cwd=entrypoint.parent,
                label=opponent.opponent_id,
            )
        process = ProcessSpec(argv=(str(binary),), cwd=entrypoint.parent)
    elif kind == "cargo":
        binary_name = str(profile.get("binary", "main"))
        binary = extracted / "target" / "release" / binary_name
        if not binary.is_file():
            build_command = (
                cargo_executable,
                "build",
                "--release",
                "--bin",
                binary_name,
                "--manifest-path",
                str(entrypoint),
            )
            _run_build(
                build_command,
                cwd=entrypoint.parent,
                label=opponent.opponent_id,
            )
        process = ProcessSpec(argv=(str(binary),), cwd=entrypoint.parent)
    else:
        raise OpponentBuildError(
            f"unsupported opponent kind for {opponent.opponent_id}: {kind}"
        )
    metadata = {
        "schema_version": "1.0",
        "opponent": opponent.opponent_id,
        "archive": str(opponent.archive),
        "archive_sha256": content_hash,
        "profile": dict(profile),
        "process": list(process.argv),
        "build_command": None if build_command is None else list(build_command),
        "python_version": _tool_version(python_executable),
        "compiler_version": (
            _tool_version(cpp_compiler) if kind == "cpp" else None
        ),
        "cargo_version": (
            _tool_version(cargo_executable) if kind == "cargo" else None
        ),
    }
    (extracted.parent / "build.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return Opponent(
        opponent_id=opponent.opponent_id,
        rank=opponent.rank,
        archive=opponent.archive,
        process=process,
        username=opponent.username,
        language=opponent.language,
        version=opponent.version,
    )


def prepare_human_pool(
    opponents: Iterable[Opponent],
    *,
    build_root: str | Path,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Opponent, ...]:
    return tuple(
        prepare_opponent(
            opponent,
            build_root=build_root,
            profiles=profiles,
        )
        for opponent in sorted(opponents, key=lambda item: item.rank)
    )

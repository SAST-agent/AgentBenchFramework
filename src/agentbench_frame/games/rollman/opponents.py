"""Prepare opaque ranked Ghost archives for isolated execution."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentbench_frame.games.rollman.evaluator import Opponent
from agentbench_frame.games.rollman.match import (
    ProcessSpec,
    _descendant_rss,
    _start,
    _stop,
)


DEFAULT_PROFILES = Path(__file__).with_name("assets") / "opponent_profiles.json"
_BUILD_ENV_NAMES = (
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    "CARGO_HOME", "RUSTUP_HOME", "SDKROOT",
)


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


def _run_build(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    label: str,
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float = 300,
    memory_limit_mb: int = 4096,
    process_limit: int = 256,
    output_limit_bytes: int = 4 * 1024 * 1024,
    disk_limit_bytes: int = 8 * 1024 * 1024 * 1024,
) -> None:
    if timeout_s <= 0:
        raise ValueError("build timeout_s must be positive")
    if memory_limit_mb <= 0:
        raise ValueError("build memory_limit_mb must be positive")
    if process_limit <= 0:
        raise ValueError("build process_limit must be positive")
    if output_limit_bytes <= 0:
        raise ValueError("build output_limit_bytes must be positive")
    if disk_limit_bytes <= 0:
        raise ValueError("build disk_limit_bytes must be positive")
    environment = {
        name: os.environ[name] for name in _BUILD_ENV_NAMES if name in os.environ
    }
    environment.update({str(k): str(v) for k, v in (extra_env or {}).items()})
    environment["CARGO_NET_OFFLINE"] = "true"
    command = argv
    scratch: Path | None = None
    write_roots = {cwd.resolve()}
    if sys.platform == "darwin":
        scratch = Path(
            tempfile.mkdtemp(prefix=".agentbench-build-", dir=cwd)
        ).resolve()
        environment["TMPDIR"] = str(scratch)
        command_argv = argv
        direct_clang = Path(
            "/Library/Developer/CommandLineTools/usr/bin/clang++"
        )
        if Path(argv[0]).name in {"c++", "clang++"} and direct_clang.is_file():
            sdk_root = Path(
                "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
            )
            command_argv = (
                str(direct_clang),
                "-isysroot",
                str(sdk_root),
                *argv[1:],
            )
            environment["SDKROOT"] = str(sdk_root)
        executable = shutil.which(command_argv[0], path=environment.get("PATH"))
        read_roots = {
            Path("/System"), Path("/usr"), Path("/bin"), Path("/sbin"),
            Path("/Library"), Path("/opt/homebrew"), Path("/private/etc"),
            Path("/dev"), cwd,
        }
        write_roots.add(scratch)
        for index, value in enumerate(command_argv[:-1]):
            if value == "-o":
                write_roots.add(Path(command_argv[index + 1]).resolve().parent)
        if executable:
            invoked_tool = Path(command_argv[0]).absolute()
            read_roots.add(
                invoked_tool.parent.parent
                if invoked_tool.parent.name == "bin"
                else invoked_tool.parent
            )
            tool = Path(executable).resolve()
            read_roots.add(tool.parent.parent if tool.parent.name == "bin" else tool.parent)
        for name in ("CARGO_HOME", "RUSTUP_HOME"):
            value = environment.get(name)
            if value:
                read_roots.add(Path(value).resolve())
        default_cargo = Path(environment.get("HOME", "")) / ".cargo"
        if default_cargo.is_dir():
            read_roots.add(default_cargo.resolve())
        for relative in ("registry", "git", ".package-cache"):
            dependency_cache = default_cargo / relative
            if dependency_cache.exists():
                read_roots.add(dependency_cache.resolve())
        clauses = " ".join(
            f"(subpath {json.dumps(str(root))})"
            for root in sorted(read_roots, key=lambda item: str(item))
            if root.exists()
        )
        write_clauses = " ".join(
            f"(subpath {json.dumps(str(root))})"
            for root in sorted(write_roots, key=lambda item: str(item))
            if root.exists()
        )
        denied_cargo_secrets = "".join(
            f"(deny file-read* (literal {json.dumps(str(default_cargo / name))}))"
            for name in (
                "credentials", "credentials.toml", "config", "config.toml",
            )
        )
        profile = (
            '(version 1)(import "system.sb")(deny default)'
            "(allow process*)(allow signal (target self))"
            "(allow process-info*)(allow sysctl-read)(allow mach-lookup)"
            "(allow ipc-posix*)(allow file-ioctl)"
            '(allow file-read-metadata (subpath "/"))'
            f"(allow file-read* {clauses})"
            f"(allow file* {write_clauses})"
            f"{denied_cargo_secrets}"
        )
        command = ("/usr/bin/sandbox-exec", "-p", profile, *command_argv)
    process: subprocess.Popen[bytes] | None = None
    violation: str | None = None
    output_overflow = threading.Event()
    output_lock = threading.Lock()
    output_total = 0
    stdout_tail = bytearray()
    stderr_tail = bytearray()

    def drain(stream: Any, tail: bytearray) -> None:
        nonlocal output_total
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            with output_lock:
                output_total += len(chunk)
                if output_total > output_limit_bytes:
                    output_overflow.set()
            tail.extend(chunk)
            if len(tail) > 4000:
                del tail[:-4000]

    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise OpponentBuildError(f"{label} failed to start: {exc}") from exc
        if process.stdout is None or process.stderr is None:
            raise OpponentBuildError(f"{label} failed to open output pipes")
        readers = (
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout_tail),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, stderr_tail),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout_s
        next_disk_check = 0.0
        limit_kib = memory_limit_mb * 1024
        while process.poll() is None:
            descendants = _descendant_rss(process.pid)
            if descendants is None:
                violation = "process-tree accounting is unavailable"
                break
            if sum(descendants.values()) > limit_kib:
                violation = f"exceeded {memory_limit_mb} MiB build memory limit"
                break
            if len(descendants) > process_limit:
                violation = f"exceeded {process_limit} build process limit"
                break
            if output_overflow.is_set():
                violation = f"exceeded {output_limit_bytes} byte build output limit"
                break
            monotonic_now = time.monotonic()
            if monotonic_now >= next_disk_check:
                if _roots_size(write_roots) > disk_limit_bytes:
                    violation = f"exceeded {disk_limit_bytes} byte build disk limit"
                    break
                next_disk_check = monotonic_now + 0.25
            if monotonic_now >= deadline:
                violation = f"exceeded {timeout_s:g}s build timeout"
                break
            time.sleep(0.02)

        if violation is not None:
            _terminate_build_tree(process)
        else:
            process.wait()
            _terminate_build_tree(process, root_finished=True)
        for reader in readers:
            reader.join(timeout=1)
        if violation is None and output_overflow.is_set():
            violation = f"exceeded {output_limit_bytes} byte build output limit"
        if violation is None and _roots_size(write_roots) > disk_limit_bytes:
            violation = f"exceeded {disk_limit_bytes} byte build disk limit"
    finally:
        if process is not None and process.poll() is None:
            _terminate_build_tree(process)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    stdout_text = stdout_tail.decode("utf-8", errors="replace")
    stderr_text = stderr_tail.decode("utf-8", errors="replace")
    if violation is not None:
        diagnostic = (stderr_text or stdout_text)[-4000:]
        raise OpponentBuildError(
            f"{label} failed: {violation}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    if process is None or process.returncode != 0:
        diagnostic = (stderr_text or stdout_text)[-4000:]
        raise OpponentBuildError(f"{label} failed: {diagnostic}")


def _terminate_build_tree(
    process: subprocess.Popen[bytes],
    *,
    root_finished: bool = False,
) -> None:
    current = _descendant_rss(process.pid) or {}
    if not root_finished:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in sorted(current, reverse=True):
        if root_finished and pid == process.pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if not root_finished:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    for pid in sorted(current, reverse=True):
        if pid == process.pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _roots_size(roots: Iterable[Path]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for root in roots:
        for path in (root, *root.rglob("*")):
            try:
                stat_result = path.stat()
            except (FileNotFoundError, PermissionError):
                continue
            identity = (stat_result.st_dev, stat_result.st_ino)
            if identity in seen or not stat.S_ISREG(stat_result.st_mode):
                continue
            seen.add(identity)
            total += stat_result.st_size
    return total


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
    python_executable: str = sys.executable,
    cpp_compiler: str = "c++",
    cargo_executable: str = "cargo",
) -> Opponent:
    profile_map = load_profiles()
    if opponent.opponent_id not in profile_map:
        raise OpponentBuildError(f"missing profile for {opponent.opponent_id}")
    profile = profile_map[opponent.opponent_id]
    content_hash = _archive_hash(opponent.archive)
    expected_hash = profile.get("archive_sha256")
    if not isinstance(expected_hash, str) or expected_hash != content_hash:
        raise OpponentBuildError(
            f"{opponent.opponent_id} archive is not the frozen reviewed artifact"
        )
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
            untrusted=True,
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
        process = ProcessSpec(
            argv=(str(binary),),
            cwd=entrypoint.parent,
            untrusted=True,
        )
    elif kind == "cargo":
        binary_name = str(profile.get("binary", "main"))
        binary = extracted / "target" / "release" / binary_name
        if not binary.is_file():
            build_env = {
                str(key): str(value)
                for key, value in profile.get("build_env", {}).items()
            }
            for package, version in profile.get("cargo_pins", {}).items():
                _run_build(
                    (
                        cargo_executable,
                        "update",
                        "-p",
                        str(package),
                        "--precise",
                        str(version),
                        "--manifest-path",
                        str(entrypoint),
                    ),
                    cwd=entrypoint.parent,
                    label=f"{opponent.opponent_id} dependency pin",
                )
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
                extra_env=build_env,
            )
        process = ProcessSpec(
            argv=(str(binary),),
            cwd=entrypoint.parent,
            untrusted=True,
        )
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
) -> tuple[Opponent, ...]:
    return tuple(
        prepare_opponent(
            opponent,
            build_root=build_root,
        )
        for opponent in sorted(opponents, key=lambda item: item.rank)
    )


def verify_opponent_start(
    opponent: Opponent,
    *,
    startup_timeout_s: float = 0.5,
) -> None:
    """Require the prepared process to initialize and wait for protocol input."""

    if opponent.process is None:
        raise OpponentBuildError(f"{opponent.opponent_id} has no process")
    process = _start(opponent.process, opponent.opponent_id)
    try:
        try:
            return_code = process.wait(timeout=startup_timeout_s)
        except subprocess.TimeoutExpired:
            return
        diagnostic = (
            process.stderr.read().decode("utf-8", errors="replace")[-2000:]
            if process.stderr is not None
            else ""
        )
        raise OpponentBuildError(
            f"{opponent.opponent_id} exited during startup "
            f"with code {return_code}: {diagnostic}"
        )
    finally:
        _stop(process)

"""Resolve and probe immutable Generals policy snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from agentbench_frame.tracking.snapshot import (
    LocalWorkspaceSnapshotter,
    WorkspaceManifest,
)

from .models import HistoricalPolicyConfig


class HistoricalPolicyError(ValueError):
    """Historical source provenance does not match its frozen contract."""


@dataclass(frozen=True)
class HistoricalPolicySource:
    version: str
    run_id: str
    content_hash: str
    source: Path
    manifest: WorkspaceManifest


@dataclass(frozen=True)
class PolicyProbeResult:
    version: str
    status: str
    deterministic: bool
    raw_actions: tuple[tuple[tuple[int, ...], ...], ...]
    elapsed_time_s: float
    stdout: str
    stderr: str
    error: str | None = None


def _read_manifest(path: Path) -> WorkspaceManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        files = {
            str(relative): str(digest)
            for relative, digest in raw["files"].items()
        }
        return WorkspaceManifest(
            content_hash=str(raw["content_hash"]),
            files=files,
            changed_files=[
                str(item) for item in raw.get("changed_files", [])
            ],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise HistoricalPolicyError(
            f"cannot read historical policy manifest: {exc}"
        ) from exc


def resolve_historical_policies(
    history: Iterable[HistoricalPolicyConfig],
    data_dir: Path,
    destination_root: Path,
) -> tuple[HistoricalPolicySource, ...]:
    """Verify each historical tree, copy exact files, then verify again."""

    data_dir = Path(data_dir).resolve()
    destination_root = Path(destination_root)
    snapshotter = LocalWorkspaceSnapshotter()
    resolved = []
    seen_versions: set[str] = set()
    for config in history:
        if config.version in seen_versions:
            raise HistoricalPolicyError(
                f"duplicate historical version: {config.version}"
            )
        seen_versions.add(config.version)
        version_root = (
            data_dir
            / "runs"
            / "28_generals"
            / "generals-hl"
            / config.run_id
            / "versions"
            / config.version
        )
        source = version_root / "source"
        manifest = _read_manifest(version_root / "manifest.json")
        if manifest.content_hash != config.content_hash:
            raise HistoricalPolicyError(
                f"{config.version} manifest content hash does not match "
                "the approved history"
            )
        try:
            actual = snapshotter.capture(source)
        except ValueError as exc:
            raise HistoricalPolicyError(str(exc)) from exc
        if (
            actual.content_hash != manifest.content_hash
            or actual.files != manifest.files
        ):
            raise HistoricalPolicyError(
                f"{config.version} source does not match its manifest"
            )

        target_root = destination_root / "policies" / config.version
        target = target_root / "source"
        if target.exists():
            copied = snapshotter.capture(target)
            if (
                copied.content_hash != manifest.content_hash
                or copied.files != manifest.files
            ):
                raise HistoricalPolicyError(
                    f"{config.version} copied source does not match manifest"
                )
        else:
            try:
                copied = snapshotter.materialize_manifest(
                    source,
                    target,
                    manifest,
                )
            except ValueError as exc:
                raise HistoricalPolicyError(str(exc)) from exc
        copied_manifest = target_root / "manifest.json"
        if copied_manifest.exists():
            preserved = _read_manifest(copied_manifest)
            if (
                preserved.content_hash != manifest.content_hash
                or preserved.files != manifest.files
            ):
                raise HistoricalPolicyError(
                    f"{config.version} copied manifest changed"
                )
        else:
            snapshotter.write_manifest(manifest, copied_manifest)
        if (
            copied.content_hash != config.content_hash
            or copied.files != manifest.files
        ):
            raise HistoricalPolicyError(
                f"{config.version} copied content hash verification failed"
            )
        resolved.append(
            HistoricalPolicySource(
                version=config.version,
                run_id=config.run_id,
                content_hash=config.content_hash,
                source=target.resolve(),
                manifest=manifest,
            )
        )
    return tuple(resolved)


def _decode_worker_output(
    output: str,
) -> tuple[str, tuple[tuple[int, ...], ...] | None, str | None]:
    lines = output.splitlines()
    if len(lines) != 1:
        return (
            "malformed_output",
            None,
            "worker stdout must contain exactly one JSON record",
        )
    try:
        raw = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return "malformed_output", None, f"invalid worker JSON: {exc}"
    if not isinstance(raw, Mapping):
        return "malformed_output", None, "worker response must be an object"
    if raw.get("status") != "complete":
        return "worker_error", None, str(raw.get("error", "worker failed"))
    commands = raw.get("commands")
    if (
        not isinstance(commands, list)
        or not commands
        or not all(
            isinstance(command, list)
            and command
            and all(type(item) is int for item in command)
            for command in commands
        )
    ):
        return (
            "malformed_output",
            None,
            "worker commands must be non-empty integer arrays",
        )
    return (
        "complete",
        tuple(tuple(command) for command in commands),
        None,
    )


def probe_historical_policy(
    policy: HistoricalPolicySource,
    measurement_state: Mapping[str, Any],
    *,
    engine_root: Path,
    sdk_root: Path,
    repeats: int = 2,
    timeout_s: float = 5.0,
) -> PolicyProbeResult:
    """Invoke a frozen ``main.agent`` in independent one-shot processes."""

    if type(repeats) is not int or repeats < 2:
        raise ValueError("repeats must be an integer of at least 2")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    request = {
        "source": str(policy.source),
        "engine_root": str(Path(engine_root).resolve()),
        "sdk_root": str(Path(sdk_root).resolve()),
        "measurement_state": measurement_state,
        "round": int(measurement_state["state"]["round"]),
        "seat": int(measurement_state["actor"]),
    }
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    started = time.monotonic()
    actions = []
    stdout_parts = []
    stderr_parts = []
    for _ in range(repeats):
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentbench_frame.generals.policy_probe_worker",
                ],
                input=json.dumps(
                    request,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            return PolicyProbeResult(
                version=policy.version,
                status="timeout",
                deterministic=False,
                raw_actions=tuple(actions),
                elapsed_time_s=time.monotonic() - started,
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                error=f"worker exceeded {timeout_s} seconds",
            )
        stdout_parts.append(completed.stdout)
        stderr_parts.append(completed.stderr)
        if completed.returncode != 0:
            return PolicyProbeResult(
                version=policy.version,
                status="worker_error",
                deterministic=False,
                raw_actions=tuple(actions),
                elapsed_time_s=time.monotonic() - started,
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                error=f"worker exited with {completed.returncode}",
            )
        status, commands, error = _decode_worker_output(completed.stdout)
        if status != "complete" or commands is None:
            return PolicyProbeResult(
                version=policy.version,
                status=status,
                deterministic=False,
                raw_actions=tuple(actions),
                elapsed_time_s=time.monotonic() - started,
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                error=error,
            )
        actions.append(commands)

    deterministic = all(item == actions[0] for item in actions[1:])
    return PolicyProbeResult(
        version=policy.version,
        status="complete" if deterministic else "nondeterministic",
        deterministic=deterministic,
        raw_actions=tuple(actions),
        elapsed_time_s=time.monotonic() - started,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
        error=None if deterministic else "worker outputs differ",
    )

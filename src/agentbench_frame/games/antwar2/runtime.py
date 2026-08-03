"""Derive, assemble, hash, and build frozen AntWar2 runtime resources."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


POLICY_FILES = ("ai.py", "common.py", "main.py", "protocol.py")


class AntWarRuntimeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == ".agentbench-package.json" or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise AntWarRuntimeError(f"runtime tree cannot contain symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class AntWarLayout:
    agentbench_root: Path
    build_root: Path
    backend_archive: Path
    human_pool_root: Path
    human_manifest: Path
    human_extracted_root: Path
    sdk_root: Path
    historical_versions_root: Path | None

    @classmethod
    def from_roots(
        cls,
        *,
        agentbench_root: str | Path,
        build_root: str | Path,
        positive_control_root: str | Path | None = None,
    ) -> "AntWarLayout":
        agentbench = Path(agentbench_root).resolve()
        build = Path(build_root).resolve()
        humans = agentbench / "top_algorithms/corpus/30_antwar2_ladder"
        control = (
            None
            if positive_control_root is None
            else Path(positive_control_root).resolve()
        )
        return cls(
            agentbench_root=agentbench,
            build_root=build,
            backend_archive=(
                agentbench
                / "backend_sources/corpus/30_antwar2/archives/gamecode_logic__141.zip"
            ),
            human_pool_root=humans,
            human_manifest=humans / "MANIFEST.tsv",
            human_extracted_root=humans / "extracted",
            sdk_root=(
                control / "04_versions/canonical_SDK"
                if control is not None
                else humans / "extracted/rank01__yyzsanyi__ai_storm__v32/SDK"
            ),
            historical_versions_root=(
                None if control is None else control / "04_versions"
            ),
        )

    def validate(self, *, require_positive_control: bool = False) -> None:
        for path in (
            self.backend_archive,
            self.human_manifest,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (self.human_extracted_root, self.sdk_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if require_positive_control and (
            self.historical_versions_root is None
            or not self.historical_versions_root.is_dir()
        ):
            raise FileNotFoundError("positive-control historical versions are required")


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise AntWarRuntimeError(f"source tree cannot contain symlink: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def assemble_candidate(
    *,
    destination: str | Path,
    policy_root: str | Path,
    sdk_root: str | Path,
    dependencies: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Materialize one source+SDK package with an explicit dependency closure."""

    target = Path(destination).resolve()
    policy = Path(policy_root).resolve()
    sdk = Path(sdk_root).resolve()
    if target.exists() and any(target.iterdir()):
        raise AntWarRuntimeError(f"candidate destination is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in POLICY_FILES:
        source = policy / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.is_symlink():
            raise AntWarRuntimeError(f"policy file cannot be a symlink: {source}")
        shutil.copy2(source, target / name)
    if not sdk.is_dir():
        raise FileNotFoundError(sdk)
    _copy_tree(sdk, target / "SDK")
    dependency_hashes: dict[str, str] = {}
    for name, root_value in sorted((dependencies or {}).items()):
        if not name or Path(name).name != name:
            raise ValueError(f"invalid dependency name: {name!r}")
        root = Path(root_value).resolve()
        source = root / "ai.py"
        if not source.is_file():
            raise FileNotFoundError(source)
        dependency_target = target.parent / name
        if dependency_target.exists() and any(dependency_target.iterdir()):
            raise AntWarRuntimeError(
                f"dependency destination is not empty: {dependency_target}"
            )
        dependency_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dependency_target / "ai.py")
        dependency_hashes[name] = _sha256(source)
    manifest = {
        "schema_version": "1.0",
        "policy_files": list(POLICY_FILES),
        "policy_source_hashes": {
            name: _sha256(policy / name) for name in POLICY_FILES
        },
        "sdk_tree_sha256": _tree_hash(target / "SDK"),
        "dependencies": dependency_hashes,
        "tree_sha256": _tree_hash(target),
    }
    (target / ".agentbench-package.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def assemble_bootstrap_candidate(
    *,
    destination: str | Path,
    support_root: str | Path,
    sdk_root: str | Path,
    policy_template: str | Path,
) -> dict[str, Any]:
    """Create a model-bootstrap package without copying a historical policy."""

    target = Path(destination).resolve()
    support = Path(support_root).resolve()
    sdk = Path(sdk_root).resolve()
    template = Path(policy_template).resolve()
    if target.exists() and any(target.iterdir()):
        raise AntWarRuntimeError(f"candidate destination is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    if not template.is_file():
        raise FileNotFoundError(template)
    shutil.copy2(template, target / "ai.py")
    for name in ("common.py", "main.py", "protocol.py"):
        source = support / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    if not sdk.is_dir():
        raise FileNotFoundError(sdk)
    _copy_tree(sdk, target / "SDK")
    manifest = {
        "schema_version": "1.0",
        "origin": "model_bootstrap",
        "policy_files": list(POLICY_FILES),
        "policy_source_hashes": {
            "ai.py": _sha256(template),
            **{
                name: _sha256(support / name)
                for name in ("common.py", "main.py", "protocol.py")
            },
        },
        "sdk_tree_sha256": _tree_hash(target / "SDK"),
        "dependencies": {},
        "tree_sha256": _tree_hash(target),
    }
    (target / ".agentbench-package.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_dependencies(
    *,
    candidate_root: str | Path,
    historical_versions_root: str | Path | None,
) -> None:
    """Restore hash-declared sibling delegates omitted from a version snapshot."""

    candidate = Path(candidate_root).resolve()
    manifest_path = candidate / ".agentbench-package.json"
    if not manifest_path.is_file():
        return
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies = value.get("dependencies", {})
    if not isinstance(dependencies, Mapping):
        raise AntWarRuntimeError("candidate dependency manifest is invalid")
    if not dependencies:
        return
    if historical_versions_root is None:
        raise AntWarRuntimeError("candidate dependencies require historical sources")
    historical = Path(historical_versions_root).resolve()
    for name, expected_hash in sorted(dependencies.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise AntWarRuntimeError(f"invalid candidate dependency name: {name!r}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise AntWarRuntimeError(f"invalid dependency hash for {name}")
        source = historical / name / "ai.py"
        if not source.is_file() or _sha256(source) != expected_hash:
            raise AntWarRuntimeError(f"dependency source hash mismatch: {name}")
        target = candidate.parent / name / "ai.py"
        if target.is_file():
            if _sha256(target) != expected_hash:
                raise AntWarRuntimeError(f"materialized dependency hash mismatch: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = Path(member.filename)
            mode = member.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise AntWarRuntimeError(
                    f"unsafe backend archive member: {member.filename}"
                )
            resolved = (destination / relative).resolve()
            if destination.resolve() not in (resolved, *resolved.parents):
                raise AntWarRuntimeError(
                    f"backend archive member escapes build root: {member.filename}"
                )
        package.extractall(destination)


def _enable_windows_binary_transport(main_source: Path) -> bool:
    """Preserve framed bytes on Windows pipes without changing game rules."""

    if sys.platform != "win32":
        return False
    text = main_source.read_text(encoding="utf-8")
    marker = "AGENTBENCH_BINARY_TRANSPORT"
    if marker in text:
        return True
    include_anchor = "#include <vector>"
    main_anchor = "int main(/*int argc, char *argv[]*/) {"
    if include_anchor not in text or main_anchor not in text:
        raise AntWarRuntimeError("cannot apply Windows binary transport patch")
    text = text.replace(
        include_anchor,
        include_anchor
        + "\n#ifdef _WIN32\n#include <fcntl.h>\n#include <io.h>\n#endif",
        1,
    ).replace(
        main_anchor,
        main_anchor
        + "\n#ifdef _WIN32 // AGENTBENCH_BINARY_TRANSPORT\n"
        + "    _setmode(_fileno(stdin), _O_BINARY);\n"
        + "    _setmode(_fileno(stdout), _O_BINARY);\n#endif",
        1,
    )
    main_source.write_text(text, encoding="utf-8")
    return True


def build_backend(
    *,
    archive: str | Path,
    build_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Compile a content-addressed official backend and return its manifest."""

    source_archive = Path(archive).resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    archive_hash = _sha256(source_archive)
    root = Path(build_root).resolve() / f"backend-{archive_hash[:16]}"
    game_root = root / "game"
    executable = game_root / "output" / (
        "main.exe" if sys.platform == "win32" else "main"
    )
    manifest_path = root / "build-manifest.json"
    if executable.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("archive_sha256") == archive_hash
            and manifest.get("executable_sha256") == _sha256(executable)
        ):
            return executable, manifest
    if root.exists() and any(root.iterdir()):
        raise AntWarRuntimeError(
            f"incomplete backend cache requires a clean target: {root}"
        )
    _safe_extract(source_archive, root)
    if not game_root.is_dir():
        raise AntWarRuntimeError("backend archive has no game directory")
    binary_transport_patch = _enable_windows_binary_transport(
        game_root / "src/main.cpp"
    )
    completed = subprocess.run(
        ("make", "-C", str(game_root), f"-j{max(1, min(os.cpu_count() or 1, 8))}"),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or not executable.is_file():
        diagnostic = (completed.stderr or completed.stdout)[-8000:]
        raise AntWarRuntimeError(f"backend build failed: {diagnostic}")
    compiler = subprocess.run(
        ("g++", "--version"),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    manifest = {
        "schema_version": "1.0",
        "archive": str(source_archive),
        "archive_sha256": archive_hash,
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "platform": platform.platform(),
        "python": sys.version,
        "compiler": (compiler.stdout or compiler.stderr).splitlines()[0],
        "windows_binary_transport_patch": binary_transport_patch,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return executable, manifest

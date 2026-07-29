"""Frozen Git-object source boundary for public AgentBench game logic.

AB-Ludi/1 measures an exact, versioned list of Git blobs. Selection policy is
still encoded here so a new allow-listed logic file fails closed instead of
silently changing the metric.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFileSpec:
    """One path and content digest in the frozen source manifest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class GameSourceSpec:
    """Authoritative public source boundary for one AgentBench game."""

    game_id: str
    title: str
    source_root: str
    files: tuple[SourceFileSpec, ...] = ()
    excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceModule:
    """One exact source blob with a canonical path relative to its game root."""

    path: str
    content: bytes


_MANIFEST_PATH = Path(__file__).with_name("agentbench_ludi_v1_manifest.json")
_MANIFEST_BYTES = _MANIFEST_PATH.read_bytes()
_MANIFEST = json.loads(_MANIFEST_BYTES.decode("utf-8"))
if _MANIFEST.get("schema_version") != "agentbench.ludi-source-manifest.v1":
    raise RuntimeError(f"unsupported AB-Ludi source manifest: {_MANIFEST_PATH}")

AGENTBENCH_SOURCE_COMMIT: str = _MANIFEST["source_commit"]
SOURCE_MANIFEST_SHA256 = hashlib.sha256(_MANIFEST_BYTES).hexdigest()

_GAME_EXCLUDED_PATHS: dict[str, tuple[str, ...]] = {
    "23_doto": ("GamePlayerInUnity.py", "server2.py", "server3.py"),
    "25_lostspace": (
        "src/MapGen/MapLexer.py",
        "src/MapGen/MapListener.py",
        "src/MapGen/MapParser.py",
    ),
    "28_generals": ("main_for_player_test.py",),
}

AGENTBENCH_GAME_SPECS: tuple[GameSourceSpec, ...] = tuple(
    GameSourceSpec(
        game_id=game["game_id"],
        title=game["title"],
        source_root=game["source_root"],
        files=tuple(
            SourceFileSpec(path=file["path"], sha256=file["sha256"])
            for file in game["files"]
        ),
        excluded_paths=_GAME_EXCLUDED_PATHS.get(game["game_id"], ()),
    )
    for game in _MANIFEST["games"]
)


_SOURCE_SUFFIXES = frozenset(
    {".py", ".c", ".cpp", ".h", ".hpp", ".json", ".g4", ".map"}
)
_EXCLUDED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        "bak",
        "backup",
        "data",
        "doc",
        "docs",
        "judge_dev_sample_ai",
        "jsoncpp",
        "lib",
        "output",
        "sample_ai",
        "slide",
        "test",
        "test_config",
        "tests",
    }
)
_EXCLUDED_FILENAMES = frozenset(
    {
        "MapLexer.py",
        "MapListener.py",
        "MapParser.py",
        "ai_demo.py",
        "json.hpp",
        "main_test.py",
        "main_with_debug.py",
        "old_main.py",
        "test.py",
        "upload.py",
    }
)


def _is_selected(relative_path: Path, spec: GameSourceSpec) -> bool:
    path_string = relative_path.as_posix()
    if path_string in spec.excluded_paths:
        return False
    if relative_path.name in _EXCLUDED_FILENAMES:
        return False
    if relative_path.suffix.lower() not in _SOURCE_SUFFIXES:
        return False
    return not any(
        part.casefold() in _EXCLUDED_DIRECTORIES
        for part in relative_path.parts[:-1]
    )


def _run_git(repo_root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot read AgentBench Git objects: {message}")
    return process.stdout


def git_head_commit(repo_root: str | Path) -> str:
    """Return the checkout's exact HEAD commit."""

    root = Path(repo_root)
    commit = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if len(commit) not in {40, 64}:
        raise ValueError(f"invalid AgentBench source commit: {commit!r}")
    return commit


def _parse_ls_tree(data: bytes) -> tuple[tuple[str, str, str, str], ...]:
    entries = []
    for record in data.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise ValueError("malformed Git ls-tree record")
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        entries.append((mode, object_type, object_id, path))
    return tuple(entries)


def _tree_files(
    repo_root: Path,
    source_commit: str,
    source_root: str,
) -> dict[str, tuple[str, str, str]]:
    prefix = f"{source_root}/"
    entries = _parse_ls_tree(
        _run_git(
            repo_root,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            source_commit,
            "--",
            source_root,
        )
    )
    result = {}
    for mode, object_type, object_id, path in entries:
        if not path.startswith(prefix):
            raise ValueError(f"Git tree path escaped source root: {path}")
        result[path[len(prefix) :]] = (mode, object_type, object_id)
    return result


def _format_manifest_delta(name: str, values: set[str]) -> str | None:
    if not values:
        return None
    return f"{name}={','.join(sorted(values))}"


def collect_game_sources(
    repo_root: str | Path,
    spec: GameSourceSpec,
    source_commit: str = AGENTBENCH_SOURCE_COMMIT,
) -> tuple[SourceModule, ...]:
    """Read one game's exact source modules from immutable Git blobs."""

    root = Path(repo_root)
    tree_files = _tree_files(root, source_commit, spec.source_root)
    selected = {
        path for path in tree_files if _is_selected(Path(path), spec)
    }
    expected = {file.path for file in spec.files}
    missing = expected - selected
    extra = selected - expected
    if missing or extra:
        details = [
            detail
            for detail in (
                _format_manifest_delta("missing", missing),
                _format_manifest_delta("extra", extra),
            )
            if detail is not None
        ]
        raise ValueError(
            f"{spec.game_id}: source manifest mismatch: {'; '.join(details)}"
        )
    if not expected:
        raise ValueError(f"{spec.game_id}: source manifest contains no logic files")

    modules = []
    for file in sorted(spec.files, key=lambda item: item.path):
        mode, object_type, object_id = tree_files[file.path]
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(
                f"{spec.game_id}: expected regular Git blob: {file.path} "
                f"(mode={mode}, type={object_type})"
            )
        content = _run_git(root, "cat-file", "blob", object_id)
        digest = hashlib.sha256(content).hexdigest()
        if digest != file.sha256:
            raise ValueError(
                f"{spec.game_id}: SHA-256 mismatch for {file.path}: "
                f"expected {file.sha256}, got {digest}"
            )
        modules.append(SourceModule(path=file.path, content=content))
    return tuple(modules)


def validate_agentbench_corpus(
    repo_root: str | Path,
    source_commit: str = AGENTBENCH_SOURCE_COMMIT,
    *,
    specs: tuple[GameSourceSpec, ...] = AGENTBENCH_GAME_SPECS,
) -> None:
    """Fail closed unless a Git commit has exactly the frozen public game set."""

    root = Path(repo_root)
    entries = _parse_ls_tree(
        _run_git(root, "ls-tree", "-z", f"{source_commit}:backend_sources/corpus")
    )
    actual = {
        path
        for _, object_type, _, path in entries
        if object_type == "tree"
    }
    expected = {spec.game_id for spec in specs}
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = [
            detail
            for detail in (
                _format_manifest_delta("missing", missing),
                _format_manifest_delta("extra", extra),
            )
            if detail is not None
        ]
        raise ValueError(f"AgentBench game set mismatch: {'; '.join(details)}")

    for spec in specs:
        collect_game_sources(root, spec, source_commit)

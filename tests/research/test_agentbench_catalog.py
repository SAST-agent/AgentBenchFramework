import hashlib
import subprocess
from pathlib import Path

import pytest

from agentbench_frame.research.agentbench_catalog import (
    AGENTBENCH_GAME_SPECS,
    AGENTBENCH_SOURCE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    GameSourceSpec,
    SourceFileSpec,
    collect_game_sources,
    git_head_commit,
    validate_agentbench_corpus,
)

EXPECTED_GAME_IDS = {
    "23_doto",
    "24_miracle",
    "25_aquawar",
    "25_lostspace",
    "26_snakego",
    "27_antwar",
    "28_generals",
    "29_rollman",
    "30_antwar2",
    "30_deepclue",
}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)


def _commit(repo: Path, message: str = "fixture") -> str:
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=AgentBench Test",
        "-c",
        "user.email=agentbench-test@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return git_head_commit(repo)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _custom_spec(
    game_id: str,
    files: dict[str, bytes],
    *,
    excluded_paths: tuple[str, ...] = (),
) -> GameSourceSpec:
    return GameSourceSpec(
        game_id=game_id,
        title=game_id,
        source_root=f"backend_sources/corpus/{game_id}/logic/gamecode_logic",
        files=tuple(
            SourceFileSpec(path=path, sha256=_sha(content))
            for path, content in sorted(files.items())
        ),
        excluded_paths=excluded_paths,
    )


def _write_files(repo: Path, spec: GameSourceSpec, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        path = repo / spec.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_catalog_freezes_every_public_game_and_exact_source_file():
    game_ids = [spec.game_id for spec in AGENTBENCH_GAME_SPECS]

    assert AGENTBENCH_SOURCE_COMMIT == "b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87"
    assert len(SOURCE_MANIFEST_SHA256) == 64
    assert len(game_ids) == 10
    assert len(set(game_ids)) == 10
    assert set(game_ids) == EXPECTED_GAME_IDS
    assert sum(len(spec.files) for spec in AGENTBENCH_GAME_SPECS) == 149
    assert all(len(file.sha256) == 64 for spec in AGENTBENCH_GAME_SPECS for file in spec.files)


def test_collect_game_sources_reads_pinned_git_blobs_not_worktree(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    committed = {"rules.py": b"RULE = 'committed'\n"}
    spec = _custom_spec("game", committed)
    _write_files(repo, spec, committed)
    commit = _commit(repo)

    root = repo / spec.source_root
    (root / "rules.py").write_bytes(b"RULE = 'dirty'\n")
    (root / "untracked.py").write_bytes(b"RULE = 'untracked'\n")

    modules = collect_game_sources(repo, spec, commit)

    assert [(module.path, module.content) for module in modules] == [
        ("rules.py", committed["rules.py"])
    ]


def test_collect_game_sources_enforces_exact_manifest_boundary(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    selected = {
        "engine.py": b"ENGINE = True\n",
        "native/rule.cpp": b"int rule = 1;\n",
        "rules/config.json": b'{"turns": 8}\n',
    }
    spec = _custom_spec("game", selected)
    _write_files(repo, spec, selected)
    commit = _commit(repo)

    modules = collect_game_sources(repo, spec, commit)
    assert [module.path for module in modules] == sorted(selected)

    extra = repo / spec.source_root / "new_rule.py"
    extra.write_text("NEW = True\n")
    changed_commit = _commit(repo, "add selected source")
    with pytest.raises(ValueError, match="source manifest mismatch.*extra=new_rule.py"):
        collect_game_sources(repo, spec, changed_commit)


def test_collect_game_sources_rejects_missing_file_or_hash_mismatch(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    original = {"rules.py": b"RULE = 1\n"}
    spec = _custom_spec("game", original)
    _write_files(repo, spec, original)
    commit = _commit(repo)

    (repo / spec.source_root / "rules.py").unlink()
    missing_commit = _commit(repo, "delete source")
    with pytest.raises(ValueError, match="source manifest mismatch.*missing=rules.py"):
        collect_game_sources(repo, spec, missing_commit)

    _write_files(repo, spec, {"rules.py": b"RULE = 2\n"})
    changed_commit = _commit(repo, "change source")
    with pytest.raises(ValueError, match="SHA-256 mismatch.*rules.py"):
        collect_game_sources(repo, spec, changed_commit)

    assert collect_game_sources(repo, spec, commit)[0].content == original["rules.py"]


def test_collect_game_sources_rejects_git_symlink(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    content = b"outside.py"
    spec = _custom_spec("game", {"rules.py": content})
    target = repo / spec.source_root / "rules.py"
    target.parent.mkdir(parents=True)
    target.symlink_to("outside.py")
    commit = _commit(repo)

    with pytest.raises(ValueError, match="regular Git blob"):
        collect_game_sources(repo, spec, commit)


def test_validate_agentbench_corpus_fails_closed_on_missing_or_extra_game(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    specs = tuple(
        _custom_spec(game_id, {"rules.py": f"GAME = {game_id!r}\n".encode()})
        for game_id in ("game_a", "game_b")
    )
    for spec in specs:
        _write_files(
            repo,
            spec,
            {"rules.py": f"GAME = {spec.game_id!r}\n".encode()},
        )
    valid_commit = _commit(repo)
    validate_agentbench_corpus(repo, valid_commit, specs=specs)

    missing_root = repo / specs[1].source_root
    (missing_root / "rules.py").unlink()
    missing_root.rmdir()
    missing_commit = _commit(repo, "remove game")
    with pytest.raises(ValueError, match="game set mismatch.*missing=game_b"):
        validate_agentbench_corpus(repo, missing_commit, specs=specs)

    _write_files(repo, specs[1], {"rules.py": b"GAME = 'game_b'\n"})
    extra = repo / "backend_sources/corpus/game_c/logic/gamecode_logic/rules.py"
    extra.parent.mkdir(parents=True)
    extra.write_text("GAME = 'game_c'\n")
    extra_commit = _commit(repo, "add extra game")
    with pytest.raises(ValueError, match="extra=game_c"):
        validate_agentbench_corpus(repo, extra_commit, specs=specs)

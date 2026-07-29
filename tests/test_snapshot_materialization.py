from pathlib import Path

import pytest

from agentbench_frame.tracking.snapshot import (
    LocalWorkspaceSnapshotter,
    WorkspaceManifest,
)


def _source_and_manifest(tmp_path: Path):
    source = tmp_path / "source"
    (source / "policy").mkdir(parents=True)
    (source / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "policy" / "rules.txt").write_text(
        "allowlisted\n",
        encoding="utf-8",
    )
    snapshotter = LocalWorkspaceSnapshotter()
    return source, snapshotter, snapshotter.capture(source)


def test_materialize_manifest_copies_only_allowlisted_files(tmp_path):
    source, snapshotter, manifest = _source_and_manifest(tmp_path)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "poison.pyc").write_bytes(b"poison")
    (source / ".venv").mkdir()
    (source / ".venv" / "poison.py").write_text(
        "raise RuntimeError\n",
        encoding="utf-8",
    )
    destination = tmp_path / "destination"

    actual = snapshotter.materialize_manifest(
        source,
        destination,
        manifest,
    )

    assert actual.content_hash == manifest.content_hash
    assert actual.files == manifest.files
    assert (destination / "main.py").is_file()
    assert not (destination / "__pycache__" / "poison.pyc").exists()
    assert not (destination / ".venv" / "poison.py").exists()


def test_materialize_manifest_preserves_only_existing_git_metadata(tmp_path):
    source, snapshotter, manifest = _source_and_manifest(tmp_path)
    destination = tmp_path / "workspace"
    (destination / ".git").mkdir(parents=True)
    marker = destination / ".git" / "HEAD"
    marker.write_text("ref: refs/heads/main\n", encoding="utf-8")

    snapshotter.materialize_manifest(source, destination, manifest)

    assert marker.read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert (destination / "policy" / "rules.txt").is_file()


@pytest.mark.parametrize("failure", ["missing", "hash", "symlink"])
def test_materialize_manifest_rejects_unverified_source_files(
    tmp_path,
    failure,
):
    source, snapshotter, manifest = _source_and_manifest(tmp_path)
    path = source / "main.py"
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        path.write_text("VALUE = 2\n", encoding="utf-8")
    else:
        external = tmp_path / "external.py"
        external.write_text("VALUE = 1\n", encoding="utf-8")
        path.unlink()
        path.symlink_to(external)

    with pytest.raises(ValueError, match=failure):
        snapshotter.materialize_manifest(
            source,
            tmp_path / "destination",
            manifest,
        )


@pytest.mark.parametrize(
    "relative",
    ["../escape.py", "/absolute.py", ".git/config", ".venv/poison.py"],
)
def test_materialize_manifest_rejects_unsafe_manifest_paths(
    tmp_path,
    relative,
):
    source, snapshotter, manifest = _source_and_manifest(tmp_path)
    unsafe = WorkspaceManifest(
        content_hash=manifest.content_hash,
        files={relative: "0" * 64},
    )

    with pytest.raises(ValueError, match="unsafe manifest path"):
        snapshotter.materialize_manifest(
            source,
            tmp_path / "destination",
            unsafe,
        )

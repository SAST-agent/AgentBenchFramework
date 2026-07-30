import zipfile
from pathlib import Path

import pytest

from agentbench_frame.games.rollman.evaluator import Opponent
from agentbench_frame.games.rollman.evaluator import load_human_pool
from agentbench_frame.games.rollman.opponents import (
    OpponentBuildError,
    prepare_opponent,
)


def test_python_opponent_is_prepared_in_content_addressed_directory(tmp_path):
    archive = tmp_path / "rank01.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("agent/main.py", "print('fixture')\n")
    opponent = Opponent("rank01", 1, archive)

    prepared = prepare_opponent(
        opponent,
        build_root=tmp_path / "build",
        profiles={
            "rank01": {"kind": "python", "entrypoint": "agent/main.py"}
        },
    )

    assert prepared.process is not None
    assert prepared.process.argv[1].endswith("source/agent/main.py")
    assert prepared.process.cwd == Path(prepared.process.argv[1]).parent


def test_archive_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "rank01.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape.py", "bad\n")
    opponent = Opponent("rank01", 1, archive)

    with pytest.raises(OpponentBuildError, match="unsafe archive"):
        prepare_opponent(
            opponent,
            build_root=tmp_path / "build",
            profiles={
                "rank01": {"kind": "python", "entrypoint": "agent/main.py"}
            },
        )


@pytest.mark.integration
def test_real_rank1_human_archive_compiles(tmp_path):
    manifest = Path(
        "/Users/qingle/Code/SAST/AgentBench/"
        "top_algorithms/corpus/29_rollman_ghost_final/MANIFEST.tsv"
    )
    if not manifest.is_file():
        pytest.skip("ranked Rollman pool is unavailable")

    prepared = prepare_opponent(
        load_human_pool(manifest)[0],
        build_root=tmp_path / "build",
    )

    assert prepared.process is not None
    assert Path(prepared.process.argv[0]).is_file()

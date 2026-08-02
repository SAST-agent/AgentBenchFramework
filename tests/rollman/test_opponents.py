import zipfile
import sys
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from agentbench_frame.games.rollman.evaluator import Opponent
from agentbench_frame.games.rollman.evaluator import load_human_pool
from agentbench_frame.games.rollman.opponents import (
    OpponentBuildError,
    _run_build,
    _safe_extract,
    load_profiles,
    prepare_opponent,
)


def test_default_profiles_pin_all_16_frozen_archive_hashes():
    profiles = load_profiles()

    assert set(profiles) == {f"rank{rank:02d}" for rank in range(1, 17)}
    assert all(
        len(profile["archive_sha256"]) == 64
        for profile in profiles.values()
    )


def test_archive_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "rank01.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape.py", "bad\n")
    with pytest.raises(OpponentBuildError, match="unsafe archive"):
        _safe_extract(archive, tmp_path / "build")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox integration")
def test_build_resource_limit_covers_descendant_processes(tmp_path):
    script = tmp_path / "build_parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        "\"import time; memory=bytearray(160*1024*1024); time.sleep(10)\""
        "])\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )

    with pytest.raises(OpponentBuildError, match="memory limit"):
        _run_build(
            (sys.executable, str(script)),
            cwd=tmp_path,
            label="memory-build",
            timeout_s=6,
            memory_limit_mb=100,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox integration")
def test_build_output_limit_is_enforced_without_buffering_all_output(tmp_path):
    script = tmp_path / "noisy_build.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('x' * 1_000_000)\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    with pytest.raises(OpponentBuildError, match="output limit"):
        _run_build(
            (sys.executable, str(script)),
            cwd=tmp_path,
            label="noisy-build",
            timeout_s=3,
            output_limit_bytes=128 * 1024,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox integration")
def test_build_directory_limit_is_enforced(tmp_path):
    script = tmp_path / "large_build.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('large.bin').write_bytes(b'x' * 2_000_000)\n",
        encoding="utf-8",
    )

    with pytest.raises(OpponentBuildError, match="disk limit"):
        _run_build(
            (sys.executable, str(script)),
            cwd=tmp_path,
            label="large-build",
            timeout_s=3,
            disk_limit_bytes=1_000_000,
        )


def test_unregistered_archive_hash_is_rejected(tmp_path):
    archive = tmp_path / "rank01.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("agent/main.py", "print('fixture')\n")

    with pytest.raises(OpponentBuildError, match="frozen reviewed artifact"):
        prepare_opponent(
            Opponent("rank01", 1, archive),
            build_root=tmp_path / "build",
        )


def test_python_opponent_runs_through_seeded_wrapper(tmp_path, monkeypatch):
    archive = tmp_path / "rank01.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("agent/main.py", "print('fixture')\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "agentbench_frame.games.rollman.opponents.load_profiles",
        lambda: {
            "rank01": {
                "archive_sha256": digest,
                "kind": "python",
                "entrypoint": "agent/main.py",
            }
        },
    )

    prepared = prepare_opponent(
        Opponent("rank01", 1, archive),
        build_root=tmp_path / "build",
    )

    assert prepared.process is not None
    assert Path(prepared.process.argv[1]).name == "seeded_agent_runner.py"
    assert prepared.process.argv[2] == "--entrypoint"
    assert Path(prepared.process.argv[3]).name == "main.py"


def test_seeded_agent_runner_repeats_python_and_numpy_randomness(tmp_path):
    from agentbench_frame.games.rollman import seeded_agent_runner

    script = tmp_path / "agent.py"
    script.write_text(
        "import random, numpy as np\n"
        "print(random.randint(0, 10**9), int(np.random.randint(0, 10**9)))\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["AGENTBENCH_ROLLMAN_SEED"] = "424242"
    command = (
        sys.executable,
        str(Path(seeded_agent_runner.__file__)),
        "--entrypoint",
        str(script),
    )

    first = subprocess.run(command, env=environment, capture_output=True, text=True)
    second = subprocess.run(command, env=environment, capture_output=True, text=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout


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


@pytest.mark.integration
def test_real_rank14_rust_archive_compiles_offline_in_build_sandbox(tmp_path):
    manifest = Path(
        "/Users/qingle/Code/SAST/AgentBench/"
        "top_algorithms/corpus/29_rollman_ghost_final/MANIFEST.tsv"
    )
    if not manifest.is_file():
        pytest.skip("ranked Rollman pool is unavailable")

    prepared = prepare_opponent(
        load_human_pool(manifest)[13],
        build_root=tmp_path / "build",
    )

    assert prepared.process is not None
    assert Path(prepared.process.argv[0]).is_file()

import json
from pathlib import Path

from agentbench_frame.doto.lifecycle import init_run
import pytest

from agentbench_frame.doto.run_store import RunState, hash_directory, snapshot_skills


SKILL_NAMES = (
    "doto-benchmark-run",
    "doto-game-rules",
    "doto-agent-authoring",
    "doto-replay-reader",
)


def make_inputs(tmp_path):
    source = tmp_path / "playerAI.cpp"
    source.write_text("void playerAI() {}\n")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "bundle-manifest.json").write_text('{"benchmark_version":"fixture-v1"}\n')
    (test / "sealed-manifest.json").write_text('{"benchmark_version":"fixture-v1"}\n')
    skills = {}
    for name in SKILL_NAMES:
        root = tmp_path / "skills" / name
        root.mkdir(parents=True)
        (root / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n# {name}\n"
        )
        skills[name] = root
    return source, train, test, skills


def test_init_snapshots_source_skills_and_pools(tmp_path):
    source, train, test, skills = make_inputs(tmp_path)
    run = init_run(tmp_path / "DotoResults", "codex", source, train, test, skills, run_id="r1")
    assert run.state == RunState.CREATED
    assert (run.run_dir / "iterations/iteration-0000/playerAI.cpp").is_file()
    rows = json.loads((run.run_dir / "skills/manifest.json").read_text())["skills"]
    assert {row["name"] for row in rows} == set(SKILL_NAMES)
    assert all(row["sha256"] == hash_directory(run.run_dir / "skills" / row["name"])
               for row in rows)
    run_toml = (run.run_dir / "run.toml").read_text()
    assert "budget" not in run_toml.lower()
    assert str(test) not in (run.run_dir / "pools/test.json").read_text()


def test_skill_snapshot_rejects_escaping_symlink(tmp_path):
    _, _, _, skills = make_inputs(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("hidden")
    (skills["doto-game-rules"] / "references").mkdir()
    (skills["doto-game-rules"] / "references/escape.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        snapshot_skills(tmp_path / "run", skills)

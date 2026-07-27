from dataclasses import replace
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import (
    AssetValidationError,
    _stable_tree_hash,
    load_pilot_config,
    load_round3_learning_config,
    resolve_assets,
)


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"
ROUND3_FIXTURE = (
    Path(__file__).parent / "fixtures" / "v3-strongest-learning-v1.toml"
)


def test_manifest_freezes_expected_matrix_and_disjoint_learning_seeds():
    config = load_pilot_config(FIXTURE)

    assert config.benchmark_id == "generals-hl-pilot-v1"
    assert config.evaluation_seeds == (280101, 280202, 280303)
    assert config.learning_seeds == (281101, 281202, 281303)
    assert set(config.evaluation_seeds).isdisjoint(config.learning_seeds)
    assert tuple(item.tier for item in config.opponents) == ("high", "medium", "low")
    assert tuple(item.opponent_id for item in config.learning_opponents) == (
        "advanced-rank08-nashjunheng-v20",
        "popular-rank16-xiaoaojianghu-v1",
    )


def test_resolver_rejects_source_path_outside_agentbench_root(tmp_path):
    config = load_pilot_config(FIXTURE)
    escaped = replace(config, engine_path=Path("../outside"))

    with pytest.raises(AssetValidationError, match="escapes AgentBench root"):
        resolve_assets(escaped, tmp_path)


def test_loader_rejects_seed_overlap(tmp_path):
    text = FIXTURE.read_text().replace(
        "learning_seeds = [281101, 281202, 281303]",
        "learning_seeds = [280101, 281202, 281303]",
    )
    manifest = tmp_path / "pilot.toml"
    manifest.write_text(text)

    with pytest.raises(AssetValidationError, match="must be disjoint"):
        load_pilot_config(manifest)


def test_loader_rejects_wrong_frozen_limits(tmp_path):
    text = FIXTURE.read_text().replace("decision_timeout_s = 2", "decision_timeout_s = 3")
    manifest = tmp_path / "pilot.toml"
    manifest.write_text(text)

    with pytest.raises(AssetValidationError, match="frozen values"):
        load_pilot_config(manifest)


def test_engine_hash_ignores_generated_python_cache(tmp_path):
    (tmp_path / "main.py").write_text("VALUE = 1\n")
    before = _stable_tree_hash(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"generated")
    assert _stable_tree_hash(tmp_path) == before


def test_round3_learning_manifest_freezes_strongest_human_matrix():
    pilot = load_pilot_config(FIXTURE)

    learning = load_round3_learning_config(ROUND3_FIXTURE, pilot)

    assert learning.learning_id == "generals-hl-v3-strongest-v1"
    assert learning.opponent_id == "advanced-rank02-robinliu-v18"
    assert learning.seeds == (284101, 284202, 284303)
    assert learning.seats == (0, 1)


def test_round3_learning_rejects_any_previously_used_seed(tmp_path):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        ROUND3_FIXTURE.read_text().replace("284101", "283101"),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="previously frozen"):
        load_round3_learning_config(manifest, pilot)


def test_round3_learning_rejects_non_highest_opponent(tmp_path):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        ROUND3_FIXTURE.read_text().replace(
            "advanced-rank02-robinliu-v18",
            "advanced-rank08-nashjunheng-v20",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="highest-tier"):
        load_round3_learning_config(manifest, pilot)

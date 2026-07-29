from dataclasses import replace
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import (
    AssetValidationError,
    _stable_tree_hash,
    load_policy_kl_reference_config,
    load_pilot_config,
    load_round3_learning_config,
    load_round4_learning_config,
    load_round5_learning_config,
    resolve_assets,
    resolve_replay_skill,
)


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"
ROUND3_FIXTURE = (
    Path(__file__).parent / "fixtures" / "v3-strongest-learning-v1.toml"
)
ROUND4_FIXTURE = (
    Path(__file__).parent / "fixtures" / "v4-strongest-learning-v1.toml"
)
ROUND5_FIXTURE = (
    Path(__file__).parent / "fixtures" / "v5-rollback-learning-v1.toml"
)
POLICY_KL_REFERENCE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "policy-kl-reference-v1.toml"
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


def test_round4_learning_manifest_freezes_new_strongest_human_matrix():
    pilot = load_pilot_config(FIXTURE)

    learning = load_round4_learning_config(ROUND4_FIXTURE, pilot)

    assert learning.learning_id == "generals-hl-v4-strongest-v1"
    assert learning.opponent_id == "advanced-rank02-robinliu-v18"
    assert learning.seeds == (285101, 285202, 285303)
    assert learning.seats == (0, 1)


@pytest.mark.parametrize("used_seed", ["280101", "283101", "284101"])
def test_round4_learning_rejects_every_previously_used_seed(
    tmp_path,
    used_seed,
):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        ROUND4_FIXTURE.read_text().replace("285101", used_seed),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="previously frozen"):
        load_round4_learning_config(manifest, pilot)


def test_round5_learning_manifest_freezes_rollback_strongest_matrix():
    pilot = load_pilot_config(FIXTURE)

    learning = load_round5_learning_config(ROUND5_FIXTURE, pilot)

    assert learning.learning_id == "generals-hl-v5-rollback-strongest-v1"
    assert learning.opponent_id == "advanced-rank02-robinliu-v18"
    assert learning.seeds == (286101, 286202, 286303)
    assert learning.seats == (0, 1)


@pytest.mark.parametrize(
    "used_seed",
    ["280101", "281101", "284101", "285101"],
)
def test_round5_learning_rejects_every_previously_used_seed(
    tmp_path,
    used_seed,
):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        ROUND5_FIXTURE.read_text().replace("286101", used_seed),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="previously frozen"):
        load_round5_learning_config(manifest, pilot)


def test_replay_skill_is_resolved_below_root_with_stable_digest(tmp_path):
    skill = tmp_path / "skills" / "replay-analysis-v1" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Replay skill\nHuman-authored semantics.\n", encoding="utf-8")

    resolved = resolve_replay_skill(
        tmp_path,
        Path("skills/replay-analysis-v1/SKILL.md"),
    )

    assert resolved.path == skill.resolve()
    assert resolved.text == skill.read_text(encoding="utf-8")
    assert len(resolved.sha256) == 64


def test_replay_skill_rejects_path_escape(tmp_path):
    with pytest.raises(AssetValidationError, match="escapes AgentBench root"):
        resolve_replay_skill(tmp_path, Path("../outside/SKILL.md"))


def test_policy_kl_reference_manifest_freezes_common_history_and_domain():
    pilot = load_pilot_config(FIXTURE)

    config = load_policy_kl_reference_config(
        POLICY_KL_REFERENCE_FIXTURE,
        pilot,
    )

    assert config.measurement_id == "generals-policy-kl-reference-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.seeds == (289101, 289202, 289303)
    assert config.seats == (0, 1)
    assert config.decision_numbers == (2, 10)
    assert config.epsilons == ("0.001", "0.01", "0.05", "0.1")
    assert config.primary_epsilon == "0.01"
    assert tuple(item.version for item in config.history) == (
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "seeds = [289101, 289202, 289303]",
            "seeds = [288101, 289202, 289303]",
            "previously frozen",
        ),
        ("seats = [0, 1]", "seats = [1, 0]", "seats"),
        ('version = "v6"', 'version = "v5"', "versions"),
        (
            'content_hash = "974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b"',
            'content_hash = "not-a-sha256"',
            "content hash",
        ),
        (
            'opponent_id = "advanced-rank02-robinliu-v18"',
            'opponent_id = "advanced-rank08-nashjunheng-v20"',
            "highest-tier",
        ),
        (
            'primary_epsilon = "0.01"',
            'primary_epsilon = "0.02"',
            "primary epsilon",
        ),
    ],
)
def test_policy_kl_reference_manifest_rejects_contract_changes(
    tmp_path,
    old,
    new,
    message,
):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "reference.toml"
    manifest.write_text(
        POLICY_KL_REFERENCE_FIXTURE.read_text(encoding="utf-8").replace(
            old,
            new,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match=message):
        load_policy_kl_reference_config(manifest, pilot)

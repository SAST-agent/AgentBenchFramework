from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)


V7_LEARNING_SEEDS = (290101, 290202, 290303)
SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_SOURCE = (
    REPOSITORY_ROOT
    / "agentbench_data/runs/28_generals/generals-hl"
    / "20260729_1653_af8eda26/versions/v6/source"
)
ASSET_ROOT = (
    REPOSITORY_ROOT.parents[2]
    / "AgentBench/.worktrees/generals-assets"
    / "backend_sources/corpus/28_generals"
)


def _module():
    from agentbench_frame.generals import prompt_v7

    return prompt_v7


def _evidence(seed: int, seat: int) -> CriticalLearningEvidence:
    replay_id = f"learn7-high-s{seed}-p{seat}"
    return CriticalLearningEvidence(
        replay_id=replay_id,
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={"army_margin": {"terminal": -200.0}},
        total_decision_count=1,
        omitted_decision_count=0,
        decisions=(
            CriticalDecision(
                state_id=f"v6:{replay_id}-d12",
                round_number=12,
                seat=seat,
                selection_reasons=("first_strategic_opportunity",),
                action=((1, 1, 1, 1, 2, 8), (8,)),
                decision_class="main_army_move",
                features={"coin": 48, "owned_resource_generals": 0},
            ),
        ),
    )


def _records() -> tuple[CriticalLearningEvidence, ...]:
    return tuple(
        _evidence(seed, seat)
        for seed in V7_LEARNING_SEEDS
        for seat in (0, 1)
    )


def _build(records=None, **overrides):
    arguments = {
        "benchmark_id": "generals-hl-pilot-v1",
        "v6_strategy": "exact editable v6 strategy",
        "v6_experience": "retained experience from learn6-high-s287101-p0",
        "rules_text": "official rules",
        "replay_skill_text": "frozen human replay skill",
        "replay_skill_sha256": SKILL_SHA256,
        "evidence": _records() if records is None else records,
        "action_profile": {
            "command_counts": {
                "1": 2480,
                "2": 0,
                "3": 64,
                "4": 0,
                "5": 3,
                "6": 0,
                "7": 0,
            }
        },
        "max_bytes": 131_072,
    }
    arguments.update(overrides)
    return _module().build_round7_prompt(**arguments)


def test_v7_prompt_contains_exact_learning_evidence_and_planner_contract():
    result = _build()

    assert result.feedback_episodes_read == 6
    assert result.feedback_decision_records_read == 6
    assert result.included_episode_ids == tuple(
        f"learn7-high-s{seed}-p{seat}"
        for seed in V7_LEARNING_SEEDS
        for seat in (0, 1)
    )
    assert result.omitted_episode_ids == ()
    assert "commands 1 through 7" in result.prompt
    assert "at most eight non-end primitives" in result.prompt
    assert "main.py is immutable" in result.prompt
    assert "private normalized-state clone" in result.prompt
    assert "v6 commands 1/3/5" in result.prompt
    assert "fixed beam width" in result.prompt
    assert "two-second decision limit" in result.prompt
    assert "compress" in result.prompt.lower()


def test_v7_prompt_manifest_is_complete_and_derived_from_output():
    result = _build()

    assert result.manifest == {
        "episode_count": 6,
        "episode_ids": list(result.included_episode_ids),
        "evidence_decision_ids": [
            f"v6:learn7-high-s{seed}-p{seat}-d12"
            for seed in V7_LEARNING_SEEDS
            for seat in (0, 1)
        ],
        "feedback_serialized_bytes": result.feedback_serialized_bytes_read,
        "forbidden_partitions": [
            "calibration",
            "controlled_policy_kl",
            "historical_formal",
            "sealed",
            "validation",
        ],
        "omitted_bytes": 0,
        "prompt_bytes": result.prompt_bytes,
        "prompt_sha256": hashlib.sha256(
            result.prompt.encode("utf-8")
        ).hexdigest(),
        "replay_skill_sha256": SKILL_SHA256,
        "selection_policy": "exact_v7_champion_learning_only",
    }


def test_v7_static_context_accepts_exact_production_bundle():
    paths = {
        "v6_strategy": PARENT_SOURCE / "strategy.py",
        "v6_experience": PARENT_SOURCE / "EXPERIENCE.md",
        "rules_text": ASSET_ROOT / "benchmark" / "rules.md",
        "replay_skill_text": (
            ASSET_ROOT / "skills" / "replay-analysis-v2" / "SKILL.md"
        ),
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("production v6 or Generals assets are unavailable")

    _module().validate_round7_static_context(
        **{
            name: path.read_text(encoding="utf-8")
            for name, path in paths.items()
        }
    )


@pytest.mark.parametrize(
    "seed",
    [
        280101,
        281101,
        282101,
        283101,
        284101,
        285101,
        286101,
        287101,
        288101,
        289101,
        291101,
        292101,
    ],
)
def test_v7_prompt_rejects_every_nonlearning_partition_seed(seed):
    records = list(_records())
    records[-1] = _evidence(seed, 1)

    with pytest.raises(ValueError, match="forbidden round-7 evidence seed"):
        _build(tuple(records))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v6_strategy", "formal result seed 280101"),
        ("rules_text", "validation trajectory seed 291101"),
        ("replay_skill_text", "sealed trajectory seed 292101"),
        ("action_profile", {"policy_kl_seed": 289101}),
    ],
)
def test_v7_prompt_rejects_nonlearning_static_or_diagnostic_payload(
    field,
    value,
):
    with pytest.raises(ValueError, match="forbidden round-7"):
        _build(**{field: value})


def test_v7_parent_experience_allows_v6_learning_only():
    _module().validate_round7_static_context(
        v6_strategy="editable strategy",
        v6_experience="learn6-high-s287101-p0",
        rules_text="official rules",
        replay_skill_text="generic replay schema",
    )

    with pytest.raises(ValueError, match="v6 experience seed"):
        _module().validate_round7_static_context(
            v6_strategy="editable strategy",
            v6_experience="validation replay s288101",
            rules_text="official rules",
            replay_skill_text="generic replay schema",
        )


def test_v7_prompt_rejects_wrong_skill_digest():
    with pytest.raises(ValueError, match="frozen replay skill"):
        _build(replay_skill_sha256="a" * 64)


def test_v7_prompt_rejects_duplicate_or_incomplete_evidence():
    records = list(_records())
    records[-1] = replace(records[-1], replay_id=records[0].replay_id)
    with pytest.raises(ValueError, match="duplicate"):
        _build(tuple(records))

    with pytest.raises(ValueError, match="exactly six"):
        _build(_records()[:-1])


def test_v7_prompt_rejects_noncanonical_episode_id_and_champion_source():
    records = list(_records())
    records[-1] = replace(records[-1], replay_id="validation-case")
    with pytest.raises(ValueError, match="canonical"):
        _build(tuple(records))

    with pytest.raises(ValueError, match="opponent source"):
        _build(v6_strategy="advanced-rank02-robinliu-v18 source")


def test_v7_prompt_rejects_oversized_context():
    with pytest.raises(ValueError, match="max_bytes"):
        _build(max_bytes=128)

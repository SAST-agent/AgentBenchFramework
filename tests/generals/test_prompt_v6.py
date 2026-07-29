from dataclasses import replace
import hashlib
import os
from pathlib import Path

import pytest

from agentbench_frame.generals import prompt_v6
from agentbench_frame.generals.prompt import FORBIDDEN_EVALUATION_SEEDS
from agentbench_frame.generals.prompt_v6 import build_round6_prompt
from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)


V6_SEEDS = (287101, 287202, 287303)
HISTORICAL_SEEDS = (
    281101,
    281202,
    281303,
    282101,
    282202,
    282303,
    282404,
    282505,
    282601,
    282702,
    282803,
    282904,
    283005,
    283101,
    283202,
    283303,
    284101,
    284202,
    284303,
    285101,
    285202,
    285303,
    286101,
    286202,
    286303,
)
VALIDATION_SEEDS = (288101, 288202, 288303)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_V5_SOURCE = (
    REPOSITORY_ROOT
    / "agentbench_data/runs/28_generals/generals-hl"
    / "20260729_0818_e6bcb9b3/versions/v5/source"
)
PRODUCTION_STATIC_SHA256 = {
    "v5_strategy": (
        "18b4560ef80d802d06c88acbfe1e790a07085a9a46a8fec0020550b81ffbe975"
    ),
    "v5_experience": (
        "c9a6d800fcca6f8091bc6221e35b61655715860f5129937ceb898f9cbaf6fc0d"
    ),
    "rules_text": (
        "9138bb16a27714b9be2403a28bac3f81d2de5b667a7eec1dbb3198887f516950"
    ),
    "replay_skill_text": (
        "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
    ),
}


def _production_static_paths() -> dict[str, Path]:
    assets_override = os.environ.get("AGENTBENCH_GENERALS_ASSETS_ROOT")
    if assets_override:
        assets_candidates = [Path(assets_override)]
    else:
        assets_candidates = [
            candidate
            for ancestor in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
            for candidate in (
                ancestor / "AgentBench/.worktrees/generals-assets",
                ancestor / "generals-assets",
            )
        ]
    assets_root = next(
        (
            candidate
            for candidate in assets_candidates
            if (
                candidate
                / "backend_sources/corpus/28_generals/benchmark/rules.md"
            ).is_file()
        ),
        assets_candidates[0],
    )
    paths = {
        "v5_strategy": PARENT_V5_SOURCE / "strategy.py",
        "v5_experience": PARENT_V5_SOURCE / "EXPERIENCE.md",
        "rules_text": (
            assets_root
            / "backend_sources/corpus/28_generals/benchmark/rules.md"
        ),
        "replay_skill_text": (
            assets_root
            / "backend_sources/corpus/28_generals"
            / "skills/replay-analysis-v2/SKILL.md"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        pytest.skip(
            "exact production static bundle is unavailable; set "
            "AGENTBENCH_GENERALS_ASSETS_ROOT for a nonstandard assets checkout; "
            f"missing: {', '.join(missing)}"
        )
    return paths


def _production_static_context() -> dict[str, str]:
    static_context = {}
    for role, path in _production_static_paths().items():
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == (
            PRODUCTION_STATIC_SHA256[role]
        )
        static_context[role] = payload.decode("utf-8")
    return static_context


def _evidence(seed: int, seat: int) -> CriticalLearningEvidence:
    return CriticalLearningEvidence(
        replay_id=f"learn6-high-s{seed}-p{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={"army_margin": {"terminal": -5.0}},
        total_decision_count=1,
        omitted_decision_count=0,
        decisions=(
            CriticalDecision(
                state_id=f"learn6-high-s{seed}-p{seat}-d1",
                round_number=1,
                seat=seat,
                selection_reasons=("first_decision",),
                action=((8,),),
                decision_class="end_only",
                features={"own_main_army": 1},
            ),
        ),
    )


def _records() -> tuple[CriticalLearningEvidence, ...]:
    return tuple(_evidence(seed, seat) for seed in V6_SEEDS for seat in (0, 1))


def _build(
    records: tuple[CriticalLearningEvidence, ...],
    max_bytes: int = 131_072,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "benchmark_id": "generals-hl-pilot-v1",
        "v5_strategy": "exact editable v5 strategy",
        "v5_experience": "v5 experience",
        "rules_text": "official rules",
        "replay_skill_text": "frozen human replay skill",
        "replay_skill_sha256": "a" * 64,
        "evidence": records,
        "action_profile": {"mean_primitives_per_turn": 1.0},
        "max_bytes": max_bytes,
    }
    arguments.update(overrides)
    return build_round6_prompt(
        **arguments,  # type: ignore[arg-type]
    )


def test_v6_exposes_one_static_context_preflight_with_prompt_leak_semantics():
    validator = getattr(prompt_v6, "validate_round6_static_context", None)

    assert callable(validator)
    validator(
        v5_strategy="editable strategy",
        v5_experience="retained experience",
        rules_text="official rules",
        replay_skill_text="generic replay schema",
    )
    with pytest.raises(ValueError, match="formal or validation material"):
        validator(
            v5_strategy="editable strategy",
            v5_experience="retained experience",
            rules_text="official rules",
            replay_skill_text="validation trajectory from a held-out split",
        )


def test_exact_production_static_bundle_passes_without_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prompt_v6.validate_round6_static_context(**_production_static_context())

    assert list(tmp_path.iterdir()) == []


def test_exact_production_bundle_builds_complete_prompt_without_artifacts(
    tmp_path,
    monkeypatch,
):
    expected_episode_ids = (
        "learn6-high-s287101-p0",
        "learn6-high-s287101-p1",
        "learn6-high-s287202-p0",
        "learn6-high-s287202-p1",
        "learn6-high-s287303-p0",
        "learn6-high-s287303-p1",
    )
    monkeypatch.chdir(tmp_path)

    result = build_round6_prompt(
        benchmark_id="generals-hl-pilot-v1",
        **_production_static_context(),
        replay_skill_sha256=PRODUCTION_STATIC_SHA256["replay_skill_text"],
        evidence=_records(),
        action_profile={
            "turn_count": 1228,
            "mean_primitives_per_turn": 0.988599348534202,
        },
    )

    assert "learn5-high-s286101-p0" in result.prompt
    assert "v3:learn5-high-s286101-p0-d64" in result.prompt
    assert result.included_episode_ids == expected_episode_ids
    assert result.feedback_episodes_read == 6
    assert result.prompt.count('"replay_id": "learn6-high-s287') == 6
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "v5_experience",
    (
        "learn1-high-s280101-p0",
        "learn6-high-s288101-p0",
        "learn4-high-s285101-p0",
        "learn6-high-s287101-p0",
        "unscoped gameplay seed 299999",
        "formal result without a seed",
        "validation trace without a seed",
    ),
)
def test_v5_experience_rejects_non_round5_learning_context(v5_experience):
    with pytest.raises(
        ValueError,
        match=r"forbidden (?:round-6|evaluation)",
    ):
        prompt_v6.validate_round6_static_context(
            v5_strategy="editable strategy",
            v5_experience=v5_experience,
            rules_text="official rules",
            replay_skill_text="generic replay schema",
        )


@pytest.mark.parametrize(
    "context_name",
    ("v5_strategy", "rules_text", "replay_skill_text"),
)
def test_round5_learning_citations_remain_forbidden_outside_experience(
    context_name,
):
    context = {
        "v5_strategy": "editable strategy",
        "v5_experience": "retained experience",
        "rules_text": "official rules",
        "replay_skill_text": "generic replay schema",
    }
    context[context_name] = "learn5-high-s286101-p0"

    with pytest.raises(ValueError, match="forbidden round-6 seed"):
        prompt_v6.validate_round6_static_context(**context)


def test_v6_prompt_isolates_exact_high_only_learning_episodes():
    result = _build(_records())

    assert result.feedback_episodes_read == 6
    assert result.feedback_decision_records_read == 6
    assert result.omitted_episode_ids == ()
    assert result.included_episode_ids == tuple(
        f"learn6-high-s{seed}-p{seat}"
        for seed in V6_SEEDS
        for seat in (0, 1)
    )
    assert "bounded macro-action planner" in result.prompt
    assert "You may edit strategy.py, state_view.py" in result.prompt
    assert "formal score" not in result.prompt.lower()
    assert "ACTION-PROFILE DIAGNOSTICS" in result.prompt
    assert "AUTHORITATIVE FROZEN HUMAN REPLAY SKILL" in result.prompt
    assert "sha256=" + "a" * 64 in result.prompt
    assert "frozen human replay skill" in result.prompt


@pytest.mark.parametrize(
    "seed",
    sorted(
        set(FORBIDDEN_EVALUATION_SEEDS)
        | set(HISTORICAL_SEEDS)
        | set(VALIDATION_SEEDS)
    ),
)
def test_v6_prompt_rejects_formal_historical_and_validation_seeds(seed: int):
    records = list(_records())
    records[-1] = _evidence(seed, 1)

    with pytest.raises(ValueError, match="forbidden round-6 evidence seed"):
        _build(tuple(records))


def test_v6_prompt_rejects_non_high_evidence():
    records = list(_records())
    records[-1] = replace(records[-1], opponent_tier="medium")

    with pytest.raises(ValueError, match="high-tier"):
        _build(tuple(records))


def test_v6_prompt_rejects_duplicate_episode_ids():
    records = list(_records())
    records[-1] = replace(records[-1], replay_id=records[0].replay_id)

    with pytest.raises(ValueError, match="duplicate round-6 evidence episode"):
        _build(tuple(records))


def test_v6_prompt_rejects_incomplete_or_unexpected_episode_sets():
    with pytest.raises(ValueError, match="exactly six"):
        _build(_records()[:-1])

    records = list(_records())
    records[-1] = _evidence(299999, 1)
    with pytest.raises(ValueError, match="exactly six"):
        _build(tuple(records))


def test_v6_prompt_rejects_complete_prompt_over_the_byte_cap():
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        _build(_records(), max_bytes=1)


@pytest.mark.parametrize(
    "benchmark_id",
    (
        "generals-formal-pilot-v1",
        "generals-validation-pilot-v1",
        "generals-286101-pilot-v1",
    ),
)
def test_v6_prompt_rejects_untrusted_benchmark_context(benchmark_id):
    with pytest.raises(
        ValueError,
        match=r"forbidden (?:round-6|evaluation)",
    ):
        _build(_records(), benchmark_id=benchmark_id)


@pytest.mark.parametrize(
    "marker",
    (
        "formal replay payload",
        "validation trajectory payload",
        "historical replay s286101 payload",
    ),
)
def test_v6_prompt_rejects_untrusted_serialized_evidence_context(marker):
    records = list(_records())
    records[-1] = replace(records[-1], dense={"marker": marker})

    with pytest.raises(
        ValueError,
        match=r"forbidden (?:round-6|evaluation)",
    ):
        _build(tuple(records))


@pytest.mark.parametrize(
    ("context_name", "payload_marker"),
    (
        ("v5_strategy", "FORMAL REPLAY: held-out turn data"),
        ("v5_experience", "VaLiDaTiOn trajectory: held-out turn data"),
        ("rules_text", "FORMAL DENSE TRACE: held-out metrics"),
        ("replay_skill_text", "validation action profile: held-out metrics"),
        ("action_profile", {"marker": "FORMAL OUTCOME: win"}),
    ),
)
def test_v6_prompt_rejects_formal_or_validation_payload_context(
    context_name: str,
    payload_marker: object,
):
    with pytest.raises(ValueError, match="formal or validation material"):
        _build(_records(), **{context_name: payload_marker})


@pytest.mark.parametrize(
    ("context_name", "payload_marker"),
    (
        ("v5_strategy", "formal methodology notes"),
        ("v5_experience", "formal evaluation results"),
        ("rules_text", "validation metrics"),
        ("replay_skill_text", "validation methodology notes"),
        ("action_profile", {"marker": "FORMAL methodology notes"}),
    ),
)
def test_v6_prompt_rejects_standalone_formal_or_validation_context(
    context_name: str,
    payload_marker: object,
):
    with pytest.raises(ValueError, match="formal or validation material"):
        _build(_records(), **{context_name: payload_marker})

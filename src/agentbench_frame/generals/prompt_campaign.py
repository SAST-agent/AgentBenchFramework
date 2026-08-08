"""Leak-closed prompts for the multi-act Generals champion campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import re

from .champion_campaign import campaign_learning_seeds
from .leaderboard_qualification import QUALIFICATION_SEEDS
from .models import ChampionCampaignConfig
from .prompt import PromptBuildResult
from .replay import CriticalLearningEvidence


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_QUALIFICATION_ARTIFACT = re.compile(
    r"(?:leaderboard[-_ ]qualification|qualification[-_ ](?:receipt|result|score)"
    r"|qualifying[-_ ]replicates|selection[-_ ]uses[-_ ]qualification)"
    r"|generals-api-leaderboard-qualification",
    re.IGNORECASE,
)
_OPPONENT_SOURCE_MARKERS = (
    "advanced-rank02-robinliu-v18",
    "top_algorithms/",
)
_REQUIRED_DECISION_TERMS = (
    "[1, row, column, direction, amount]",
    "[2, general_id, row, column]",
    "[3, general_id, quality]",
    "[4, general_id, skill, row, column]",
    "[5, technology]",
    "[6, weapon, row, column]",
    "[7, row, column]",
    "[8]",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reject_held_out_material(text: str, *, field: str) -> None:
    for seed in QUALIFICATION_SEEDS:
        if str(seed) in text:
            raise ValueError(
                f"forbidden qualification seed in campaign {field}: {seed}"
            )
    if _QUALIFICATION_ARTIFACT.search(text):
        raise ValueError(
            f"forbidden qualification artifact in campaign {field}"
        )
    for marker in _OPPONENT_SOURCE_MARKERS:
        if marker in text:
            raise ValueError(
                f"forbidden opponent source material in campaign {field}"
            )


def _validate_static_context(
    config: ChampionCampaignConfig,
    *,
    benchmark_id: str,
    rules_text: str,
    decision_space_text: str,
    replay_skill_text: str,
) -> None:
    expected = (
        (rules_text, config.rules_sha256, "rules"),
        (
            decision_space_text,
            config.decision_space_sha256,
            "decision space",
        ),
        (replay_skill_text, config.replay_skill_sha256, "replay skill"),
    )
    _reject_held_out_material(benchmark_id, field="benchmark ID")
    for text, digest, field in expected:
        if _sha256_text(text) != digest:
            raise ValueError(f"campaign {field} digest does not match")
        _reject_held_out_material(text, field=field)
    missing = tuple(
        term for term in _REQUIRED_DECISION_TERMS if term not in decision_space_text
    )
    if missing:
        raise ValueError(
            "campaign decision space is incomplete: " + ", ".join(missing)
        )


def _validate_evidence(
    config: ChampionCampaignConfig,
    *,
    replicate_id: str,
    act_index: int,
    evidence: Sequence[CriticalLearningEvidence],
) -> tuple[CriticalLearningEvidence, ...]:
    expected_pairs = frozenset(
        (seed, seat)
        for seed in campaign_learning_seeds(config, act_index)
        for seat in config.seats
    )
    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.seed,
                item.evaluated_seat,
                item.replay_id,
            ),
        )
    )
    seen: set[str] = set()
    for item in ordered:
        if item.seed in QUALIFICATION_SEEDS:
            raise ValueError(
                f"forbidden qualification seed in campaign evidence: {item.seed}"
            )
        if item.opponent_tier != "high":
            raise ValueError("campaign evidence must be high-tier data")
        if item.replay_id in seen:
            raise ValueError(
                f"duplicate campaign evidence episode: {item.replay_id}"
            )
        seen.add(item.replay_id)
        expected_id = (
            f"campaign-{replicate_id}-a{act_index}-high-"
            f"s{item.seed}-p{item.evaluated_seat}"
        )
        if item.replay_id != expected_id:
            raise ValueError(
                "campaign evidence episode ID must use the canonical identity"
            )
        _reject_held_out_material(item.to_json(), field="evidence")

    pairs = frozenset(
        (item.seed, item.evaluated_seat) for item in ordered
    )
    if len(ordered) != len(expected_pairs) or pairs != expected_pairs:
        raise ValueError(
            "campaign evidence must contain exactly the six declared episodes"
        )
    return ordered


def build_champion_campaign_prompt(
    *,
    config: ChampionCampaignConfig,
    benchmark_id: str,
    replicate_id: str,
    act_index: int,
    current_policy_version: str,
    current_policy_hash: str,
    rules_text: str,
    decision_space_text: str,
    replay_skill_text: str,
    evidence: Sequence[CriticalLearningEvidence],
    action_profile: Mapping[str, object],
) -> PromptBuildResult:
    """Build one exact, learning-only act with no held-out gate feedback."""

    if replicate_id not in {
        f"replicate-{index}" for index in range(1, config.replicates + 1)
    }:
        raise ValueError("campaign replicate ID is not frozen")
    campaign_learning_seeds(config, act_index)
    if not current_policy_version or not _DIGEST.fullmatch(current_policy_hash):
        raise ValueError("campaign current policy identity is invalid")
    _reject_held_out_material(current_policy_version, field="policy version")
    _validate_static_context(
        config,
        benchmark_id=benchmark_id,
        rules_text=rules_text,
        decision_space_text=decision_space_text,
        replay_skill_text=replay_skill_text,
    )
    ordered = _validate_evidence(
        config,
        replicate_id=replicate_id,
        act_index=act_index,
        evidence=evidence,
    )
    try:
        action_profile_json = json.dumps(
            dict(action_profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "campaign action profile must be JSON serializable"
        ) from exc
    _reject_held_out_material(action_profile_json, field="action profile")

    mandatory = f"""You are performing act {act_index} of a deterministic
Generals champion campaign for {benchmark_id}, {replicate_id}. The editable
workspace is the entire current policy. Its version is
{current_policy_version} and its content hash is {current_policy_hash}.

Use only that workspace, the three frozen authorities, the learning-only
action profile, and the exact six high-tier learning episodes below. Held-out
gate games are unavailable and must not influence implementation or policy
selection. Do not inspect any opponent implementation or other gameplay
artifact.

Preserve choose_actions(round_number, my_seat, view) -> list[list[int]]. You
may edit strategy.py, state_view.py, STRATEGY.md, EXPERIENCE.md, tests/**, and
policy/**. main.py is immutable. Runtime policy inputs are only round_number,
my_seat, and the normalized view. Do not use randomness, seeds, replay IDs,
opponent identity, filesystem paths, environment variables, network, wall
clock, process state, or future/hidden state.

Implement and improve a deterministic planner over all official command
families 1 through 7. Candidate generation must consider army movement,
general movement, general upgrades, all five skills, all four technologies,
all four super weapons, and recruitment before deterministic pruning. Model
each selected primitive on a private cloned state before planning the next.
Unknown transition semantics must end the verified prefix. Emit at most
{config.max_non_end_primitives} non-end primitives and exactly one final [8].

state_view.py must expose all legality-bearing state, including both players'
coins and army movement budgets (rest_move_step); terrain, owner, army, and general ID for every
cell; exact general type, levels, skills_cd, active skill durations, and
rest_move; technology, super_weapon_unlocked, super_weapon_cd, next general
ID; and each active weapon's owner, type, position, cd, and duration.

Use the official transition contract below as authority. Add or strengthen
legal/illegal and transition-parity tests for every enabled family. Maintain
deterministic tie ordering, one-army source reserve, affordability, ownership,
cooldown, final-[8], macro-length, fallback, and latency tests. Run the full
candidate test suite. Keep STRATEGY.md executable and compress EXPERIENCE.md
into evidence-backed hypotheses, retained behavior, rejected ideas, tests,
and unresolved risks.

Episode outcomes and dense/action diagnostics are learning evidence, not a
benchmark score or proof of victory. Make one coherent improvement even when
the six episodes all lose. Finish with a concise change and test summary.

OFFICIAL RULE SUMMARY
{rules_text}

FROZEN DECISION SPACE
schema={config.action_space_schema}
spec_id={config.action_space_spec_id}
sha256={config.decision_space_sha256}
{decision_space_text}

FROZEN REPLAY READING SKILL
sha256={config.replay_skill_sha256}
{replay_skill_text}

LEARNING-ONLY CURRENT-POLICY ACTION PROFILE
{action_profile_json}

EXACT SIX CURRENT-ACT LEARNING EPISODES
"""
    lines = tuple(item.to_json() + "\n" for item in ordered)
    prompt = mandatory + "".join(lines)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > config.prompt_max_bytes:
        raise ValueError("campaign prompt context exceeds prompt_max_bytes")

    included_ids = tuple(item.replay_id for item in ordered)
    feedback_bytes = sum(len(line.encode("utf-8")) for line in lines)
    result = PromptBuildResult(
        prompt=prompt,
        included_episode_ids=included_ids,
        omitted_episode_ids=(),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=False,
        selection_policy="exact_champion_campaign_learning_only_v1",
        feedback_episodes_read=len(ordered),
        feedback_decision_records_read=sum(
            len(item.decisions) for item in ordered
        ),
        feedback_serialized_bytes_read=feedback_bytes,
    )
    manifest = {
        "act_index": act_index,
        "action_space_spec_id": config.action_space_spec_id,
        "current_policy_hash": current_policy_hash,
        "decision_space_sha256": config.decision_space_sha256,
        "episode_count": len(ordered),
        "episode_ids": list(included_ids),
        "feedback_serialized_bytes": feedback_bytes,
        "forbidden_partitions": ["leaderboard_qualification"],
        "omitted_bytes": 0,
        "prompt_bytes": prompt_bytes,
        "prompt_sha256": _sha256_text(prompt),
        "replay_skill_sha256": config.replay_skill_sha256,
        "replicate_id": replicate_id,
        "rules_sha256": config.rules_sha256,
        "selection_policy": result.selection_policy,
    }
    return replace(result, manifest=manifest)

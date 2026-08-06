"""Leak-safe single-act prompt for the clean-room Generals v8 iteration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
from typing import Mapping, Sequence

from .challenge_v7 import ROUND7_LEARNING_SEEDS
from .challenge_v8 import (
    FROZEN_BEFORE_ROUND8,
    ROUND8_LEARNING_SEEDS,
    ROUND8_VALIDATION_SEEDS,
)
from .prompt import PromptBuildResult
from .replay import CriticalLearningEvidence


ROUND8_REPLAY_SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)
_EPISODE_PAIRS = frozenset(
    (seed, seat) for seed in ROUND8_LEARNING_SEEDS for seat in (0, 1)
)
_SIX_DIGIT_SEED = re.compile(r"(?<!\d)\d{6}(?!\d)")
_FORBIDDEN_MARKER = re.compile(
    r"\b(?:formal(?:_score)?|validation|sealed|calibration|"
    r"controlled[_ -]?policy[_ -]?kl|transition-parity act|blind v8 retry)\b",
    re.IGNORECASE,
)
_SOURCE_MARKERS = ("top_algorithms/", "advanced-rank02-robinliu-v18")


def _reject_text(
    text: str,
    *,
    allowed_seeds: frozenset[int] = frozenset(),
) -> None:
    if any(marker in text for marker in _SOURCE_MARKERS):
        raise ValueError("forbidden opponent source material")
    if _FORBIDDEN_MARKER.search(text):
        raise ValueError("forbidden nonlearning partition material")
    for match in _SIX_DIGIT_SEED.finditer(text):
        if int(match.group()) not in allowed_seeds:
            raise ValueError(f"forbidden context seed: {match.group()}")


def validate_round8_static_context(
    *,
    v7_strategy: str,
    v7_experience: str,
    rules_text: str,
    replay_skill_text: str,
) -> None:
    _reject_text(v7_strategy)
    _reject_text(
        v7_experience,
        allowed_seeds=frozenset(ROUND7_LEARNING_SEEDS),
    )
    _reject_text(rules_text)
    _reject_text(replay_skill_text)


def _validate_evidence(
    evidence: Sequence[CriticalLearningEvidence],
) -> tuple[CriticalLearningEvidence, ...]:
    if len(evidence) != 6:
        raise ValueError("round-8 prompt requires exactly six learning episodes")
    pairs = [(item.seed, item.evaluated_seat) for item in evidence]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate round-8 learning seed/seat pair")
    ids = [item.replay_id for item in evidence]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate round-8 learning replay id")
    for item in evidence:
        if item.seed in ROUND8_VALIDATION_SEEDS or item.seed in FROZEN_BEFORE_ROUND8:
            raise ValueError(f"forbidden round-8 evidence seed: {item.seed}")
        if (item.seed, item.evaluated_seat) not in _EPISODE_PAIRS:
            raise ValueError("forbidden round-8 evidence seed/seat pair")
        expected = f"learn8-high-s{item.seed}-p{item.evaluated_seat}"
        if item.replay_id != expected:
            raise ValueError("round-8 evidence replay id is not canonical")
        _reject_text(item.to_json(), allowed_seeds=frozenset(ROUND8_LEARNING_SEEDS))
    if frozenset(pairs) != _EPISODE_PAIRS:
        raise ValueError("round-8 prompt evidence partition is incomplete")
    return tuple(sorted(evidence, key=lambda item: (item.seed, item.evaluated_seat)))


def build_round8_prompt(
    *,
    benchmark_id: str,
    parent_content_hash: str,
    v7_strategy: str,
    v7_experience: str,
    rules_text: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    evidence: Sequence[CriticalLearningEvidence],
    action_profile: Mapping[str, object],
    max_bytes: int = 131_072,
) -> PromptBuildResult:
    if replay_skill_sha256 != ROUND8_REPLAY_SKILL_SHA256:
        raise ValueError("round-8 frozen replay skill digest changed")
    if not re.fullmatch(r"[0-9a-f]{64}", parent_content_hash):
        raise ValueError("round-8 parent content hash is invalid")
    validate_round8_static_context(
        v7_strategy=v7_strategy,
        v7_experience=v7_experience,
        rules_text=rules_text,
        replay_skill_text=replay_skill_text,
    )
    profile_json = json.dumps(action_profile, ensure_ascii=False, sort_keys=True)
    _reject_text(profile_json)
    ordered = _validate_evidence(evidence)
    mandatory = f"""You are performing exactly one coding-agent act for {benchmark_id}.

This is a clean-room v7-to-v8 heuristic-learning iteration. Use only the
editable workspace, official rules, frozen replay Skill, and the six declared
learning episodes below. Do not infer validation or formal cases, benchmark
scores, controlled-policy-KL actions, or any deleted experiment.

Produce deterministic, explainable policy code. You may refactor Python,
search, planning, state evaluation, decision trees, and compressed policy
structure; do not merely append an unbounded rule pile. Preserve main.py and
all harness assets. Update STRATEGY.md as a compact executable specification
and compress EXPERIENCE.md into replay-anchored hypotheses, retained behavior,
rejected ideas, tests, and risks.

Prioritize main-general safety and counter-capture, large-stack consolidation
and attack timing, contact-phase economy/combat allocation, and safe-prefix
behavior for invalid later macro actions. Keep output legal, deterministic,
dual-seat safe, and below the official decision timeout. Run candidate tests.

PARENT VERSION v7
PARENT CONTENT HASH {parent_content_hash}

EDITABLE V7 STRATEGY
{v7_strategy}

RETAINED V7 EXPERIENCE
{v7_experience}

OFFICIAL RULES
{rules_text}

FROZEN HUMAN REPLAY SKILL
sha256={replay_skill_sha256}
{replay_skill_text}

LEARNING-ONLY V7 ACTION PROFILE
{profile_json}

SIX LEARNING-ONLY CRITICAL REPLAYS
"""
    lines = tuple(item.to_json() + "\n" for item in ordered)
    prompt = mandatory + "".join(lines)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_bytes:
        raise ValueError("round-8 prompt context exceeds max_bytes")
    ids = tuple(item.replay_id for item in ordered)
    feedback_bytes = sum(len(line.encode("utf-8")) for line in lines)
    result = PromptBuildResult(
        prompt=prompt,
        included_episode_ids=ids,
        omitted_episode_ids=(),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=False,
        selection_policy="exact_v8_clean_room_learning_only",
        feedback_episodes_read=len(ordered),
        feedback_decision_records_read=sum(len(item.decisions) for item in ordered),
        feedback_serialized_bytes_read=feedback_bytes,
    )
    manifest = {
        "episode_count": len(ordered),
        "episode_ids": list(ids),
        "evidence_decision_ids": [
            decision.state_id for item in ordered for decision in item.decisions
        ],
        "feedback_serialized_bytes": feedback_bytes,
        "forbidden_partitions": [
            "deleted_v8",
            "calibration",
            "controlled_policy_kl",
            "formal",
            "sealed",
            "validation",
        ],
        "learning_seeds": list(ROUND8_LEARNING_SEEDS),
        "omitted_bytes": 0,
        "parent_content_hash": parent_content_hash,
        "parent_version": "v7",
        "prompt_bytes": prompt_bytes,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "provider_act_limit": 1,
        "replay_skill_sha256": replay_skill_sha256,
        "selection_policy": result.selection_policy,
    }
    return replace(result, manifest=manifest)

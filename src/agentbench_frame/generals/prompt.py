"""Leak-resistant prompt construction for the single Codex act."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
import json

from .measurement import DecisionClassSummary
from .replay import (
    CompactLearningEvidence,
    CriticalLearningEvidence,
    LearningReplay,
)


FORBIDDEN_EVALUATION_SEEDS = frozenset(
    {280101, 280202, 280303, 282601, 282702, 282803, 282904, 283005}
)
FORBIDDEN_PROMPT_TEXT = ("top_algorithms/", "advanced-rank02-robinliu-v18")
FORBIDDEN_ROUND4_EVIDENCE_SEEDS = (
    FORBIDDEN_EVALUATION_SEEDS | {284101, 284202, 284303}
)


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    included_episode_ids: tuple[str, ...]
    omitted_episode_ids: tuple[str, ...]
    prompt_bytes: int
    estimated_tokens: int
    truncated: bool
    selection_policy: str = "sorted_whole_records_v1"
    feedback_episodes_read: int = 0
    feedback_decision_records_read: int = 0
    feedback_serialized_bytes_read: int = 0
    manifest: Mapping[str, object] = field(default_factory=dict)


def build_codex_prompt(
    benchmark_id: str,
    strategy_doc: str,
    rules_text: str,
    replay_guide: str,
    learning_replays: Sequence[LearningReplay],
) -> str:
    ordered = sorted(
        learning_replays,
        key=lambda item: (item.seed, item.evaluated_seat, item.opponent_tier),
    )
    evidence = "\n".join(item.to_json() for item in ordered)
    return f"""You are improving the deterministic Generals rule baseline for {benchmark_id}.

Inspect the editable workspace and use only the official-rule summary and the
redacted learning episodes below. You may change only strategy.py,
STRATEGY.md, and files under tests/. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not use randomness, time, network access, opponent-specific
hard-coding, evaluation seeds, or external files.

Run the strategy tests. Make one coherent improvement justified by the
learning evidence, then finish with a concise change summary.

CURRENT STRATEGY
{strategy_doc}

OFFICIAL RULE SUMMARY
{rules_text}

REPLAY FIELD GUIDE
{replay_guide}

REDACTED LEARNING EPISODES
{evidence}
"""


def _reject_leaks(text: str) -> None:
    for seed in FORBIDDEN_EVALUATION_SEEDS:
        if str(seed) in text:
            raise ValueError(f"forbidden evaluation seed in prompt: {seed}")
    for forbidden in FORBIDDEN_PROMPT_TEXT:
        if forbidden in text:
            raise ValueError(f"forbidden evaluation material in prompt: {forbidden}")


def build_round2_prompt(
    benchmark_id: str,
    strategy_doc: str,
    rules_text: str,
    replay_guide: str,
    first_diff: str,
    evidence: Sequence[CompactLearningEvidence],
    max_bytes: int = 262_144,
) -> PromptBuildResult:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.opponent_tier,
            item.seed,
            item.evaluated_seat,
            item.replay_id,
        ),
    )
    for item in ordered:
        if item.seed in FORBIDDEN_EVALUATION_SEEDS:
            raise ValueError(f"forbidden evaluation seed in evidence: {item.seed}")
        if item.opponent_tier == "high":
            raise ValueError("high-tier feedback is held out from Codex")
    mandatory = f"""You are performing the second heuristic-learning act for {benchmark_id}.

Inspect the editable v1 workspace and make one coherent deterministic strategy
improvement. You may change only strategy.py, STRATEGY.md, and files under
tests/. Preserve choose_actions(round_number, my_seat, view) -> list[list[int]]
and the SDK entrypoint. Do not use randomness, time, network access,
opponent-specific hard-coding, evaluation material, or external files.

Run the strategy tests and finish with a concise change summary. Dense values
below are trajectory diagnostics, not reward, policy KL, or benchmark score.

CURRENT STRATEGY
{strategy_doc}

OFFICIAL RULE SUMMARY
{rules_text}

REPLAY FIELD GUIDE
{replay_guide}

FIRST ACT DIFF
{first_diff}

COMPACT LEARNING EPISODES
"""
    _reject_leaks(mandatory)
    if len(mandatory.encode("utf-8")) > max_bytes:
        raise ValueError("mandatory prompt context exceeds max_bytes")
    prompt = mandatory
    included: list[str] = []
    omitted: list[str] = []
    for item in ordered:
        line = item.to_json() + "\n"
        _reject_leaks(line)
        candidate = prompt + line
        if len(candidate.encode("utf-8")) <= max_bytes:
            prompt = candidate
            included.append(item.replay_id)
        else:
            omitted.append(item.replay_id)
    prompt_bytes = len(prompt.encode("utf-8"))
    return PromptBuildResult(
        prompt=prompt,
        included_episode_ids=tuple(included),
        omitted_episode_ids=tuple(omitted),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=bool(omitted),
    )


def build_round3_prompt(
    benchmark_id: str,
    strategy_doc: str,
    rules_text: str,
    replay_guide: str,
    evidence: Sequence[CompactLearningEvidence],
    decision_classes: DecisionClassSummary,
    max_bytes: int = 65_536,
) -> PromptBuildResult:
    """Build compact strongest-human feedback for an architecture-level act."""
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.seed,
            item.evaluated_seat,
            item.replay_id,
        ),
    )
    for item in ordered:
        if item.seed in FORBIDDEN_EVALUATION_SEEDS:
            raise ValueError(
                f"forbidden evaluation seed in evidence: {item.seed}"
            )
        if item.opponent_tier != "high":
            raise ValueError(
                "round-3 evidence must come only from the high-tier learner"
            )
    class_payload = json.dumps(
        {
            "total": decision_classes.total,
            "counts": dict(decision_classes.counts),
            "rates": dict(decision_classes.rates),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    mandatory = f"""You are performing the v2-to-v3 pilot rescue act for {benchmark_id}.

Redesign the deterministic Generals strategy using only the official rules,
the replay guide, and the strongest-human learning summaries below. You are not
limited to a local rule patch. You may delete, merge, or compress old rules and
may implement readable deterministic search, state-derived scoring, planning,
or state machines. Do not use opaque learned weights.

You may edit strategy.py, STRATEGY.md, EXPERIENCE.md, tests/**, and Python
modules under policy/**. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not edit main.py or state_view.py. Do not use randomness, wall
clock, network access, external files, opponent-specific source knowledge,
evaluation material, or frozen evaluation seeds.

Run the strategy tests. Create or update EXPERIENCE.md with replay-ID-backed
observations, retained and rejected hypotheses, the structural v3 changes, and
risks for the next iteration. Finish with a concise change and test summary.

The existing policy is behaviorally stuck: it repeatedly emits army moves from
the main-general coordinate, restarts the same BFS every turn, and leaves its
non-main frontier fallback unreachable. Fix the policy architecture, not only
the fallback body. Dense values are trajectory diagnostics, not reward, policy
KL, information gain, or benchmark score.

CURRENT V2 STRATEGY
{strategy_doc}

OFFICIAL RULE SUMMARY
{rules_text}

REPLAY FIELD GUIDE
{replay_guide}

V2 OBSERVABLE DECISION CLASSES
{class_payload}

COMPACT STRONGEST-HUMAN LEARNING EPISODES
"""
    _reject_leaks(mandatory)
    if len(mandatory.encode("utf-8")) > max_bytes:
        raise ValueError("mandatory prompt context exceeds max_bytes")
    prompt = mandatory
    included: list[str] = []
    omitted: list[str] = []
    for item in ordered:
        line = item.to_json() + "\n"
        _reject_leaks(line)
        candidate = prompt + line
        if len(candidate.encode("utf-8")) <= max_bytes:
            prompt = candidate
            included.append(item.replay_id)
        else:
            omitted.append(item.replay_id)
    prompt_bytes = len(prompt.encode("utf-8"))
    return PromptBuildResult(
        prompt=prompt,
        included_episode_ids=tuple(included),
        omitted_episode_ids=tuple(omitted),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=bool(omitted),
        selection_policy="sorted_high_tier_whole_records_v3",
    )


def build_round4_prompt(
    benchmark_id: str,
    strategy_doc: str,
    experience_text: str,
    rules_text: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    evidence: Sequence[CriticalLearningEvidence],
    decision_classes: DecisionClassSummary,
    max_bytes: int = 65_536,
) -> PromptBuildResult:
    """Build replay-skill-guided strongest-human feedback for v4."""
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.seed,
            item.evaluated_seat,
            item.replay_id,
        ),
    )
    for item in ordered:
        if item.seed in FORBIDDEN_ROUND4_EVIDENCE_SEEDS:
            raise ValueError(
                f"forbidden round-4 evidence seed: {item.seed}"
            )
        if item.opponent_tier != "high":
            raise ValueError(
                "round-4 evidence must come only from the high-tier learner"
            )
    if (
        len(replay_skill_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in replay_skill_sha256
        )
    ):
        raise ValueError("replay skill sha256 must be a lowercase digest")
    class_payload = json.dumps(
        {
            "total": decision_classes.total,
            "counts": dict(decision_classes.counts),
            "rates": dict(decision_classes.rates),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    mandatory = f"""You are performing the v3-to-v4 replay-guided act for {benchmark_id}.

Redesign the deterministic Generals strategy using only the official rules,
the frozen human-authored replay skill, retained v3 experience, and the new
strongest-human critical windows below. The v3 policy expanded from many
stacks, but it often moved all but one army, created fragile territory, failed
to retain or merge reserves, and lost army and economic position. Treat this
as a hypothesis to verify against the cited replay/state evidence.

You may delete, merge, compress, or replace old rules. You may implement
readable deterministic search, state-derived scoring, planning, or state
machines. Prioritize explicit main danger, dynamic reserve requirements,
controlled force concentration and merging, high-value targets, and safe
economic timing. Do not use opaque learned weights.

You may edit strategy.py, STRATEGY.md, EXPERIENCE.md, tests/**, and Python
modules under policy/**. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not edit main.py or state_view.py. Do not use randomness, wall
clock, network access, external files, opponent-specific source knowledge,
evaluation material, or frozen evaluation seeds.

Run the strategy tests. Update EXPERIENCE.md with replay/state-ID-backed
observations, retained and rejected hypotheses, structural v4 changes, and
remaining risks. Compress obsolete or conflicting v3 rules instead of simply
adding another independent threshold. Finish with a concise change and test
summary.

Dense values are trajectory diagnostics, not reward or benchmark score.
Deterministic action disagreement is behavior change, not epistemic information
gain or policy KL.

CURRENT V3 STRATEGY
{strategy_doc}

RETAINED V3 EXPERIENCE
{experience_text}

OFFICIAL RULE SUMMARY
{rules_text}

FROZEN HUMAN REPLAY SKILL
sha256={replay_skill_sha256}
{replay_skill_text}

V3 OBSERVABLE DECISION CLASSES ON NEW FEEDBACK STATES
{class_payload}

CRITICAL STRONGEST-HUMAN LEARNING WINDOWS
"""
    _reject_leaks(mandatory)
    if len(mandatory.encode("utf-8")) > max_bytes:
        raise ValueError("mandatory prompt context exceeds max_bytes")
    prompt = mandatory
    included: list[str] = []
    omitted: list[str] = []
    feedback_decisions = 0
    feedback_bytes = 0
    for item in ordered:
        line = item.to_json() + "\n"
        _reject_leaks(line)
        candidate = prompt + line
        if len(candidate.encode("utf-8")) <= max_bytes:
            prompt = candidate
            included.append(item.replay_id)
            feedback_decisions += len(item.decisions)
            feedback_bytes += len(line.encode("utf-8"))
        else:
            omitted.append(item.replay_id)
    prompt_bytes = len(prompt.encode("utf-8"))
    return PromptBuildResult(
        prompt=prompt,
        included_episode_ids=tuple(included),
        omitted_episode_ids=tuple(omitted),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=bool(omitted),
        selection_policy=(
            "deterministic_critical_windows_whole_episode_v4"
        ),
        feedback_episodes_read=len(included),
        feedback_decision_records_read=feedback_decisions,
        feedback_serialized_bytes_read=feedback_bytes,
    )

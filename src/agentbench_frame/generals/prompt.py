"""Leak-resistant prompt construction for the single Codex act."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .replay import CompactLearningEvidence, LearningReplay


FORBIDDEN_EVALUATION_SEEDS = frozenset(
    {280101, 280202, 280303, 282601, 282702, 282803, 282904, 283005}
)
FORBIDDEN_PROMPT_TEXT = ("top_algorithms/", "advanced-rank02-robinliu-v18")


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    included_episode_ids: tuple[str, ...]
    omitted_episode_ids: tuple[str, ...]
    prompt_bytes: int
    estimated_tokens: int
    truncated: bool
    selection_policy: str = "sorted_whole_records_v1"


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

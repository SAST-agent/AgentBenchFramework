"""High-only, leak-resistant prompt construction for the v6 planner act."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re

from .assets import ROUND5_LEARNING_SEEDS
from .prompt import (
    FORBIDDEN_EVALUATION_SEEDS,
    PromptBuildResult,
    _reject_leaks,
)
from .replay import CriticalLearningEvidence


ROUND6_EVIDENCE_SEEDS = frozenset({287101, 287202, 287303})
FORBIDDEN_ROUND6_EVIDENCE_SEEDS = (
    FORBIDDEN_EVALUATION_SEEDS
    | frozenset(
        {
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
            288101,
            288202,
            288303,
        }
    )
)
_ROUND6_EPISODE_PAIRS = frozenset(
    (seed, seat) for seed in ROUND6_EVIDENCE_SEEDS for seat in (0, 1)
)
_EXTERNAL_CONTEXT_PHASE_WORD = re.compile(
    r"\b(?:formal|validation)\b",
    re.IGNORECASE,
)
_SIX_DIGIT_SEED = re.compile(r"(?<!\d)\d{6}(?!\d)")


def _validate_evidence(
    evidence: Sequence[CriticalLearningEvidence],
) -> tuple[CriticalLearningEvidence, ...]:
    """Reject every learning set except the frozen six high-only episodes."""
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
    seen_episode_ids: set[str] = set()
    for item in ordered:
        if item.seed in FORBIDDEN_ROUND6_EVIDENCE_SEEDS:
            raise ValueError(
                "forbidden round-6 evidence seed: " f"{item.seed}"
            )
        if item.opponent_tier != "high":
            raise ValueError(
                "round-6 evidence must come only from the high-tier learner"
            )
        if item.replay_id in seen_episode_ids:
            raise ValueError(
                "duplicate round-6 evidence episode: " f"{item.replay_id}"
            )
        seen_episode_ids.add(item.replay_id)

    pairs = frozenset((item.seed, item.evaluated_seat) for item in ordered)
    if (
        len(ordered) != len(_ROUND6_EPISODE_PAIRS)
        or pairs != _ROUND6_EPISODE_PAIRS
    ):
        raise ValueError(
            "round-6 evidence must contain exactly six declared high-only "
            "episodes"
        )
    return ordered


def _reject_round6_leaks(text: str) -> None:
    """Protect the prompt boundary beyond the shared formal-evaluation gate."""
    _reject_leaks(text)
    for seed in FORBIDDEN_ROUND6_EVIDENCE_SEEDS:
        if str(seed) in text:
            raise ValueError(f"forbidden round-6 seed in prompt: {seed}")


def _reject_round6_external_context(text: str) -> None:
    """Reject held-out phase material before it can be interpolated."""
    _reject_round6_leaks(text)
    if _EXTERNAL_CONTEXT_PHASE_WORD.search(text):
        raise ValueError("forbidden round-6 formal or validation material")


def _reject_round6_parent_experience(text: str) -> None:
    """Allow only declared round-5 learning citations in parent experience."""
    _reject_leaks(text)
    if _EXTERNAL_CONTEXT_PHASE_WORD.search(text):
        raise ValueError("forbidden round-6 formal or validation material")
    for match in _SIX_DIGIT_SEED.finditer(text):
        seed = int(match.group())
        if seed not in ROUND5_LEARNING_SEEDS:
            raise ValueError(f"forbidden round-6 v5 experience seed: {seed}")


def validate_round6_static_context(
    *,
    v5_strategy: str,
    v5_experience: str,
    rules_text: str,
    replay_skill_text: str,
) -> None:
    """Reject contaminated static inputs before learning games are run."""
    for context in (v5_strategy, rules_text, replay_skill_text):
        _reject_round6_external_context(context)
    _reject_round6_parent_experience(v5_experience)


def build_round6_prompt(
    *,
    benchmark_id: str,
    v5_strategy: str,
    v5_experience: str,
    rules_text: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    evidence: Sequence[CriticalLearningEvidence],
    action_profile: Mapping[str, object],
    max_bytes: int = 131_072,
) -> PromptBuildResult:
    """Build the complete v6 prompt from only frozen high-only learning data."""
    _reject_round6_external_context(benchmark_id)
    if (
        len(replay_skill_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in replay_skill_sha256
        )
    ):
        raise ValueError("replay skill sha256 must be a lowercase digest")

    ordered = _validate_evidence(evidence)
    try:
        action_profile_json = json.dumps(
            dict(action_profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "round-6 action profile must be JSON serializable"
        ) from exc
    validate_round6_static_context(
        v5_strategy=v5_strategy,
        v5_experience=v5_experience,
        rules_text=rules_text,
        replay_skill_text=replay_skill_text,
    )
    _reject_round6_external_context(action_profile_json)

    mandatory = f"""You are performing the v5-to-v6 bounded macro-action planner
act for {benchmark_id}.

The exact editable parent is v5. Redesign only the deterministic policy needed
for a bounded macro-action planner, using the official rules, frozen replay
skill, exact v5 source and experience, action-profile diagnostics, and the six
high-only learning windows below. Do not consult or infer any other gameplay
data.

You may edit strategy.py, state_view.py, STRATEGY.md, EXPERIENCE.md, tests/**,
and files under policy/**. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not edit main.py. Do not read or use opponent identity, seed,
filesystem, network, wall-clock, random, replay-ID, or future-state input.

Keep the planner deterministic and explainable. Generate a macro by applying
each primitive to a cloned normalized state before choosing the next primitive:
deterministic sequential state updates are required. Respect movement-budget
bounds, stop at an uncertain transition, emit no zero/negative movement or
movement from a non-owned stack, and use deterministic row/column and command
tie-breaking. Emit exactly one final [8] and no earlier end command.

Commands 1, 3, and 5 are the required-safe scope: army movement, production /
defense / mobility purchases, and technology purchases only when ownership,
affordability, level, and other official preconditions are known. Commands 2,
4, 6, and 7 are optional only with complete official legality tests and a
normalized-state transition model that establishes their full legality.

Use an ordered hierarchy: main safety first (including actionable adjacent
capture pressure, counter-capture/reinforcement, and an explainable reserve);
then resource/main production economics using exact costs, a coin reserve, and
expected remaining horizon; then valuable routing. Routing must account for
enemy-main, enemy-sub, resource-general, enemy territory, consolidation,
defender cost, attacking surplus, terrain, route cost, and main exposure.
Neutral plain expansion must not beat a reachable higher-value objective except
as a necessary route step.

Run strategy tests. Add focused legality and invariant tests for movement
budgets, sequential transitions, exact one-final-[8], deterministic output,
main safety, and every optional command actually enabled. Update STRATEGY.md
with the compact hierarchy and invariants. Update EXPERIENCE.md with learning
replay-ID/state-ID-backed observations, tested hypotheses, retained v5
behavior, rejected/compressed behavior, and remaining risks.

Dense values and action profiles are diagnostics, not reward, causal
information gain, policy KL, or benchmark score. They may describe behavior
but do not establish a causal effect.

EXACT EDITABLE V5 STRATEGY
{v5_strategy}

RETAINED V5 EXPERIENCE
{v5_experience}

OFFICIAL RULE SUMMARY
{rules_text}

AUTHORITATIVE FROZEN HUMAN REPLAY SKILL
sha256={replay_skill_sha256}
{replay_skill_text}

ACTION-PROFILE DIAGNOSTICS FROM V5 HIGH-ONLY LEARNING
{action_profile_json}

CRITICAL HIGH-ONLY V5 LEARNING WINDOWS
"""

    lines = tuple(item.to_json() + "\n" for item in ordered)
    for line in lines:
        _reject_round6_external_context(line)
    prompt = mandatory + "".join(lines)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_bytes:
        raise ValueError("round-6 prompt context exceeds max_bytes")

    return PromptBuildResult(
        prompt=prompt,
        included_episode_ids=tuple(item.replay_id for item in ordered),
        omitted_episode_ids=(),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=False,
        selection_policy="exact_high_only_whole_episodes_v6",
        feedback_episodes_read=len(ordered),
        feedback_decision_records_read=sum(
            len(item.decisions) for item in ordered
        ),
        feedback_serialized_bytes_read=sum(
            len(line.encode("utf-8")) for line in lines
        ),
    )

"""Learning-only prompt construction for the v7 champion challenge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import re

from .assets import ROUND6_LEARNING_SEEDS
from .challenge_v7 import (
    FROZEN_BEFORE_ROUND7,
    ROUND7_LEARNING_SEEDS,
    ROUND7_SEALED_SEEDS,
    ROUND7_VALIDATION_SEEDS,
)
from .prompt import PromptBuildResult
from .replay import CriticalLearningEvidence


ROUND7_REPLAY_SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)
FORBIDDEN_ROUND7_EVIDENCE_SEEDS = (
    FROZEN_BEFORE_ROUND7
    | frozenset(ROUND7_VALIDATION_SEEDS)
    | frozenset(ROUND7_SEALED_SEEDS)
)
_ROUND7_EPISODE_PAIRS = frozenset(
    (seed, seat)
    for seed in ROUND7_LEARNING_SEEDS
    for seat in (0, 1)
)
_SIX_DIGIT_SEED = re.compile(r"(?<!\d)\d{6}(?!\d)")
_NONLEARNING_PHASE = re.compile(
    r"\b(?:formal|validation|sealed|calibration|controlled[_ -]?policy[_ -]?kl)\b",
    re.IGNORECASE,
)
_CHAMPION_SOURCE_MARKERS = (
    "advanced-rank02-robinliu-v18",
    "top_algorithms/",
)


def _reject_source_material(text: str) -> None:
    for marker in _CHAMPION_SOURCE_MARKERS:
        if marker in text:
            raise ValueError(
                "forbidden round-7 opponent source material in prompt input"
            )


def _reject_nonlearning_context(text: str) -> None:
    _reject_source_material(text)
    if _NONLEARNING_PHASE.search(text):
        raise ValueError("forbidden round-7 nonlearning partition material")
    for seed in FORBIDDEN_ROUND7_EVIDENCE_SEEDS:
        if str(seed) in text:
            raise ValueError(f"forbidden round-7 context seed: {seed}")


def _validate_parent_experience(text: str) -> None:
    _reject_source_material(text)
    for match in _SIX_DIGIT_SEED.finditer(text):
        seed = int(match.group())
        if seed not in ROUND6_LEARNING_SEEDS:
            raise ValueError(
                f"forbidden round-7 v6 experience seed: {seed}"
            )
    if _NONLEARNING_PHASE.search(text):
        raise ValueError("forbidden round-7 nonlearning partition material")


def validate_round7_static_context(
    *,
    v6_strategy: str,
    v6_experience: str,
    rules_text: str,
    replay_skill_text: str,
) -> None:
    """Reject held-out or opponent-source material before learning games."""
    for text in (v6_strategy, rules_text, replay_skill_text):
        _reject_nonlearning_context(text)
    _validate_parent_experience(v6_experience)


def _validate_evidence(
    evidence: Sequence[CriticalLearningEvidence],
) -> tuple[CriticalLearningEvidence, ...]:
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
        if item.seed in FORBIDDEN_ROUND7_EVIDENCE_SEEDS:
            raise ValueError(
                f"forbidden round-7 evidence seed: {item.seed}"
            )
        if item.opponent_tier != "high":
            raise ValueError(
                "round-7 evidence must be strongest-human high-tier data"
            )
        if item.replay_id in seen:
            raise ValueError(
                f"duplicate round-7 evidence episode: {item.replay_id}"
            )
        seen.add(item.replay_id)
        expected_id = (
            f"learn7-high-s{item.seed}-p{item.evaluated_seat}"
        )
        if item.replay_id != expected_id:
            raise ValueError(
                "round-7 evidence episode ID must use the canonical "
                "learning case identity"
            )

    pairs = frozenset((item.seed, item.evaluated_seat) for item in ordered)
    if len(ordered) != 6 or pairs != _ROUND7_EPISODE_PAIRS:
        raise ValueError(
            "round-7 evidence must contain exactly six declared "
            "learning episodes"
        )
    return ordered


def build_round7_prompt(
    *,
    benchmark_id: str,
    v6_strategy: str,
    v6_experience: str,
    rules_text: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    evidence: Sequence[CriticalLearningEvidence],
    action_profile: Mapping[str, object],
    max_bytes: int = 131_072,
) -> PromptBuildResult:
    """Build one v7 act from the exact six learning episodes and no others."""
    _reject_nonlearning_context(benchmark_id)
    if replay_skill_sha256 != ROUND7_REPLAY_SKILL_SHA256:
        raise ValueError(
            "round-7 frozen replay skill digest does not match"
        )
    validate_round7_static_context(
        v6_strategy=v6_strategy,
        v6_experience=v6_experience,
        rules_text=rules_text,
        replay_skill_text=replay_skill_text,
    )
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
            "round-7 action profile must be JSON serializable"
        ) from exc
    _reject_nonlearning_context(action_profile_json)

    mandatory = f"""You are performing the single v6-to-v7 champion-learning
act for {benchmark_id}. Use only the exact editable v6 policy, retained
experience, official rules, frozen Replay Skill, learning-only action profile,
and six declared high-tier learning episodes below. Do not inspect any other
gameplay payload or opponent implementation.

The runtime interface remains
choose_actions(round_number, my_seat, view) -> list[list[int]].
You may edit strategy.py, state_view.py, STRATEGY.md, EXPERIENCE.md, tests/**,
and policy/**. main.py is immutable. Do not use the network, randomness,
filesystem paths, wall-clock time, opponent identity, replay IDs, seeds, or
future state as runtime policy inputs.

Build a deterministic, explainable full planner for commands 1 through 7.
Maintain a private normalized-state clone. Check every primitive's official
preconditions, apply its modeled transition to that clone, then generate the
next primitive. Unknown or partially modeled transitions must stop expansion
and use a verified safe prefix. Keep the existing v6 commands 1/3/5 planner as
that explicit fallback until each new command family passes transition-parity
tests.

Candidate families:
1. Army movement: enemy-main capture, own-main defense, counter-capture,
   neutral/resource-general capture, consolidation, and valuable routing.
2. General movement: move an owned main/sub-general only to a legal owned,
   general-free cell when safety, production coverage, or reach improves.
3. General upgrades: exact production, defense, and mobility level/cost rules
   with reserve and remaining-horizon payback.
4. General skills: surprise attack, rout, command, defense, and weaken with
   exact ownership, cost, cooldown, range, target, and frozen-state checks.
5. Technology: movement, climbing, swamp immunity, and weapon unlock scored
   against current reachability and remaining horizon.
6. Super weapons: legal nuclear, strengthen, transmission, and time-stop
   candidates selected from visible material, generals, pressure, and armies.
7. Recruitment: add a sub-general only on an owned, empty, safe cell when the
   expected value exceeds 50 coins while preserving a documented reserve.

Score successor states lexicographically by terminal main outcome; own-main
safety and forced enemy-main access; productive resource/sub-generals; army
and defensible territory; coin/income/payback; skill/technology/weapon
readiness; then objective distance, terrain, and recapture risk. Use documented
opening, economy, contact, and assault weight sets based only on round and
visible state. All ties use deterministic command and coordinate order.

Use a fixed beam width and documented focused top-k for every command family.
Deduplicate exact successor states. Emit at most eight non-end primitives,
terminate immediately after enemy-main capture or an uncertain transition,
and emit exactly one final [8]. Preserve movement budgets and one army at every
source. The official two-second decision limit is unchanged and fixed replay
probes must demonstrate headroom.

Add official legal/illegal fixtures and transition-parity tests for every
enabled family, especially commands 2, 4, 6, and 7. Add macro-length,
one-final-[8], ownership, affordability, cooldown, deterministic-output,
fallback, and latency tests. Run all candidate tests before finishing.

Rewrite STRATEGY.md as a compact executable specification: beam width, family
top-k, macro limit, evaluator weights by phase, tie order, and fallback.
Compress and integrate EXPERIENCE.md around replay/state-anchored hypotheses,
tests, retained behavior, rejected ideas, and remaining risks; do not append an
unbounded rule pile.

Dense metrics and action profiles are diagnostics. They are not reward,
benchmark score, policy KL, causal information gain, or proof of a champion
win.

EXACT EDITABLE V6 STRATEGY
{v6_strategy}

RETAINED V6 EXPERIENCE
{v6_experience}

OFFICIAL RULE SUMMARY
{rules_text}

AUTHORITATIVE FROZEN HUMAN REPLAY SKILL
sha256={replay_skill_sha256}
{replay_skill_text}

LEARNING-ONLY V6 ACTION PROFILE
{action_profile_json}

SIX CRITICAL V6 LEARNING EPISODES
"""
    lines = tuple(item.to_json() + "\n" for item in ordered)
    for line in lines:
        _reject_source_material(line)
        for seed in FORBIDDEN_ROUND7_EVIDENCE_SEEDS:
            if str(seed) in line:
                raise ValueError(
                    f"forbidden round-7 evidence seed: {seed}"
                )
    prompt = mandatory + "".join(lines)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_bytes:
        raise ValueError("round-7 prompt context exceeds max_bytes")

    included_episode_ids = tuple(item.replay_id for item in ordered)
    feedback_bytes = sum(
        len(line.encode("utf-8"))
        for line in lines
    )
    result = PromptBuildResult(
        prompt=prompt,
        included_episode_ids=included_episode_ids,
        omitted_episode_ids=(),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=False,
        selection_policy="exact_v7_champion_learning_only",
        feedback_episodes_read=len(ordered),
        feedback_decision_records_read=sum(
            len(item.decisions) for item in ordered
        ),
        feedback_serialized_bytes_read=feedback_bytes,
    )
    manifest = {
        "episode_count": len(ordered),
        "episode_ids": list(included_episode_ids),
        "evidence_decision_ids": [
            decision.state_id
            for item in ordered
            for decision in item.decisions
        ],
        "feedback_serialized_bytes": feedback_bytes,
        "forbidden_partitions": [
            "calibration",
            "controlled_policy_kl",
            "historical_formal",
            "sealed",
            "validation",
        ],
        "omitted_bytes": 0,
        "prompt_bytes": prompt_bytes,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "replay_skill_sha256": replay_skill_sha256,
        "selection_policy": result.selection_policy,
    }
    return replace(result, manifest=manifest)

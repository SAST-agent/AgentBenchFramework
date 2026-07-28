"""Paired replay prompt for the rollback-guided v5 coding act."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping, Sequence

from .measurement import DecisionClassSummary
from .prompt import (
    FORBIDDEN_EVALUATION_SEEDS,
    PromptBuildResult,
    _reject_leaks,
)
from .replay import CriticalLearningEvidence


FORBIDDEN_ROUND5_EVIDENCE_SEEDS = (
    FORBIDDEN_EVALUATION_SEEDS
    | {
        281101,
        281202,
        281303,
        284101,
        284202,
        284303,
        285101,
        285202,
        285303,
    }
)


@dataclass(frozen=True)
class VersionedCriticalEvidence:
    version: str
    evidence: CriticalLearningEvidence

    @property
    def episode_id(self) -> str:
        return f"{self.version}:{self.evidence.replay_id}"

    def to_json(self) -> str:
        evidence = self.evidence
        return json.dumps(
            {
                "version": self.version,
                "replay_id": evidence.replay_id,
                "seed": evidence.seed,
                "evaluated_seat": evidence.evaluated_seat,
                "opponent_tier": evidence.opponent_tier,
                "termination_type": evidence.termination_type,
                "outcome": evidence.outcome,
                "dense": _compact_dense(evidence.dense),
                "total_decision_count": evidence.total_decision_count,
                "omitted_decision_count": evidence.omitted_decision_count,
                "decisions": [
                    {
                        "state_id": decision.state_id,
                        "round_number": decision.round_number,
                        "seat": decision.seat,
                        "selection_reasons": list(
                            decision.selection_reasons
                        ),
                        "action": [
                            list(command)
                            for command in decision.action
                        ],
                        "decision_class": decision.decision_class,
                        "features": _compact_features(
                            decision.features
                        ),
                    }
                    for decision in evidence.decisions
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _compact_dense(
    dense: Mapping[str, object],
) -> dict[str, object]:
    """Keep terminal, average, and AUC evidence without repeated extrema."""
    result: dict[str, object] = {}
    for name, value in dense.items():
        if isinstance(value, Mapping):
            result[name] = {
                key: value[key]
                for key in ("terminal", "time_average", "auc")
                if key in value
            }
        else:
            result[name] = value
    return result


def _compact_features(
    features: Mapping[str, object],
) -> dict[str, object]:
    """Project each selected state to bounded, explainable prompt features."""
    result = dict(features)
    if (
        result.get("one_step_reachable_enemy_pressure")
        == result.get("adjacent_enemy_pressure")
    ):
        result.pop("one_step_reachable_enemy_pressure", None)
    stacks = result.get("largest_movable_stacks")
    if isinstance(stacks, Sequence) and not isinstance(
        stacks, (str, bytes)
    ):
        result["largest_movable_stacks"] = [
            {
                key: stack[key]
                for key in (
                    "position",
                    "army",
                    "movable_army",
                    "general_id",
                )
                if key in stack
            }
            for stack in stacks[:2]
            if isinstance(stack, Mapping)
        ]
    targets = result.get("strategic_general_targets")
    if isinstance(targets, Sequence) and not isinstance(
        targets, (str, bytes)
    ):
        result["strategic_general_targets"] = [
            {
                key: target[key]
                for key in (
                    "id",
                    "player",
                    "type",
                    "position",
                    "distance_from_own_main",
                    "cell_army",
                    "produce_level",
                )
                if key in target
            }
            for target in targets[:3]
            if isinstance(target, Mapping)
        ]
    return result


def _class_payload(summary: DecisionClassSummary) -> dict[str, object]:
    return {
        "total": summary.total,
        "counts": dict(summary.counts),
        "rates": dict(summary.rates),
    }


def build_round5_prompt(
    *,
    benchmark_id: str,
    v3_strategy: str,
    v3_experience: str,
    v4_strategy: str,
    v4_experience: str,
    rules_text: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    evidence: Sequence[VersionedCriticalEvidence],
    decision_classes: Mapping[str, DecisionClassSummary],
    dense_deltas: Mapping[str, float | None],
    max_bytes: int = 131_072,
) -> PromptBuildResult:
    """Build complete, version-labeled v3/v4 learning feedback."""
    if set(decision_classes) != {"v3", "v4"}:
        raise ValueError(
            "round-5 decision classes must contain exactly v3 and v4"
        )
    if (
        len(replay_skill_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in replay_skill_sha256
        )
    ):
        raise ValueError("replay skill sha256 must be a lowercase digest")
    version_order = {"v3": 0, "v4": 1}
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.evidence.seed,
            item.evidence.evaluated_seat,
            version_order.get(item.version, 99),
            item.evidence.replay_id,
        ),
    )
    seen: set[str] = set()
    for item in ordered:
        if item.version not in version_order:
            raise ValueError(
                f"round-5 evidence version is invalid: {item.version}"
            )
        if item.evidence.seed in FORBIDDEN_ROUND5_EVIDENCE_SEEDS:
            raise ValueError(
                "forbidden round-5 evidence seed: "
                f"{item.evidence.seed}"
            )
        if item.evidence.opponent_tier != "high":
            raise ValueError(
                "round-5 evidence must come only from the high-tier learner"
            )
        if item.episode_id in seen:
            raise ValueError(
                f"duplicate round-5 evidence episode: {item.episode_id}"
            )
        seen.add(item.episode_id)

    classes_json = json.dumps(
        {
            version: _class_payload(decision_classes[version])
            for version in ("v3", "v4")
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    deltas_json = json.dumps(
        dict(dense_deltas),
        ensure_ascii=False,
        sort_keys=True,
    )
    mandatory = f"""You are performing the rollback-guided v5 act for {benchmark_id}.

The experiment lineage retains v4, but this editable workspace is the exact
historical v3 source. Use only the official rules, frozen human replay skill,
retained v3/v4 source experience, and the new paired strongest-human learning
windows below. The trajectories for v3 and v4 diverge; never treat records
with different state IDs as aligned states.

Preserve v3's active global-stack enumeration while testing one coherent,
explainable reserve calculation. Investigate the learning-data hypothesis that
v4 counted the same adjacent hostile pressure inside both available surplus
and urgent-defense comparison. Do not turn an inconclusive defense calculation
into a broad unconditional end-turn branch.

You may delete, merge, compress, or replace rules. You may implement readable
deterministic search, state-derived scoring, planning, or state machines. Do
not use opaque learned weights.

You may edit strategy.py, STRATEGY.md, EXPERIENCE.md, tests/**, and Python
modules under policy/**. Preserve
choose_actions(round_number, my_seat, view) -> list[list[int]] and the SDK
entrypoint. Do not edit main.py or state_view.py. Do not use randomness, wall
clock, network access, external files, opponent-specific source knowledge, or
evaluation material.

Run strategy tests. Add targeted cases for a defeatable adjacent main threat,
an undefeatable but reinforceable main threat, no duplicate threat accounting,
continued non-main capture/routing, deterministic output, and exactly one
final end command. Update EXPERIENCE.md with replay/state-ID-backed retained
and rejected hypotheses, v5 structural changes, and remaining risks. Compress
conflicting rules rather than stacking another threshold.

Dense values are trajectory diagnostics, not reward. Deterministic action
disagreement is behavior change, not epistemic information gain or policy KL.

CURRENT ROLLBACK V3 STRATEGY
{v3_strategy}

RETAINED V3 EXPERIENCE
{v3_experience}

FAILED V4 CANDIDATE STRATEGY
{v4_strategy}

RETAINED V4 LEARNING EXPERIENCE
{v4_experience}

OFFICIAL RULE SUMMARY
{rules_text}

FROZEN HUMAN REPLAY SKILL
sha256={replay_skill_sha256}
{replay_skill_text}

V3/V4 OBSERVABLE DECISION CLASSES ON NEW LEARNING STATES
{classes_json}

V4 MINUS V3 PAIRED LEARNING DENSE DELTAS
{deltas_json}

VERSION-LABELED CRITICAL STRONGEST-HUMAN LEARNING WINDOWS
"""
    _reject_leaks(mandatory)
    if len(mandatory.encode("utf-8")) > max_bytes:
        raise ValueError("mandatory round-5 prompt context exceeds max_bytes")

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
            included.append(item.episode_id)
            feedback_decisions += len(item.evidence.decisions)
            feedback_bytes += len(line.encode("utf-8"))
        else:
            omitted.append(item.episode_id)

    prompt_bytes = len(prompt.encode("utf-8"))
    return PromptBuildResult(
        prompt=prompt,
        included_episode_ids=tuple(included),
        omitted_episode_ids=tuple(omitted),
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=bool(omitted),
        selection_policy=(
            "paired_v3_v4_critical_windows_whole_episode_v5"
        ),
        feedback_episodes_read=len(included),
        feedback_decision_records_read=feedback_decisions,
        feedback_serialized_bytes_read=feedback_bytes,
    )

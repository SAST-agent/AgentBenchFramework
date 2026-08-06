"""Leak-safe attribution-guided prompt for the single v9 coding-agent act."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .challenge_v9 import (
    ROUND9_ATTRIBUTION_SEEDS,
    ROUND9_PARENT_HASH,
    ROUND9_PREDECESSOR_HASH,
    ROUND9_VALIDATION_SEEDS,
)
from .measurement_state import measurement_state_id
from .prompt import PromptBuildResult


ROUND9_PROMPT_MAX_BYTES = 196_608
_CELLS = ("A", "B", "C", "D")
_EXPECTED_PAIRS = tuple(
    f"s{seed}-p{seat}"
    for seed in ROUND9_ATTRIBUTION_SEEDS
    for seat in (0, 1)
)
_QUALITY_DEFECTS = (
    "malformed_lines",
    "invalid_events",
    "unknown_event_types",
    "duplicate_event_ids",
    "missing_event_ids",
    "missing_run_ids",
)
_FORBIDDEN = re.compile(
    r"(?:eval-high|formal[-_/ ]?spec|formal_score|benchmark_score|"
    r"controlled[_ -]?reference[_ -]?policy[_ -]?kl|"
    r"versions/v8/source|validation[-_/ ]?spec|policy[_ -]?kl)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Round9PromptReceipt:
    prompt_sha256: str
    prompt_bytes: int
    diagnosis_report_sha256: str
    diagnosis_evidence_sha256: str
    included_replay_ids: tuple[str, ...]
    included_state_ids: tuple[str, ...]
    policy_parent_hash: str
    provider_act_limit: int = 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _reject_forbidden(text: str, label: str) -> None:
    match = _FORBIDDEN.search(text)
    if match:
        raise ValueError(f"forbidden {label} material: {match.group()}")
    for seed in ROUND9_VALIDATION_SEEDS:
        if str(seed) in text:
            raise ValueError(f"forbidden validation seed in {label}: {seed}")


def _digest_matches(text: str, digest: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} hash must be lowercase sha256")
    if _sha256_text(text) != digest:
        raise ValueError(f"{label} hash does not match content")


def _policy_hashes(
    report: Mapping[str, Any],
    expected: Mapping[str, str],
) -> dict[str, str]:
    if set(expected) != set(_CELLS):
        raise ValueError("expected policy hashes must contain A, B, C, and D")
    if expected["A"] != ROUND9_PARENT_HASH:
        raise ValueError("cell A must be the frozen v7 policy parent")
    if expected["D"] != ROUND9_PREDECESSOR_HASH:
        raise ValueError("cell D must be the frozen v8 predecessor")
    policies = report.get("policies")
    if not isinstance(policies, list) or len(policies) != 4:
        raise ValueError("attribution report must contain four policies")
    observed: dict[str, str] = {}
    for policy in policies:
        if not isinstance(policy, Mapping):
            raise ValueError("attribution policy must be an object")
        cell = str(policy.get("cell"))
        digest = str(policy.get("content_hash"))
        if cell in observed or cell not in _CELLS:
            raise ValueError("attribution policy cells must be unique A/B/C/D")
        observed[cell] = digest
    if observed != {cell: expected[cell] for cell in _CELLS}:
        raise ValueError("attribution policy hashes changed")
    return observed


def _validate_cases(evidence: Mapping[str, Any]) -> set[str]:
    cases = evidence.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("attribution evidence must contain exactly 12 cases")
    pairs = []
    case_ids = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("attribution case must be an object")
        metadata = case.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("phase") != "attribute9":
            raise ValueError("non-attribution case in diagnosis evidence")
        pair = str(metadata.get("pair_id"))
        expected = f"s{int(case['seed'])}-p{int(case['first_player'])}"
        if pair != expected:
            raise ValueError("attribution case pair ID changed")
        pairs.append(pair)
        case_ids.add(str(case["case_id"]))
    if tuple(sorted(pairs)) != tuple(sorted(_EXPECTED_PAIRS)):
        raise ValueError("attribution evidence pair IDs changed")
    if len(case_ids) != 12:
        raise ValueError("duplicate attribution case ID")
    return case_ids


def _validate_states(
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    case_ids: set[str],
) -> tuple[dict[str, Any], ...]:
    raw_states = evidence.get("diagnostic_states")
    report_states = report.get("diagnostic_states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("diagnostic states are missing")
    if raw_states != report_states:
        raise ValueError("report and evidence diagnostic states differ")
    states = []
    seen = set()
    for raw in raw_states:
        if not isinstance(raw, dict):
            raise ValueError("diagnostic state must be an object")
        state_id = str(raw.get("measurement_state_id"))
        if state_id in seen:
            raise ValueError("duplicate diagnostic state ID")
        seen.add(state_id)
        snapshot = raw.get("measurement_state")
        if not isinstance(snapshot, Mapping) or measurement_state_id(snapshot) != state_id:
            raise ValueError("diagnostic state hash changed")
        if str(raw.get("pair_id")) not in _EXPECTED_PAIRS:
            raise ValueError("diagnostic state pair ID is not selected")
        replay = str(raw.get("replay_ref"))
        parts = Path(replay).parts
        if (
            len(parts) != 4
            or parts[0] != "matches"
            or parts[1] not in {"A", "D"}
            or parts[2] not in case_ids
            or parts[3] != "replay.jsonl"
        ):
            raise ValueError("diagnostic replay is not a selected attribution replay")
        states.append(raw)
    if len(states) > 48:
        raise ValueError("diagnostic state count exceeds 48")
    return tuple(states)


def _validate_probes(
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    state_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    raw_probes = evidence.get("diagnostic_probes")
    if not isinstance(raw_probes, list) or raw_probes != report.get("diagnostic_probes"):
        raise ValueError("report and evidence diagnostic probes differ")
    by_state = {state_id: [] for state_id in state_ids}
    for raw in raw_probes:
        if not isinstance(raw, dict):
            raise ValueError("diagnostic probe must be an object")
        state_id = str(raw.get("measurement_state_id"))
        if state_id not in by_state:
            raise ValueError("probe references an unselected diagnostic state")
        by_state[state_id].append(raw)
    for state_id, probes in by_state.items():
        if {str(probe.get("policy_cell")) for probe in probes} != set(_CELLS):
            raise ValueError(f"state {state_id} is missing A/B/C/D probes")
        if len(probes) != 4 or any(
            probe.get("status") != "complete"
            or probe.get("deterministic") is not True
            or probe.get("legal") is not True
            for probe in probes
        ):
            raise ValueError("diagnostic probes must be deterministic and legal")
        probes.sort(key=lambda item: _CELLS.index(str(item["policy_cell"])))
    return by_state


def _effect_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    attribution = report.get("attribution")
    if not isinstance(attribution, Mapping):
        raise ValueError("attribution effects are missing")
    pairs = attribution.get("pair_ids")
    if tuple(sorted(str(item) for item in pairs or ())) != tuple(sorted(_EXPECTED_PAIRS)):
        raise ValueError("attribution report pair IDs changed")
    metrics = attribution.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("attribution metrics are missing")
    return {
        str(metric): {
            key: effect.get(key)
            for key in (
                "large_stack",
                "contact",
                "interaction",
                "complete_pair_count",
                "intervals",
            )
        }
        for metric, effect in metrics.items()
        if isinstance(effect, Mapping)
    }


def round9_prompt_receipt(result: PromptBuildResult) -> Round9PromptReceipt:
    manifest = result.manifest
    return Round9PromptReceipt(
        prompt_sha256=str(manifest["prompt_sha256"]),
        prompt_bytes=int(manifest["prompt_bytes"]),
        diagnosis_report_sha256=str(manifest["diagnosis_report_sha256"]),
        diagnosis_evidence_sha256=str(manifest["diagnosis_evidence_sha256"]),
        included_replay_ids=tuple(manifest["included_replay_ids"]),
        included_state_ids=tuple(manifest["included_state_ids"]),
        policy_parent_hash=str(manifest["policy_parent_hash"]),
        provider_act_limit=int(manifest["provider_act_limit"]),
    )


def build_round9_prompt(
    *,
    benchmark_id: str,
    attribution_run_dir: Path,
    policy_parent_hash: str,
    expected_policy_hashes: Mapping[str, str],
    v7_strategy: str,
    v7_experience: str,
    rules_text: str,
    rules_sha256: str,
    replay_skill_text: str,
    replay_skill_sha256: str,
    max_bytes: int = ROUND9_PROMPT_MAX_BYTES,
) -> PromptBuildResult:
    """Build one auditable prompt from v7 and the frozen attribution report."""

    if type(max_bytes) is not int or max_bytes <= 0 or max_bytes > ROUND9_PROMPT_MAX_BYTES:
        raise ValueError("round-9 prompt maximum is 196608 bytes")
    if policy_parent_hash != ROUND9_PARENT_HASH:
        raise ValueError("round-9 policy parent must be the frozen v7 hash")
    _digest_matches(rules_text, rules_sha256, "rules")
    _digest_matches(replay_skill_text, replay_skill_sha256, "replay skill")
    for label, text in (
        ("v7 strategy", v7_strategy),
        ("v7 experience", v7_experience),
        ("rules", rules_text),
        ("replay skill", replay_skill_text),
    ):
        _reject_forbidden(text, label)

    run_dir = Path(attribution_run_dir).resolve()
    summary = _read_object(run_dir / "summary.json", "attribution summary")
    if summary.get("status") != "complete":
        raise ValueError("attribution run must be complete")
    if summary.get("run_id") != run_dir.name:
        raise ValueError("attribution run identity changed")
    if (
        summary.get("policy_order") != list(_CELLS)
        or summary.get("case_count_per_policy") != 12
        or summary.get("valid_case_count") != 48
        or summary.get("coding_agent_act_count") != 0
        or summary.get("formal_benchmark_opened") is not False
    ):
        raise ValueError("attribution run boundary counters changed")
    quality = summary.get("event_quality")
    if not isinstance(quality, Mapping) or any(quality.get(key) != 0 for key in _QUALITY_DEFECTS):
        raise ValueError("attribution run has event quality defects")

    report_path = run_dir / "diagnosis/report.json"
    evidence_path = run_dir / "diagnosis/evidence.json"
    report_hash = _sha256_bytes(report_path.read_bytes())
    evidence_hash = _sha256_bytes(evidence_path.read_bytes())
    if (
        summary.get("report_hash") != report_hash
        or summary.get("diagnosis_report_hash") != report_hash
        or summary.get("diagnosis_evidence_hash") != evidence_hash
    ):
        raise ValueError("attribution diagnosis hash changed")
    report = _read_object(report_path, "attribution report")
    evidence = _read_object(evidence_path, "attribution evidence")
    if (
        report.get("status") != "complete"
        or report.get("formal_benchmark_opened") is not False
        or report.get("coding_agent_act_count") != 0
    ):
        raise ValueError("attribution report boundary changed")
    policy_hashes = _policy_hashes(report, expected_policy_hashes)
    case_ids = _validate_cases(evidence)
    states = _validate_states(report, evidence, case_ids)
    state_ids = {str(state["measurement_state_id"]) for state in states}
    probes = _validate_probes(report, evidence, state_ids)
    effects = _effect_summary(report)

    safe_attribution = json.dumps(
        {
            "estimands": ["B-A", "C-A", "D-B-C+A"],
            "metrics": effects,
            "policy_hashes": policy_hashes,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    mandatory = f"""You are performing exactly one coding-agent act for {benchmark_id}.

The editable policy parent is the frozen v7 workspace. The chronological
predecessor is v8, but no v8 source file is available or permitted. The
scientific-attribution results below are diagnostic evidence, not reward proof.
All validation, formal-benchmark, and policy-KL partitions are sealed until
after this act. Do not infer or request them.

Produce one deterministic, explainable v9 policy reconstruction. You may use
bounded search, scoring functions, finite-state machines, decision trees, and
coherent refactoring—not only if/else. Compress and integrate the policy;
do not append an unbounded rule pile. Replay IDs, seeds, state IDs, opponent
identity, filesystem paths, clock time, environment variables, randomness,
network, and subprocess results must never become runtime policy features.

Preserve the official SDK entrypoint and protected harness files. Keep the
policy dual-seat safe, deterministic, legal, and within the decision latency
limit. Run tests. Update STRATEGY.md and EXPERIENCE.md with retained lessons,
rejected hypotheses, new evidence-backed lessons, remaining risks, and the
compressed policy structure.

POLICY PARENT VERSION v7
POLICY PARENT HASH {policy_parent_hash}
ITERATION PREDECESSOR VERSION v8
PROVIDER ACT LIMIT 1

EDITABLE V7 STRATEGY
{v7_strategy}

RETAINED V7 EXPERIENCE
{v7_experience}

OFFICIAL RULES sha256={rules_sha256}
{rules_text}

HUMAN REPLAY SKILL sha256={replay_skill_sha256}
{replay_skill_text}

PAIRED ATTRIBUTION EFFECTS (DIAGNOSTIC, NOT CAUSAL PROOF)
{safe_attribution}

SELECTED SAME-STATE DIAGNOSTICS (FIXED PRIORITY ORDER)
"""
    if len(mandatory.encode("utf-8")) > max_bytes:
        raise ValueError("mandatory round-9 prompt exceeds max_bytes")

    prompt = mandatory
    included_states: list[str] = []
    omitted_states: list[str] = []
    included_replays: list[str] = []
    omitted_replays: list[str] = []
    for index, state in enumerate(states):
        state_id = str(state["measurement_state_id"])
        replay_id = str(state["replay_ref"])
        record = {
            "pair_id": state["pair_id"],
            "source_version": state["source_version"],
            "selection_reason": state["reason"],
            "round_number": state["round_number"],
            "actor": state["actor"],
            "measurement_state_id": state_id,
            "replay_ref": replay_id,
            "measurement_state": state["measurement_state"],
            "same_state_actions": [
                {
                    "policy_cell": probe["policy_cell"],
                    "canonical_action": probe["canonical_action"],
                }
                for probe in probes[state_id]
            ],
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        if len((prompt + line).encode("utf-8")) > max_bytes:
            tail = states[index:]
            omitted_states.extend(str(item["measurement_state_id"]) for item in tail)
            omitted_replays.extend(str(item["replay_ref"]) for item in tail)
            break
        prompt += line
        included_states.append(state_id)
        included_replays.append(replay_id)

    prompt_bytes = len(prompt.encode("utf-8"))
    prompt_hash = _sha256_text(prompt)
    included_replay_ids = tuple(dict.fromkeys(included_replays))
    omitted_replay_ids = tuple(
        replay for replay in dict.fromkeys(omitted_replays)
        if replay not in included_replay_ids
    )
    result = PromptBuildResult(
        prompt=prompt,
        included_episode_ids=included_replay_ids,
        omitted_episode_ids=omitted_replay_ids,
        prompt_bytes=prompt_bytes,
        estimated_tokens=(prompt_bytes + 3) // 4,
        truncated=bool(omitted_states),
        selection_policy="v9_fixed_diagnostic_priority_tail_omission",
        feedback_episodes_read=len(included_replay_ids),
        feedback_decision_records_read=len(included_states),
        feedback_serialized_bytes_read=prompt_bytes - len(mandatory.encode("utf-8")),
    )
    manifest = {
        "attribution_run_dir": str(run_dir),
        "attribution_run_id": run_dir.name,
        "policy_parent_version": "v7",
        "policy_parent_hash": policy_parent_hash,
        "iteration_predecessor_version": "v8",
        "iteration_predecessor_hash": ROUND9_PREDECESSOR_HASH,
        "provider_act_limit": 1,
        "diagnosis_report_sha256": report_hash,
        "diagnosis_evidence_sha256": evidence_hash,
        "rules_sha256": rules_sha256,
        "replay_skill_sha256": replay_skill_sha256,
        "policy_hashes": policy_hashes,
        "included_replay_ids": list(included_replay_ids),
        "omitted_replay_ids": list(omitted_replay_ids),
        "included_state_ids": included_states,
        "omitted_state_ids": omitted_states,
        "selection_policy": result.selection_policy,
        "prompt_bytes": prompt_bytes,
        "prompt_sha256": prompt_hash,
        "truncated": result.truncated,
        "sealed_partitions": ["validation", "formal", "policy_kl"],
    }
    return replace(result, manifest=manifest)

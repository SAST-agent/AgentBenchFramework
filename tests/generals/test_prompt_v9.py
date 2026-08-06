import hashlib
import json
from pathlib import Path

import pytest

from agentbench_frame.generals.challenge_v9 import (
    ROUND9_ATTRIBUTION_SEEDS,
    ROUND9_PARENT_HASH,
    ROUND9_PREDECESSOR_HASH,
)


SKILL_TEXT = "human-authored replay field guide"
RULES_TEXT = "official Generals rules"
SKILL_HASH = hashlib.sha256(SKILL_TEXT.encode()).hexdigest()
RULES_HASH = hashlib.sha256(RULES_TEXT.encode()).hexdigest()
POLICY_HASHES = {
    "A": ROUND9_PARENT_HASH,
    "B": "b" * 64,
    "C": "c" * 64,
    "D": ROUND9_PREDECESSOR_HASH,
}


def _attribution_run(root: Path) -> Path:
    run = root / "runs/28_generals/generals-scientific-attribution/attribute-run"
    diagnosis = run / "diagnosis"
    diagnosis.mkdir(parents=True)
    pair_ids = [
        f"s{seed}-p{seat}"
        for seed in ROUND9_ATTRIBUTION_SEEDS
        for seat in (0, 1)
    ]
    states = []
    for index, pair_id in enumerate(pair_ids):
        measurement_state = {
            "schema": "generals-measurement-state-v1",
            "actor": int(pair_id[-1]),
            "state": {"round": index + 1},
        }
        from agentbench_frame.generals.measurement_state import measurement_state_id
        states.append({
            "pair_id": pair_id,
            "source_version": "v7",
            "reason": "first_enemy_contact",
            "round_number": index + 1,
            "actor": int(pair_id[-1]),
            "measurement_state_id": measurement_state_id(measurement_state),
            "measurement_state": measurement_state,
            "replay_ref": f"matches/A/attribute9-high-human-{pair_id}/replay.jsonl",
        })
    probes = [
        {
            "measurement_state_id": state["measurement_state_id"],
            "policy_cell": cell,
            "status": "complete",
            "deterministic": True,
            "canonical_action": [[8]],
            "legal": True,
        }
        for state in states
        for cell in ("A", "B", "C", "D")
    ]
    policies = [
        {
            "cell": cell,
            "policy_id": f"policy-{cell}",
            "content_hash": POLICY_HASHES[cell],
            "interventions": [],
            "source_authority": "v7" if cell != "D" else "v8",
        }
        for cell in ("A", "B", "C", "D")
    ]
    report = {
        "schema": "generals-scientific-attribution-v1",
        "status": "complete",
        "formal_benchmark_opened": False,
        "coding_agent_act_count": 0,
        "policies": policies,
        "attribution": {
            "pair_ids": pair_ids,
            "metrics": {
                "outcome_score": {
                    "large_stack": 0.25,
                    "contact": 0.0,
                    "interaction": -0.25,
                    "complete_pair_count": 12,
                    "intervals": {
                        "large_stack": [0.0, 0.5],
                        "contact": [0.0, 0.0],
                        "interaction": [-0.5, 0.0],
                    },
                }
            },
        },
        "diagnostic_states": states,
        "diagnostic_probes": probes,
        "artifacts": {"evidence": "diagnosis/evidence.json"},
    }
    cases = [
        {
            "case_id": f"attribute9-high-human-{pair_id}",
            "seed": seed,
            "first_player": seat,
            "metadata": {"pair_id": pair_id, "phase": "attribute9"},
        }
        for seed in ROUND9_ATTRIBUTION_SEEDS
        for seat in (0, 1)
        for pair_id in (f"s{seed}-p{seat}",)
    ]
    evidence = {
        "challenge_id": "generals-hl-v9-scientific-attribution-v1",
        "cases": cases,
        "dense_episode_summaries": {},
        "diagnostic_states": states,
        "diagnostic_probes": probes,
        "diagnostic_missing_reasons": {},
    }
    report_path = diagnosis / "report.json"
    evidence_path = diagnosis / "evidence.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (run / "summary.json").write_text(json.dumps({
        "run_id": run.name,
        "status": "complete",
        "policy_order": ["A", "B", "C", "D"],
        "case_count_per_policy": 12,
        "valid_case_count": 48,
        "diagnostic_state_count": 12,
        "coding_agent_act_count": 0,
        "formal_benchmark_opened": False,
        "report_hash": report_hash,
        "diagnosis_report_hash": report_hash,
        "diagnosis_evidence_hash": evidence_hash,
        "event_quality": {
            "malformed_lines": 0,
            "invalid_events": 0,
            "unknown_event_types": 0,
            "duplicate_event_ids": 0,
            "missing_event_ids": 0,
            "missing_run_ids": 0,
        },
    }))
    return run


def _context(tmp_path: Path, **overrides):
    values = {
        "benchmark_id": "generals-hl-pilot-v1",
        "attribution_run_dir": _attribution_run(tmp_path),
        "policy_parent_hash": ROUND9_PARENT_HASH,
        "expected_policy_hashes": POLICY_HASHES,
        "v7_strategy": "deterministic v7 strategy",
        "v7_experience": "retained v7 experience",
        "rules_text": RULES_TEXT,
        "rules_sha256": RULES_HASH,
        "replay_skill_text": SKILL_TEXT,
        "replay_skill_sha256": SKILL_HASH,
        "max_bytes": 196_608,
    }
    values.update(overrides)
    return values


def test_v9_prompt_uses_attribution_and_v7_but_not_v8_source(tmp_path):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    result = build_round9_prompt(**_context(tmp_path))

    assert result.manifest["policy_parent_version"] == "v7"
    assert result.manifest["iteration_predecessor_version"] == "v8"
    assert result.manifest["provider_act_limit"] == 1
    assert result.manifest["diagnosis_report_sha256"] == json.loads(
        (Path(result.manifest["attribution_run_dir"]) / "summary.json").read_text()
    )["report_hash"]
    assert "versions/v8/source" not in result.prompt
    assert "formal_score" not in result.prompt
    assert "exactly one coding-agent act" in result.prompt
    assert "explainable" in result.prompt.lower()
    assert "STRATEGY.md" in result.prompt
    assert "EXPERIENCE.md" in result.prompt
    assert result.manifest["included_state_ids"]
    assert len(result.prompt.encode()) <= 196_608


@pytest.mark.parametrize("forbidden", [
    "303101",
    "eval-high",
    "formal-spec",
    "controlled_reference_policy_kl",
    "versions/v8/source/strategy.py",
    "benchmark_score",
])
def test_v9_prompt_rejects_sealed_or_nonparent_material(tmp_path, forbidden):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    with pytest.raises(ValueError, match="forbidden"):
        build_round9_prompt(**_context(tmp_path, v7_strategy=forbidden))


def test_v9_prompt_rejects_incomplete_or_altered_attribution(tmp_path):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    context = _context(tmp_path)
    run = Path(context["attribution_run_dir"])
    summary = json.loads((run / "summary.json").read_text())
    summary["status"] = "incomplete"
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete"):
        build_round9_prompt(**context)

    context = _context(tmp_path / "altered")
    run = Path(context["attribution_run_dir"])
    (run / "diagnosis/report.json").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        build_round9_prompt(**context)


def test_v9_prompt_rejects_duplicate_state_or_unselected_replay(tmp_path):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    context = _context(tmp_path)
    run = Path(context["attribution_run_dir"])
    evidence_path = run / "diagnosis/evidence.json"
    report_path = run / "diagnosis/report.json"
    evidence = json.loads(evidence_path.read_text())
    report = json.loads(report_path.read_text())
    evidence["diagnostic_states"].append(evidence["diagnostic_states"][0])
    report["diagnostic_states"].append(report["diagnostic_states"][0])
    evidence_path.write_text(json.dumps(evidence) + "\n")
    report_path.write_text(json.dumps(report) + "\n")
    summary = json.loads((run / "summary.json").read_text())
    summary["diagnosis_evidence_hash"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    summary["diagnosis_report_hash"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    summary["report_hash"] = summary["diagnosis_report_hash"]
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="duplicate"):
        build_round9_prompt(**context)

    context = _context(tmp_path / "replay")
    run = Path(context["attribution_run_dir"])
    evidence_path = run / "diagnosis/evidence.json"
    report_path = run / "diagnosis/report.json"
    evidence = json.loads(evidence_path.read_text())
    report = json.loads(report_path.read_text())
    evidence["diagnostic_states"][0]["replay_ref"] = "matches/A/unselected/replay.jsonl"
    report["diagnostic_states"][0]["replay_ref"] = "matches/A/unselected/replay.jsonl"
    evidence_path.write_text(json.dumps(evidence) + "\n")
    report_path.write_text(json.dumps(report) + "\n")
    summary = json.loads((run / "summary.json").read_text())
    summary["diagnosis_evidence_hash"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    summary["diagnosis_report_hash"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    summary["report_hash"] = summary["diagnosis_report_hash"]
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="replay"):
        build_round9_prompt(**context)


def test_v9_prompt_rejects_wrong_source_static_hash_or_size(tmp_path):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    with pytest.raises(ValueError, match="parent"):
        build_round9_prompt(**_context(tmp_path / "parent", policy_parent_hash="0" * 64))
    with pytest.raises(ValueError, match="rules"):
        build_round9_prompt(**_context(tmp_path / "rules", rules_sha256="0" * 64))
    with pytest.raises(ValueError, match="replay skill"):
        build_round9_prompt(**_context(tmp_path / "skill", replay_skill_sha256="0" * 64))
    with pytest.raises(ValueError, match="maximum"):
        build_round9_prompt(**_context(tmp_path / "size", max_bytes=196_609))


def test_v9_prompt_records_deterministic_tail_omission(tmp_path):
    from agentbench_frame.generals.prompt_v9 import build_round9_prompt

    roomy = build_round9_prompt(**_context(tmp_path / "roomy"))
    compact = build_round9_prompt(**_context(tmp_path / "compact", max_bytes=5_000))

    assert roomy.truncated is False
    assert compact.truncated is True
    assert compact.omitted_episode_ids
    assert compact.manifest["omitted_state_ids"]
    assert len(compact.prompt.encode()) <= 5_000

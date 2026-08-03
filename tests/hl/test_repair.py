import json
from pathlib import Path

import pytest


def _result(*, version_id, branch_index, status="complete", margin=-100):
    from agentbench_frame.hl.codebase import Version
    from agentbench_frame.hl.controller import CandidateResult
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.tracking.provider import ProviderInvocation

    version = Version(
        version_id=version_id,
        content_hash=f"hash-{version_id}",
        parent_version_id="v000000",
        act_id=f"act-{version_id}",
        edit_type="candidate",
        created_at="2026-08-02T00:00:00.000Z",
        files=("ai.py",),
    )
    evaluation = CandidateEvaluation(
        status=status,
        score=0.0 if status == "complete" else None,
        error=None if status == "complete" else status,
        matches=(
            {
                "status": "complete",
                "seed": 101,
                "opponent": "rank15",
                "result": "loss",
                "rollman_score": 400 + margin,
                "ghosts_score": 400,
                "replay": f"/matches/{version_id}/replay.jsonl",
                "trace": f"/matches/{version_id}/trace.jsonl",
            },
        )
        if status == "complete"
        else (),
    )
    return CandidateResult(
        act_id=version.act_id,
        branch_index=branch_index,
        version=version,
        evaluation=evaluation,
        provider=ProviderInvocation(
            status="completed" if status == "complete" else status
        ),
    )


def _brief(branch_index=1):
    from agentbench_frame.hl.proposal import BranchBrief

    return BranchBrief(
        branch_index=branch_index,
        diagnosis="level 3 round 26 capture",
        mechanism="bounded route recovery",
        activation_condition="shield broke and the next edge was recently dangerous",
        preservation_contract="ordinary portal and safety routing stays unchanged",
        expected_change="avoid the repeated capture edge",
        falsifier="the same-seed score margin does not improve",
        code_symbols=("ai_func", "helper"),
    )


def test_repair_packet_pairs_parent_and_candidate_on_same_seed(tmp_path):
    from agentbench_frame.hl.repair import build_repair_packet

    parent = _result(version_id="v000037", branch_index=-1, margin=-195)
    candidate = _result(version_id="v000041", branch_index=1, margin=-295)

    parent_summary = tmp_path / "parent-summary.md"
    candidate_summary = tmp_path / "candidate-summary.md"
    parent_summary.write_text("parent evidence", encoding="utf-8")
    candidate_summary.write_text("candidate evidence", encoding="utf-8")

    path = build_repair_packet(
        output_path=tmp_path / "repair.json",
        iteration_id="iter-000012",
        branch_brief=_brief(),
        parent=parent,
        candidate=candidate,
        summary_resolver=lambda match: {
            "summary": str(
                parent_summary if "v000037" in match["replay"] else candidate_summary
            ),
            "replay": match["replay"],
            "trace": match["trace"],
        },
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "1.0"
    assert value["iteration_id"] == "iter-000012"
    assert value["scope"]["activation_condition"].startswith("shield broke")
    assert value["parent"]["version_id"] == "v000037"
    assert value["candidate"]["version_id"] == "v000041"
    assert value["parent"]["matches"][0]["seed"] == 101
    assert value["candidate"]["matches"][0]["seed"] == 101
    assert value["candidate"]["matches"][0]["summary"].endswith("summary.md")
    assert value["parent"]["matches"][0]["summary_text"] == "parent evidence"
    assert value["candidate"]["matches"][0]["summary_text"] == "candidate evidence"


def test_repair_packet_bounds_inline_summary_text(tmp_path):
    from agentbench_frame.hl.repair import build_repair_packet

    summary = tmp_path / "summary.md"
    summary.write_text("x" * 13000, encoding="utf-8")
    path = build_repair_packet(
        output_path=tmp_path / "repair.json",
        iteration_id="iter-000012",
        branch_brief=_brief(),
        parent=_result(version_id="v000037", branch_index=-1),
        candidate=_result(version_id="v000041", branch_index=1),
        summary_resolver=lambda match: {"summary": str(summary)},
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    text = value["candidate"]["matches"][0]["summary_text"]
    assert len(text) <= 12032
    assert text.endswith("[summary truncated]")


def test_repair_packet_includes_candidate_activation_evidence(tmp_path):
    from dataclasses import replace

    from agentbench_frame.hl.repair import build_repair_packet

    candidate = replace(
        _result(version_id="v000041", branch_index=1),
        activation={
            "status": "complete",
            "decision_count": 100,
            "changed_action_count": 3,
            "changed_fraction": 0.03,
            "episodes": [],
        },
    )
    path = build_repair_packet(
        output_path=tmp_path / "repair.json",
        iteration_id="iter-000012",
        branch_brief=_brief(),
        parent=_result(version_id="v000037", branch_index=-1),
        candidate=candidate,
        summary_resolver=lambda match: {},
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["candidate"]["activation"]["changed_action_count"] == 3


def test_activation_repair_packet_needs_no_match_and_preserves_scope(tmp_path):
    from dataclasses import replace

    from agentbench_frame.hl.repair import build_activation_repair_packet

    candidate = replace(
        _result(version_id="v000041", branch_index=1, status="failed"),
        activation={
            "status": "complete",
            "decision_count": 256,
            "changed_action_count": 0,
            "changed_fraction": 0.0,
            "episodes": [],
            "details": {"changed_examples": []},
        },
    )
    path = build_activation_repair_packet(
        output_path=tmp_path / "activation-repair.json",
        iteration_id="iter-000012",
        branch_brief=_brief(),
        parent=_result(version_id="v000037", branch_index=-1),
        candidate=candidate,
        minimum_changed_actions=2,
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["repair_kind"] == "activation_integration"
    assert value["minimum_changed_actions"] == 2
    assert value["candidate"]["activation"]["decision_count"] == 256
    assert value["candidate"]["matches"] == []
    assert value["scope"]["preservation_contract"].startswith("ordinary portal")


def test_activation_repair_packet_embeds_bounded_edit_context(tmp_path):
    from dataclasses import replace

    from agentbench_frame.hl.repair import (
        build_activation_repair_packet,
        enrich_activation_repair_packet,
    )

    candidate = replace(
        _result(version_id="v000041", branch_index=1, status="failed"),
        activation={
            "status": "complete",
            "decision_count": 256,
            "changed_action_count": 0,
            "details": {"changed_examples": []},
        },
    )
    path = build_activation_repair_packet(
        output_path=tmp_path / "activation-repair.json",
        iteration_id="iter-000012",
        branch_brief=_brief(),
        parent=_result(version_id="v000037", branch_index=-1),
        candidate=candidate,
        minimum_changed_actions=2,
    )
    source = tmp_path / "ai.py"
    source.write_text(
        "def ai_func(state):\n    return helper(state)\n\n"
        "def helper(state):\n    return 0\n",
        encoding="utf-8",
    )
    digest = tmp_path / "digest.json"
    digest.write_text('{"atomic_actions":["HOLD","BUILD"]}\n', encoding="utf-8")
    research = tmp_path / "research.json"
    research.write_text('{"open_questions":["activation"]}\n', encoding="utf-8")
    experience = tmp_path / "SKILL.md"
    experience.write_text("condition-scoped experience", encoding="utf-8")

    enrich_activation_repair_packet(
        path,
        game_digest_path=digest,
        research_state_path=research,
        experience_path=experience,
        candidate_source_path=source,
        policy_entry_symbol="ai_func",
        smoke_command=("python", "smoke.py", "--workspace", str(tmp_path)),
        activation_command=(
            "python",
            "activation_check.py",
            "--candidate",
            str(tmp_path),
        ),
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["candidate"]["activation"]["changed_action_count"] == 0
    assert value["game_digest"]["atomic_actions"] == ["HOLD", "BUILD"]
    assert value["research_state"]["open_questions"] == ["activation"]
    assert value["experience_skill"] == "condition-scoped experience"
    assert [item["name"] for item in value["candidate_code_slices"]] == [
        "ai_func",
        "helper",
    ]
    assert value["smoke_contract"]["command"][0] == "python"
    assert value["activation_contract"]["command"][1] == "activation_check.py"
    assert value["activation_contract"]["minimum_changed_actions"] == 2
    assert value["experience_update_contract"]["required_arrays"] == [
        "positive_patterns",
        "negative_patterns",
        "open_questions",
        "compression_notes",
    ]


def test_repair_packet_rejects_feedback_without_shared_seed(tmp_path):
    from agentbench_frame.hl.repair import build_repair_packet

    parent = _result(version_id="v000037", branch_index=-1)
    candidate = _result(version_id="v000041", branch_index=1)
    candidate_match = dict(candidate.evaluation.matches[0], seed=102)
    candidate = candidate.__class__(
        act_id=candidate.act_id,
        branch_index=candidate.branch_index,
        version=candidate.version,
        evaluation=candidate.evaluation.__class__(
            status="complete",
            score=0.0,
            matches=(candidate_match,),
        ),
        provider=candidate.provider,
    )

    with pytest.raises(ValueError, match="shared seed"):
        build_repair_packet(
            output_path=tmp_path / "repair.json",
            iteration_id="iter-000012",
            branch_brief=_brief(),
            parent=parent,
            candidate=candidate,
            summary_resolver=lambda match: {"summary": match["replay"]},
        )


def test_better_repair_becomes_branch_representative():
    from agentbench_frame.hl.repair import select_branch_representative

    initial = _result(version_id="v000041", branch_index=1, margin=-295)
    repaired = _result(version_id="v000045", branch_index=1, margin=-150)

    assert select_branch_representative(initial, repaired) is repaired


def test_failed_or_tied_repair_cannot_erase_initial_candidate():
    from agentbench_frame.hl.repair import select_branch_representative

    initial = _result(version_id="v000041", branch_index=1, margin=-295)
    failed = _result(version_id="v000045", branch_index=1, status="timeout")
    tied = _result(version_id="v000046", branch_index=1, margin=-295)

    assert select_branch_representative(initial, failed) is initial
    assert select_branch_representative(initial, tied) is initial


def test_representative_selection_rejects_cross_branch_repair():
    from agentbench_frame.hl.repair import select_branch_representative

    initial = _result(version_id="v000041", branch_index=1)
    repaired = _result(version_id="v000045", branch_index=2)

    with pytest.raises(ValueError, match="same branch"):
        select_branch_representative(initial, repaired)

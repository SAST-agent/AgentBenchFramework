import json

import pytest


def test_branch_briefs_require_four_distinct_mechanisms(tmp_path):
    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "branch_index": index,
                        "diagnosis": f"failure-{index}",
                        "mechanism": mechanism,
                        "activation_condition": f"condition-{index}",
                        "preservation_contract": f"preserve-{index}",
                        "expected_change": f"change-{index}",
                        "falsifier": f"falsifier-{index}",
                    }
                    for index, mechanism in enumerate(
                        ("path planning", "ghost prediction", "shield state", "portal goals")
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    briefs = load_branch_briefs(path, expected_count=4)

    assert [brief.branch_index for brief in briefs] == [0, 1, 2, 3]
    assert len({brief.mechanism for brief in briefs}) == 4
    assert briefs[2].activation_condition == "condition-2"
    assert briefs[2].preservation_contract == "preserve-2"


def test_branch_briefs_reject_duplicate_mechanisms(tmp_path):
    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "branch_index": index,
                        "diagnosis": "same diagnosis",
                        "mechanism": "change threshold 1" if index == 0 else "change threshold 2",
                        "activation_condition": f"condition-{index}",
                        "preservation_contract": f"preserve-{index}",
                        "expected_change": "same",
                        "falsifier": "same",
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct mechanisms"):
        load_branch_briefs(path, expected_count=4)


def test_branch_briefs_reject_missing_scope_contract(tmp_path):
    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"failure-{index}",
                    "mechanism": f"mechanism-{index}",
                    "expected_change": f"change-{index}",
                    "falsifier": f"falsifier-{index}",
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields are invalid"):
        load_branch_briefs(path, expected_count=4)


def test_candidate_input_packet_inlines_bounded_reusable_context(tmp_path):
    from agentbench_frame.hl.proposal import write_candidate_input_packet

    digest = tmp_path / "game_digest.json"
    research = tmp_path / "research_state.json"
    experience = tmp_path / "SKILL.md"
    summary = tmp_path / "summary.md"
    distillation = tmp_path / "ghost.json"
    digest.write_text('{"actions":[0,1,2,3,4]}', encoding="utf-8")
    research.write_text('{"open_questions":["corner"]}', encoding="utf-8")
    experience.write_text("stable experience", encoding="utf-8")
    summary.write_text("s" * 13000, encoding="utf-8")
    distillation.write_text('{"fine":{"stay":0.5}}', encoding="utf-8")

    path = write_candidate_input_packet(
        output_path=tmp_path / "candidate_input.json",
        iteration_id="iter-000008",
        branch_brief={"branch_index": 2, "mechanism": "corner embargo"},
        game_digest_path=digest,
        research_state_path=research,
        experience_path=experience,
        replay_evidence=[
            {
                "opponent": "rank15",
                "seed": 101,
                "summary": str(summary),
                "trace": "/matches/seed-101/trace.jsonl",
            }
        ],
        previous_measurements={
            "benchmark_score": 0.25,
            "opponent_distillation_path": str(distillation),
        },
        active_target="rank15",
        locked_opponents=(),
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "1.0"
    assert value["game_digest"]["actions"] == [0, 1, 2, 3, 4]
    assert value["research_state"]["open_questions"] == ["corner"]
    assert value["experience_skill"] == "stable experience"
    assert value["opponent_distillation"]["fine"]["stay"] == 0.5
    assert value["replay_evidence"][0]["trace"].endswith("trace.jsonl")
    assert len(value["replay_evidence"][0]["summary_text"]) <= 12032
    assert value["replay_evidence"][0]["summary_text"].endswith(
        "[summary truncated]"
    )

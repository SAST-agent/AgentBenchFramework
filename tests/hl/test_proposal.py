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

import json

import pytest


def test_control_matrix_loads_strict_explicit_groups(tmp_path):
    from agentbench_frame.hl.control_matrix import load_control_matrix

    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "groups": [
                    {
                        "name": "rank01-robustness",
                        "opponent_ids": ["rank01"],
                        "roles": ["P0", "P1"],
                        "seeds": [1, 2, 7],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    groups = load_control_matrix(path)

    assert groups[0].name == "rank01-robustness"
    assert groups[0].opponent_ids == ("rank01",)
    assert groups[0].roles == ("P0", "P1")
    assert groups[0].seeds == (1, 2, 7)


def test_control_matrix_rejects_unknown_fields_and_duplicate_cases(tmp_path):
    from agentbench_frame.hl.control_matrix import load_control_matrix

    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "groups": [
                    {
                        "name": "rank01",
                        "opponent_ids": ["rank01", "rank01"],
                        "roles": ["P0"],
                        "seeds": [1],
                        "invented": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown control matrix group fields"):
        load_control_matrix(path)


def test_control_group_summary_separates_games_from_package_failures():
    from agentbench_frame.hl.control_matrix import summarize_group
    from agentbench_frame.hl.evaluator import CandidateEvaluation

    evaluation = CandidateEvaluation(
        status="incomplete",
        score=None,
        error="one case failed",
        matches=(
            {
                "status": "complete",
                "result": "win",
                "opponent": "rank01",
                "candidate_role": "P0",
                "seed": 1,
                "dense_margin": 8.0,
            },
            {
                "status": "failed",
                "result": None,
                "opponent": "rank05",
                "candidate_role": "P1",
                "seed": 1,
                "dense_margin": None,
                "faults": ["player timed out"],
            },
        ),
    )

    summary = summarize_group(evaluation)

    assert summary == {
        "status": "incomplete",
        "attempted": 2,
        "valid_games": 1,
        "package_failures": 1,
        "wins": 1,
        "draws": 0,
        "losses": 0,
        "win_rate_valid": 1.0,
        "mean_dense_margin_valid": 8.0,
    }

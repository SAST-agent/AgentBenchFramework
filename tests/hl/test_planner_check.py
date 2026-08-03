import json


def _branch(index, *, symbols=None, action="BUILD_TOWER"):
    labels = ("alpha", "beta", "gamma", "delta")
    return {
        "branch_index": index,
        "diagnosis": "parent holds on a reachable public state",
        "mechanism": f"{labels[index]} emits {action} through the public entry",
        "activation_condition": "activate at reference-0:0:P0",
        "preservation_contract": "preserve all other public states",
        "expected_change": "change at least two atomic decisions",
        "falsifier": "no activation or no live improvement",
        "code_symbols": symbols or ["AI.choose_operations", "AI.helper"],
    }


def _packet():
    return {
        "candidate_code_index": [
            {"name": "AI.choose_operations"},
            {"name": "AI.helper"},
            {"name": "AI.other"},
        ],
        "game_digest": {
            "roles": {
                "P0": {
                    "actions": [
                        {"name": "HOLD", "code": None},
                        {"name": "BUILD_TOWER", "code": 11},
                    ]
                }
            }
        },
        "parent_occupancy": {
            "state_examples": [
                {
                    "state_id": "reference-0:0:P0",
                    "legal_operation_types": [0, 11],
                    "parent_selected": [0, -1, -1],
                }
            ]
        },
    }


def test_planner_check_requires_valid_symbols_and_reachable_legal_action(tmp_path):
    from agentbench_frame.hl.planner_check import run_planner_check

    packet = tmp_path / "planner.json"
    briefs = tmp_path / "briefs.json"
    output = tmp_path / "result.json"
    packet.write_text(json.dumps(_packet()), encoding="utf-8")
    briefs.write_text(
        json.dumps([_branch(index) for index in range(4)]),
        encoding="utf-8",
    )

    returncode = run_planner_check(
        briefs_path=briefs,
        planner_input_path=packet,
        output_path=output,
        expected_count=4,
        entry_symbol="AI.choose_operations",
    )

    assert returncode == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "complete"
    assert value["activation_evidence"][0]["legal_pairs"][0] == {
        "state_id": "reference-0:0:P0",
        "action": "BUILD_TOWER",
        "operation_type": 11,
    }


def test_planner_check_rejects_unqualified_entry_symbol(tmp_path):
    from agentbench_frame.hl.planner_check import run_planner_check

    packet = tmp_path / "planner.json"
    briefs = tmp_path / "briefs.json"
    output = tmp_path / "result.json"
    packet.write_text(json.dumps(_packet()), encoding="utf-8")
    branches = [_branch(index) for index in range(4)]
    branches[2]["code_symbols"] = ["AI.helper", "AI.other"]
    briefs.write_text(json.dumps(branches), encoding="utf-8")

    returncode = run_planner_check(
        briefs_path=briefs,
        planner_input_path=packet,
        output_path=output,
        expected_count=4,
        entry_symbol="AI.choose_operations",
    )

    assert returncode == 2
    assert "must include AI.choose_operations" in json.loads(
        output.read_text(encoding="utf-8")
    )["error"]


def test_planner_check_failure_reports_exact_legal_names_for_correction(tmp_path):
    """A numeric-only proposal must receive a bounded semantic correction hint."""
    from agentbench_frame.hl.planner_check import run_planner_check

    packet_value = _packet()
    packet_value["game_digest"]["roles"]["P0"]["actions"].append(
        {"name": "DOWNGRADE_TOWER", "code": 13}
    )
    packet_value["parent_occupancy"]["state_examples"][0][
        "legal_operation_types"
    ] = [0, 13]
    packet = tmp_path / "planner.json"
    briefs = tmp_path / "briefs.json"
    output = tmp_path / "result.json"
    packet.write_text(json.dumps(packet_value), encoding="utf-8")
    briefs.write_text(
        json.dumps(
            [
                _branch(index, action="atomic operation code 13")
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )

    returncode = run_planner_check(
        briefs_path=briefs,
        planner_input_path=packet,
        output_path=output,
        expected_count=4,
        entry_symbol="AI.choose_operations",
    )

    assert returncode == 2
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "failed"
    assert value["correction_hint"] == {
        "branch_index": 0,
        "cited_states": [
            {
                "state_id": "reference-0:0:P0",
                "legal_operations": [
                    {"code": 0, "name": "HOLD"},
                    {"code": 13, "name": "DOWNGRADE_TOWER"},
                ],
            }
        ],
        "requirement": (
            "Name one listed legal operation exactly in mechanism or "
            "activation_condition."
        ),
    }


def test_planner_check_missing_state_reports_bounded_reachable_examples(tmp_path):
    """A range-only branch must receive exact state ids for its sole correction."""
    from agentbench_frame.hl.planner_check import run_planner_check

    packet = tmp_path / "planner.json"
    briefs = tmp_path / "briefs.json"
    output = tmp_path / "result.json"
    packet.write_text(json.dumps(_packet()), encoding="utf-8")
    branches = [_branch(index) for index in range(4)]
    branches[0]["activation_condition"] = "activate within the observed range"
    briefs.write_text(json.dumps(branches), encoding="utf-8")

    returncode = run_planner_check(
        briefs_path=briefs,
        planner_input_path=packet,
        output_path=output,
        expected_count=4,
        entry_symbol="AI.choose_operations",
    )

    assert returncode == 2
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["correction_hint"] == {
        "branch_index": 0,
        "available_states": [
            {
                "state_id": "reference-0:0:P0",
                "legal_operations": [
                    {"code": 0, "name": "HOLD"},
                    {"code": 11, "name": "BUILD_TOWER"},
                ],
            }
        ],
        "requirement": (
            "Cite one listed state_id and name one of its legal operations "
            "exactly."
        ),
    }


def test_planner_check_rejects_branch_that_only_repeats_parent_action(tmp_path):
    """A branch must predict an observable atomic change on its cited state."""
    from agentbench_frame.hl.planner_check import run_planner_check

    packet = tmp_path / "planner.json"
    briefs = tmp_path / "briefs.json"
    output = tmp_path / "result.json"
    packet.write_text(json.dumps(_packet()), encoding="utf-8")
    briefs.write_text(
        json.dumps([_branch(index, action="HOLD") for index in range(4)]),
        encoding="utf-8",
    )

    returncode = run_planner_check(
        briefs_path=briefs,
        planner_input_path=packet,
        output_path=output,
        expected_count=4,
        entry_symbol="AI.choose_operations",
    )

    assert returncode == 2
    value = json.loads(output.read_text(encoding="utf-8"))
    assert "only repeats the parent atomic operation" in value["error"]
    assert value["correction_hint"] == {
        "branch_index": 0,
        "cited_states": [
            {
                "state_id": "reference-0:0:P0",
                "parent_operation_type": 0,
                "divergent_legal_operations": [
                    {"code": 11, "name": "BUILD_TOWER"}
                ],
            }
        ],
        "requirement": (
            "Propose one listed operation whose code differs from the "
            "parent operation on the same state."
        ),
    }

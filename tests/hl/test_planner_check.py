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

import json


def test_activation_check_enforces_minimum_and_writes_bounded_result(tmp_path):
    from agentbench_frame.games.antwar2.activation_check import run_activation_check
    from agentbench_frame.hl.game_profile import BehaviorComparison

    references = tmp_path / "references.json"
    references.write_text(
        json.dumps([{"replay": str(tmp_path / "replay.json"), "role": "P0"}]),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    seen = {}

    def comparator(parent, candidate, **kwargs):
        seen.update(parent=parent, candidate=candidate, **kwargs)
        return BehaviorComparison(
            status="complete",
            decision_count=8,
            changed_action_count=1,
            details={"changed_examples": [{"state_id": "s0"}]},
        )

    returncode = run_activation_check(
        parent_root=tmp_path / "parent",
        candidate_root=tmp_path / "candidate",
        references_path=references,
        output_path=output,
        epsilon=0.05,
        minimum_changed_actions=2,
        comparator=comparator,
    )

    assert returncode == 2
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "failed"
    assert value["changed_action_count"] == 1
    assert value["minimum_changed_actions"] == 2
    assert value["details"]["changed_examples"][0]["state_id"] == "s0"
    assert seen["references"][0][1] == "P0"


def test_activation_check_command_freezes_reference_manifest(tmp_path):
    from agentbench_frame.games.antwar2.activation_check import (
        build_activation_check_command,
    )

    command = build_activation_check_command(
        parent_root=tmp_path / "parent",
        candidate_root=tmp_path / "candidate",
        references=((tmp_path / "one.json", "P1"),),
        references_path=tmp_path / "references.json",
        output_path=tmp_path / "result.json",
        epsilon=0.05,
        minimum_changed_actions=2,
    )

    assert command[0]
    assert "--minimum-changed-actions" in command
    assert command[-1] == "2"
    assert json.loads((tmp_path / "references.json").read_text(encoding="utf-8"))[0][
        "role"
    ] == "P1"

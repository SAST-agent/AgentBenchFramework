import json
import subprocess
import sys
from pathlib import Path

import yaml


ASSETS = (
    Path(__file__).parents[2]
    / "src"
    / "agentbench_frame"
    / "games"
    / "antwar2"
    / "assets"
)


def test_builtin_antwar2_profile_declares_portable_policy_contract():
    from agentbench_frame.hl.game_profile import (
        get_game_profile,
        reset_game_profiles_for_testing,
    )

    reset_game_profiles_for_testing()
    profile = get_game_profile("30_antwar2")
    prompt = profile.prompt_profile()

    assert profile.game_id == "30_antwar2"
    assert prompt.roles == ("P0", "P1")
    assert prompt.policy_entry_symbol == "AI.choose_operations"
    assert prompt.candidate_source_relative == "ai.py"
    assert "ordered list[Operation]" in prompt.output_contract
    assert profile.required_local_paths == (
        "backend_source_archive",
        "backend_executable",
        "backend_workdir",
        "sdk_root",
        "human_pool_root",
        "human_manifest",
        "workspace",
        "runs_root",
    )


def test_rules_match_frozen_backend_terminal_and_operation_contracts():
    text = (ASSETS / "rules.md").read_text(encoding="utf-8")

    for required in (
        "19 × 19",
        "512",
        "P0 wins",
        "BUILD_TOWER(x, y)",
        "Basic",
        "UPGRADE_TOWER(tower_id, target_type)",
        "50 coins",
        "accepted prefix",
        "winner",
    ):
        assert required in text
    assert "BUILD_TOWER(x, y, type)" not in text


def test_atomic_decision_space_contains_only_protocol_operations():
    value = yaml.safe_load(
        (ASSETS / "decision_space.yaml").read_text(encoding="utf-8")
    )
    actions = value["roles"]["P0"]["actions"]

    assert actions == [
        {"name": "HOLD", "code": None, "arguments": []},
        {"name": "BUILD_TOWER", "code": 11, "arguments": ["x", "y"]},
        {
            "name": "UPGRADE_TOWER",
            "code": 12,
            "arguments": ["tower_id", "target_type"],
        },
        {
            "name": "DOWNGRADE_TOWER",
            "code": 13,
            "arguments": ["tower_id"],
        },
        {
            "name": "USE_LIGHTNING_STORM",
            "code": 21,
            "arguments": ["x", "y"],
        },
        {
            "name": "USE_EMP_BLASTER",
            "code": 22,
            "arguments": ["x", "y"],
        },
        {
            "name": "USE_DEFLECTOR",
            "code": 23,
            "arguments": ["x", "y"],
        },
        {
            "name": "USE_EMERGENCY_EVASION",
            "code": 24,
            "arguments": ["x", "y"],
        },
        {
            "name": "UPGRADE_GENERATION_SPEED",
            "code": 31,
            "arguments": [],
        },
        {
            "name": "UPGRADE_GENERATED_ANT",
            "code": 32,
            "arguments": [],
        },
    ]
    assert value["roles"]["P1"]["actions"] == actions
    serialized = json.dumps(value, ensure_ascii=False).lower()
    for invented in ("rush", "turtle", "producer loop", "counter strategy"):
        assert invented not in serialized


def test_replay_skill_requires_atomic_causal_evidence_and_live_validation():
    text = (
        ASSETS / "replay-skill" / "antwar2-replay" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "public condition" in text
    assert "accepted atomic operation" in text
    assert "immediate public effect" in text
    assert "delayed public effect" in text
    assert "live match" in text
    assert "Do not infer intent" in text


def test_replay_summarizer_translates_operations_and_breaches(tmp_path):
    replay = tmp_path / "replay.json"
    output = tmp_path / "summary.json"
    replay.write_text(
        json.dumps(
            [
                {
                    "seed": 7,
                    "op0": [],
                    "op1": [],
                    "round_state": {
                        "camps": [50, 50],
                        "coins": [50, 50],
                        "towers": [],
                        "ants": [],
                        "winner": -1,
                    },
                },
                {
                    "op0": [
                        {"type": 11, "id": -1, "args": -1, "pos": {"x": 4, "y": 5}}
                    ],
                    "op1": [{"type": 31, "id": -1, "args": -1, "pos": {"x": -1, "y": -1}}],
                    "round_state": {
                        "camps": [50, 49],
                        "coins": [38, 3],
                        "towers": [
                            {"id": 0, "player": 0, "type": 0, "hp": 10, "pos": {"x": 4, "y": 5}}
                        ],
                        "ants": [],
                        "winner": 0,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(
                ASSETS
                / "replay-skill"
                / "antwar2-replay"
                / "scripts"
                / "summarize_replay.py"
            ),
            str(replay),
            "--output",
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))

    assert value["rounds"] == 2
    assert value["winner"] == 0
    assert value["terminal_camps"] == [50, 49]
    assert value["players"][0]["accepted_operation_counts"] == {
        "BUILD_TOWER": 1,
        "HOLD": 1,
    }
    assert value["players"][0]["breaches"] == 1
    assert value["event_index"][0]["operation"]["name"] == "BUILD_TOWER"


def test_trace_window_is_round_bounded_and_omits_unrequested_records(tmp_path):
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "window.json"
    records = []
    for round_index in range(10):
        records.extend(
            (
                {
                    "sequence": round_index * 2,
                    "round": round_index,
                    "kind": "public_state",
                    "public_state": {
                        "round": round_index,
                        "coins": [50 + round_index, 50],
                        "camps": [50, 50],
                        "towers": [],
                        "ants": [],
                    },
                },
                {
                    "sequence": round_index * 2 + 1,
                    "round": round_index,
                    "kind": "ai_operations",
                    "player": 0,
                    "operations": [],
                },
            )
        )
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(
                ASSETS
                / "replay-skill"
                / "antwar2-replay"
                / "scripts"
                / "inspect_trace_window.py"
            ),
            str(trace),
            "--round",
            "5",
            "--radius",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))

    assert value["requested_round"] == 5
    assert value["rounds"] == [4, 5, 6]
    assert {item["round"] for item in value["events"]} == {4, 5, 6}
    assert "sequence" not in json.dumps(value)

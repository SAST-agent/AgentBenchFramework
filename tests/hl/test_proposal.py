import json
import sys

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
                        "code_symbols": ["ai_func", f"helper_{index}"],
                    }
                    for index, mechanism in enumerate(
                        ("path planning", "ghost prediction", "shield state", "portal goals")
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    briefs = load_branch_briefs(
        path,
        expected_count=4,
        known_code_symbols={"ai_func", *(f"helper_{index}" for index in range(4))},
    )

    assert [brief.branch_index for brief in briefs] == [0, 1, 2, 3]
    assert len({brief.mechanism for brief in briefs}) == 4
    assert briefs[2].activation_condition == "condition-2"
    assert briefs[2].preservation_contract == "preserve-2"
    assert briefs[2].code_symbols == ("ai_func", "helper_2")


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
                        "code_symbols": ["ai_func", f"helper_{index}"],
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct mechanisms"):
        load_branch_briefs(
            path,
            expected_count=4,
            known_code_symbols={"ai_func", *(f"helper_{index}" for index in range(4))},
        )


def test_branch_briefs_reject_missing_scope_contract(tmp_path):
    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"failure-{index}",
                    "mechanism": ("route", "shield", "portal", "escape")[
                        index
                    ],
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


def test_branch_briefs_accept_redundant_scope_annotation_from_file_mode(
    tmp_path,
):
    """Validated-file planners may spell out the prompted scope annotation."""

    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"failure-{index}",
                    "mechanism": ("route", "shield", "portal", "escape")[
                        index
                    ],
                    "activation_condition": f"condition-{index}",
                    "preservation_contract": f"preserve-{index}",
                    "scope_contract": f"touch ai_func and helper_{index}",
                    "expected_change": f"change-{index}",
                    "falsifier": f"falsifier-{index}",
                    "code_symbols": ["ai_func", f"helper_{index}"],
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )

    briefs = load_branch_briefs(
        path,
        expected_count=4,
        known_code_symbols={
            "ai_func",
            *(f"helper_{index}" for index in range(4)),
        },
    )

    assert briefs[0].code_symbols == ("ai_func", "helper_0")


@pytest.mark.parametrize(
    ("symbols", "message"),
    [
        (["helper"], "2-8"),
        (["ai_func", "ai_func"], "unique"),
        (["helper", "other"], "include ai_func"),
        (["ai_func", "invented"], "unknown"),
        (["ai_func", *[f"helper_{index}" for index in range(8)]], "2-8"),
    ],
)
def test_branch_briefs_reject_invalid_code_symbol_contract(
    tmp_path, symbols, message
):
    """Catch planner branches that cannot be resolved to bounded policy slices."""
    from agentbench_frame.hl.proposal import load_branch_briefs

    path = tmp_path / "branch_briefs.json"
    path.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "branch_index": index,
                        "diagnosis": f"failure-{index}",
                        "mechanism": f"mechanism-{index}",
                        "activation_condition": f"condition-{index}",
                        "preservation_contract": f"preserve-{index}",
                        "expected_change": f"change-{index}",
                        "falsifier": f"falsifier-{index}",
                        "code_symbols": symbols,
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_branch_briefs(
            path,
            expected_count=4,
            known_code_symbols={
                "ai_func",
                "helper",
                "other",
                *(f"helper_{index}" for index in range(8)),
            },
        )


def test_candidate_input_packet_inlines_bounded_reusable_context(tmp_path):
    from agentbench_frame.hl.proposal import write_candidate_input_packet

    digest = tmp_path / "game_digest.json"
    research = tmp_path / "research_state.json"
    experience = tmp_path / "SKILL.md"
    summary = tmp_path / "summary.md"
    distillation = tmp_path / "ghost.json"
    candidate_source = tmp_path / "ai.py"
    smoke_fixture = tmp_path / "context" / "rollman_smoke_fixture.py"
    digest.write_text('{"actions":[0,1,2,3,4]}', encoding="utf-8")
    research.write_text('{"open_questions":["corner"]}', encoding="utf-8")
    experience.write_text("stable experience", encoding="utf-8")
    summary.write_text("s" * 13000, encoding="utf-8")
    distillation.write_text('{"fine":{"stay":0.5}}', encoding="utf-8")
    candidate_source.write_text(
        "def ai_func(state):\n"
        "    return helper(state)\n"
        "\n"
        "def helper(state):\n"
        "    return 0\n",
        encoding="utf-8",
    )
    smoke_fixture.parent.mkdir()
    smoke_fixture.write_text("# framework fixture\n", encoding="utf-8")

    path = write_candidate_input_packet(
        output_path=tmp_path / "candidate_input.json",
        iteration_id="iter-000008",
        branch_brief={
            "branch_index": 2,
            "mechanism": "corner embargo",
            "code_symbols": ["ai_func", "helper"],
        },
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
        candidate_source_path=candidate_source,
        smoke_fixture_path=smoke_fixture,
        candidate_workspace=tmp_path,
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "1.0"
    assert value["game_digest"]["actions"] == [0, 1, 2, 3, 4]
    assert value["research_state"]["open_questions"] == ["corner"]
    assert value["experience_skill"] == "stable experience"
    assert value["opponent_distillation"]["fine"]["stay"] == 0.5
    assert value["candidate_code_index"] == [
        {
            "name": "ai_func",
            "signature": "ai_func(state)",
            "start_line": 1,
            "end_line": 2,
        },
        {
            "name": "helper",
            "signature": "helper(state)",
            "start_line": 4,
            "end_line": 5,
        },
    ]
    assert value["candidate_code_slices"] == [
        {
            "name": "ai_func",
            "signature": "ai_func(state)",
            "start_line": 1,
            "end_line": 2,
            "completeness": "complete",
            "source": "def ai_func(state):\n    return helper(state)\n",
        },
        {
            "name": "helper",
            "signature": "helper(state)",
            "start_line": 4,
            "end_line": 5,
            "completeness": "complete",
            "source": "def helper(state):\n    return 0\n",
        },
    ]
    assert value["smoke_contract"] == {
        "fixture_path": str(smoke_fixture.resolve()),
        "scenario_path": str((tmp_path / ".agentbench" / "smoke_scenario.json").resolve()),
        "result_path": str(
            (tmp_path / ".agentbench" / "candidate_smoke_result.json").resolve()
        ),
        "command": [
            sys.executable,
            str(smoke_fixture.resolve()),
            "--workspace",
            str(tmp_path.resolve()),
            "--scenario",
            str((tmp_path / ".agentbench" / "smoke_scenario.json").resolve()),
            "--output",
            str(
                (tmp_path / ".agentbench" / "candidate_smoke_result.json").resolve()
            ),
        ],
    }
    assert value["replay_evidence"][0]["trace"].endswith("trace.jsonl")
    assert len(value["replay_evidence"][0]["summary_text"]) <= 12032
    assert value["replay_evidence"][0]["summary_text"].endswith(
        "[summary truncated]"
    )


def test_candidate_code_slices_preserve_entry_and_bound_truncated_helpers(
    tmp_path,
):
    """Catch packets that reintroduce a full-policy read through selected helpers."""
    from agentbench_frame.hl.proposal import build_candidate_code_slices

    source = tmp_path / "ai.py"
    source.write_text(
        "def large_helper(state):\n"
        + "".join(f"    value_{index} = {index}\n" for index in range(80))
        + "    return 1\n\n"
        + "def ai_func(state):\n    return large_helper(state)\n\n"
        + "def small_helper(state):\n    return 0\n",
        encoding="utf-8",
    )

    slices = build_candidate_code_slices(
        source,
        code_symbols=("large_helper", "ai_func", "small_helper"),
        source_limit=700,
    )

    assert [item["name"] for item in slices] == [
        "large_helper",
        "ai_func",
        "small_helper",
    ]
    assert slices[0]["completeness"] == "truncated"
    assert "[source truncated]" in slices[0]["source"]
    assert slices[1]["completeness"] == "complete"
    assert slices[1]["source"] == (
        "def ai_func(state):\n    return large_helper(state)\n"
    )
    assert slices[2]["completeness"] == "complete"
    assert sum(len(item["source"]) for item in slices) <= 700


def test_candidate_code_slices_reject_unknown_symbol_and_oversized_entry(
    tmp_path,
):
    """Catch silent partial entry functions and planner/source mismatches."""
    from agentbench_frame.hl.proposal import build_candidate_code_slices

    source = tmp_path / "ai.py"
    source.write_text(
        "def ai_func(state):\n" + "    state += 1\n" * 50 + "    return 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown code symbol"):
        build_candidate_code_slices(
            source,
            code_symbols=("ai_func", "invented"),
            source_limit=2000,
        )
    with pytest.raises(ValueError, match="ai_func exceeds"):
        build_candidate_code_slices(
            source,
            code_symbols=("ai_func",),
            source_limit=100,
        )


def test_candidate_code_index_and_briefs_support_qualified_class_entry(tmp_path):
    from agentbench_frame.hl.proposal import (
        branch_briefs_json_schema,
        build_candidate_code_index,
        build_candidate_code_slices,
        load_branch_briefs,
    )

    source = tmp_path / "ai.py"
    source.write_text(
        "class AI:\n"
        "    def choose_operations(self, state, player, bundles=None):\n"
        "        return self._economy(state, player)\n"
        "\n"
        "    def _economy(self, state, player):\n"
        "        return []\n",
        encoding="utf-8",
    )
    symbols = {item["name"] for item in build_candidate_code_index(source)}
    briefs = tmp_path / "briefs.json"
    briefs.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "branch_index": index,
                        "diagnosis": f"evidence {index}",
                        "mechanism": ("economy", "defense", "timing", "counterplay")[index],
                        "activation_condition": f"condition {index}",
                        "preservation_contract": f"preserve {index}",
                        "expected_change": f"change {index}",
                        "falsifier": f"falsifier {index}",
                        "code_symbols": [
                            "AI.choose_operations",
                            "AI._economy",
                        ],
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    assert symbols == {"AI.choose_operations", "AI._economy"}
    assert load_branch_briefs(
        briefs,
        expected_count=4,
        known_code_symbols=symbols,
        required_entry_symbol="AI.choose_operations",
    )[0].code_symbols[0] == "AI.choose_operations"
    assert (
        branch_briefs_json_schema(
            expected_count=4,
            required_entry_symbol="AI.choose_operations",
        )["properties"]["branches"]["items"]["properties"]["code_symbols"][
            "contains"
        ]["const"]
        == "AI.choose_operations"
    )
    slices = build_candidate_code_slices(
        source,
        code_symbols=("AI.choose_operations", "AI._economy"),
        entry_symbol="AI.choose_operations",
    )
    assert slices[0]["source"].lstrip().startswith("def choose_operations")


def test_planner_input_packet_collapses_exact_inputs_without_replay_paths(tmp_path):
    """Catch planner contexts that require one tool call per summary file."""
    from agentbench_frame.hl.proposal import write_planner_input_packet

    digest = tmp_path / "game_digest.json"
    manifest = tmp_path / "context-manifest.json"
    research = tmp_path / "research_state.json"
    summary = tmp_path / "summary.md"
    distillation = tmp_path / "ghost.json"
    candidate_source = tmp_path / "ai.py"
    digest.write_text('{"actions":[0,1,2,3,4]}', encoding="utf-8")
    manifest.write_text('{"bundle_hash":"frozen"}', encoding="utf-8")
    research.write_text('{"open_questions":["corner"]}', encoding="utf-8")
    summary.write_text("rank15 evidence", encoding="utf-8")
    distillation.write_text('{"fine":{"chase":0.8}}', encoding="utf-8")
    candidate_source.write_text(
        "def ai_func(game_state):\n"
        "    return helper(game_state)\n"
        "\n"
        "def helper(game_state):\n"
        "    return 0\n",
        encoding="utf-8",
    )

    path = write_planner_input_packet(
        output_path=tmp_path / "planner_input.json",
        iteration_id="iter-000009",
        parent_version_id="v000004",
        game_digest_path=digest,
        context_manifest_path=manifest,
        research_state_path=research,
        replay_evidence=[
            {
                "opponent": "rank15",
                "candidate_role": "attacker",
                "seed": 1103,
                "result": "loss",
                "phase": "learning",
                "points": 0.0,
                "candidate_score": 10,
                "opponent_score": 100,
                "dense_margin": -90,
                "summary": str(summary),
                "replay": "/forbidden/replay.jsonl",
                "trace": "/forbidden/trace.jsonl",
            }
        ],
        previous_measurements={
            "benchmark_score": 0.5,
            "opponent_distillation_path": str(distillation),
        },
        active_target="rank15",
        candidate_source_path=candidate_source,
        parent_occupancy={
            "state_count": 128,
            "roles": {"attacker": {"self_coins": {"min": 0, "max": 50}}},
        },
        planner_command=("python", "-m", "agentbench_frame.hl.planner_check"),
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "1.0"
    assert value["iteration_id"] == "iter-000009"
    assert value["parent_version_id"] == "v000004"
    assert value["game_digest"]["actions"] == [0, 1, 2, 3, 4]
    assert value["context_manifest"]["bundle_hash"] == "frozen"
    assert value["research_state"]["open_questions"] == ["corner"]
    assert value["opponent_distillation"]["fine"]["chase"] == 0.8
    assert value["parent_occupancy"]["state_count"] == 128
    assert value["planner_contract"]["command"][-1] == (
        "agentbench_frame.hl.planner_check"
    )
    assert value["candidate_code_index"] == [
        {
            "name": "ai_func",
            "signature": "ai_func(game_state)",
            "start_line": 1,
            "end_line": 2,
        },
        {
            "name": "helper",
            "signature": "helper(game_state)",
            "start_line": 4,
            "end_line": 5,
        },
    ]
    assert value["replay_evidence"] == [
        {
            "candidate_role": "attacker",
            "candidate_score": 10,
            "dense_margin": -90,
            "opponent": "rank15",
            "opponent_score": 100,
            "phase": "learning",
            "points": 0.0,
            "result": "loss",
            "seed": 1103,
            "summary_text": "rank15 evidence",
        }
    ]
    assert "/forbidden/" not in path.read_text(encoding="utf-8")


def test_planner_input_names_legal_atomic_operations_in_each_state(tmp_path):
    """Prevent planners from guessing the meaning of bare operation codes."""
    from agentbench_frame.hl.proposal import write_planner_input_packet

    digest = tmp_path / "game_digest.json"
    manifest = tmp_path / "context-manifest.json"
    research = tmp_path / "research_state.json"
    candidate_source = tmp_path / "ai.py"
    digest.write_text(
        json.dumps(
            {
                "roles": {
                    "P0": {
                        "actions": [
                            {"name": "HOLD", "code": None},
                            {"name": "BUILD_TOWER", "code": 11},
                            {"name": "DOWNGRADE_TOWER", "code": 13},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text("{}", encoding="utf-8")
    research.write_text("{}", encoding="utf-8")
    candidate_source.write_text(
        "class AI:\n"
        "    def choose_operations(self, state):\n"
        "        return []\n",
        encoding="utf-8",
    )

    path = write_planner_input_packet(
        output_path=tmp_path / "planner_input.json",
        iteration_id="iter-000001",
        parent_version_id="v000000",
        game_digest_path=digest,
        context_manifest_path=manifest,
        research_state_path=research,
        replay_evidence=[],
        previous_measurements={},
        active_target="rank01",
        candidate_source_path=candidate_source,
        parent_occupancy={
            "state_count": 1,
            "state_examples": [
                {
                    "state_id": "reference-0:23:P0",
                    "legal_operation_types": [0, 13],
                    "legal_atomic_actions": [
                        [0, -1, -1],
                        [13, 7, -1],
                    ],
                }
            ],
        },
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["parent_occupancy"]["state_examples"][0][
        "legal_operations"
    ] == [
        {"code": 0, "name": "HOLD"},
        {"code": 13, "name": "DOWNGRADE_TOWER"},
    ]
    assert value["parent_occupancy"]["state_examples"][0][
        "legal_atomic_actions"
    ] == [
        {"atom": [0, -1, -1], "name": "HOLD"},
        {"atom": [13, 7, -1], "name": "DOWNGRADE_TOWER"},
    ]


def test_reducer_packet_embeds_static_context_and_compacts_activation_details(
    tmp_path,
):
    from agentbench_frame.hl.proposal import enrich_reducer_input_packet

    reducer = tmp_path / "reducer.json"
    digest = tmp_path / "digest.json"
    research = tmp_path / "research.json"
    reducer.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "activation": {
                            "changed_action_count": 3,
                            "decision_count": 256,
                            "details": {
                                "mean_kl_nats_per_decision": 0.2,
                                "changed_examples": [
                                    {"state_id": "reference-0:3:P0"}
                                ],
                                "state_examples": [
                                    {"state_id": "large-unneeded-sample"}
                                ],
                            },
                        }
                    }
                ],
                "initial_candidates": [],
                "repairs": [],
                "representatives": [],
            }
        ),
        encoding="utf-8",
    )
    digest.write_text('{"roles":{"P0":{}}}', encoding="utf-8")
    research.write_text('{"open_questions":["tempo"]}', encoding="utf-8")

    path = enrich_reducer_input_packet(
        reducer,
        game_digest_path=digest,
        research_state_path=research,
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["game_digest"] == {"roles": {"P0": {}}}
    assert value["research_state"] == {"open_questions": ["tempo"]}
    activation = value["candidates"][0]["activation"]
    assert activation["changed_action_count"] == 3
    assert activation["details"]["changed_examples"] == [
        {"state_id": "reference-0:3:P0"}
    ]
    assert "state_examples" not in activation["details"]


def test_k4_evidence_packets_assign_complementary_hard_opponent_failures():
    from agentbench_frame.hl.proposal import stratify_rollout_evidence

    evidence = [
        {
            "opponent": opponent,
            "candidate_role": "attacker",
            "seed": seed,
            "result": "loss",
            "candidate_score": rollman,
            "opponent_score": 100,
            "dense_margin": rollman - 100,
            "phase": "learning",
            "summary": f"{opponent}-{seed}.md",
            "trace": f"{opponent}-{seed}.jsonl",
        }
        for opponent, seed, rollman in (
            ("rank15", 101, -50),
            ("rank15", 102, 10),
            ("rank15", 103, 20),
            ("rank16", 101, -40),
            ("rank16", 102, 15),
            ("rank16", 103, 25),
        )
    ]
    evidence.append(
        {
            "opponent": "rank15",
            "seed": 999,
            "result": "loss",
            "phase": "certification",
            "summary": "sealed.md",
            "trace": "sealed.jsonl",
        }
    )

    packets = stratify_rollout_evidence(
        evidence,
        hard_opponents=("rank15", "rank16"),
    )

    assert len(packets) == 4
    assert {row["opponent"] for row in packets[0]} == {"rank15"}
    assert {row["opponent"] for row in packets[1]} == {"rank16"}
    assert {row["opponent"] for row in packets[2]} == {"rank15", "rank16"}
    assert packets[3] != packets[0] and packets[3] != packets[1]
    assert all(len(packet) <= 2 for packet in packets)
    assert all(row["seed"] != 999 for packet in packets for row in packet)

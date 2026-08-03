import json
import hashlib
import random
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pytest

from agentbench_frame.eval import TrajectoryKLAgent, TrajectoryKLConfig
from agentbench_frame.games.miracle import research_protocol as research_protocol_module
from agentbench_frame.games.miracle.research_protocol import (
    ACTION_SCHEMA_VERSION,
    FROZEN_OPPONENT_DETERMINISM,
    TEST_OPPONENTS,
    TRAIN_OPPONENTS,
    TRAJECTORY_KL_EPSILON,
    VALIDATION_OPPONENTS,
    DeterministicHLActivePolicy,
    DeterministicHLReferencePolicy,
    IncompleteActionSupportError,
    IsolatedDeterministicPolicy,
    LegalCommandSet,
    build_action_support,
    build_frozen_test_cases,
    canonical_command,
    command_action_id,
    decode_ai_observation,
    enumerate_legal_commands,
    research_protocol_manifest,
)
from agentbench_frame.games.miracle.seeded_python_entry import run_seeded_script


def command(operation_type, **parameters):
    return {
        "player": 0,
        "round": 7,
        "operation_type": operation_type,
        "operation_parameters": parameters,
    }


def unit(
    unit_id,
    camp,
    unit_type,
    pos,
    *,
    can_atk=True,
    can_move=True,
    flying=False,
    max_move=3,
):
    return [
        unit_id, camp, unit_type, 2, 2, 4, 4, [1, 4], max_move, 4,
        list(pos), 1, int(flying), 1, 0, 0, int(can_atk), int(can_move),
    ]


def game_observation(*, artifact_type=2):
    return {
        "round": 7,
        "camp": 0,
        "map": {
            "units": [
                unit(3, 0, 0, (-6, 6, 0)),
                unit(4, 1, 1, (-3, 3, 0)),
                unit(5, 1, 1, (-2, 3, -1)),
            ],
            "barracks": [0, 1, -1, -1],
            "miracles": [30, 30],
        },
        "players": [
            [
                [[0, artifact_type, 8, 6, 0, 0, 0, [-1, -1, -1]]],
                12,
                12,
                [[0, 2, []], [1, 2, []], [3, 1, []]],
                [],
            ],
            [
                [[1, 0, 6, 5, 0, 0, 0, [-1, -1, -1]]],
                12,
                12,
                [[0, 2, []], [1, 2, []], [3, 1, []]],
                [],
            ],
        ],
    }


def test_population_split_is_disjoint_complete_and_keeps_existing_roles():
    groups = (set(TRAIN_OPPONENTS), set(VALIDATION_OPPONENTS), set(TEST_OPPONENTS))
    assert tuple(map(len, groups)) == (10, 3, 3)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == {f"rank{rank:02d}" for rank in range(1, 17)}
    assert "rank04" in TRAIN_OPPONENTS
    assert "rank09" in VALIDATION_OPPONENTS


def test_frozen_test_matrix_has_72_unique_fully_crossed_cases():
    cases = build_frozen_test_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 72
    for opponent in TEST_OPPONENTS:
        selected = [case for case in cases if case.opponent == opponent]
        assert len(selected) == 24
        assert {
            (case.evaluated_agent_camp, case.map_type, case.day_time, case.repeat)
            for case in selected
        } == {
            (camp, map_type, day_time, repeat)
            for camp in (0, 1)
            for map_type in (0, 1)
            for day_time in (0, 1)
            for repeat in (1, 2, 3)
        }


def test_each_logic_seed_realizes_the_frozen_map_and_day_draws():
    for case in build_frozen_test_cases():
        rng = random.Random(case.seeds.logic_seed)
        assert (rng.randint(0, 1), rng.randint(0, 1)) == (
            case.map_type,
            case.day_time,
        )


def test_manifest_freezes_kl_and_explicitly_excludes_decision_change_rate():
    manifest = research_protocol_manifest()
    assert research_protocol_module.MANIFEST_SCHEMA_VERSION == (
        "24-miracle-research-manifest-v2"
    )
    assert manifest["manifest_schema_version"] == (
        research_protocol_module.MANIFEST_SCHEMA_VERSION
    )
    assert research_protocol_module.BENCHMARK_VERSION == "24m-frozen-v1"
    assert manifest["protocol_version"] == "24-miracle-research-v2"
    assert manifest["benchmark_version"] == research_protocol_module.BENCHMARK_VERSION
    assert manifest["frozen_test"]["case_count"] == 72
    assert manifest["trajectory_kl"]["epsilon"] == 0.01
    assert manifest["trajectory_kl"]["measurement_profile"] == (
        "24_miracle_policy_information_gain_v2"
    )
    assert manifest["trajectory_kl"]["direction"] == "new||old"
    assert manifest["trajectory_kl"]["smoothing"] == "symmetric_epsilon_uniform_full_support"
    assert manifest["trajectory_kl"]["primary_episode_information_gain"] == "arithmetic_mean"
    assert manifest["trajectory_kl"]["primary_unit"] == "nats / decision"
    assert manifest["trajectory_kl"]["optional_sum_unit"] == "nats / episode"
    assert manifest["trajectory_kl"]["current_internal_memory_evidence"] == "not_collected"
    assert manifest["trajectory_kl"]["decision_change_rate"] == "not_collected"
    assert manifest["optimization_class"]["criterion"] == (
        "no_backpropagation_or_gradient_updates"
    )


def test_research_manifest_canonical_bytes_and_hash_are_stable():
    first = research_protocol_module.canonical_research_manifest_bytes()
    second = research_protocol_module.canonical_research_manifest_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert b"\r" not in first
    assert json.loads(first.decode("utf-8")) == research_protocol_manifest()
    expected = hashlib.sha256(first).hexdigest()
    assert research_protocol_module.research_manifest_sha256() == expected
    assert research_protocol_module.research_manifest_sha256(
        research_protocol_manifest()
    ) == expected
    assert expected == "7557015c0979c5acfbb1c553fc18614e3ea7246f6d2627478a94baa262ce61c5"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_manifest_rejects_nonstandard_top_level_json_numbers(value):
    with pytest.raises(ValueError):
        research_protocol_module.canonical_research_manifest_bytes(value)
    with pytest.raises(ValueError):
        research_protocol_module.research_manifest_sha256(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_manifest_rejects_nested_nonstandard_json_numbers(value):
    manifest = research_protocol_manifest()
    manifest["optimization_class"]["nonstandard"] = value
    with pytest.raises(ValueError):
        research_protocol_module.canonical_research_manifest_bytes(manifest)
    with pytest.raises(ValueError):
        research_protocol_module.research_manifest_sha256(manifest)


def test_manifest_tool_emits_exact_canonical_bytes():
    tool = Path(__file__).resolve().parents[2] / "tools" / "miracle_research_manifest.py"
    completed = subprocess.run(
        [sys.executable, str(tool)], capture_output=True, check=True
    )
    assert completed.stderr == b""
    assert completed.stdout == research_protocol_module.canonical_research_manifest_bytes()


def test_command_support_is_stable_ordered_and_fails_closed_when_incomplete():
    move = command("move", mover=3, position=[1, -1, 0])
    end = command("endround")
    left = build_action_support(LegalCommandSet([move, end], complete=True))
    right = build_action_support(LegalCommandSet([end, move], complete=True))
    assert left.schema_version == ACTION_SCHEMA_VERSION
    assert left.action_ids == right.action_ids
    assert left.resolve(command_action_id(move))["operation_type"] == "move"
    with pytest.raises(IncompleteActionSupportError, match="path enumeration"):
        build_action_support(
            LegalCommandSet([], complete=False, errors=["path enumeration failed"])
        )


def test_strict_deterministic_hl_adapters_produce_complete_finite_kl():
    move = command("move", mover=3, position=[1, -1, 0])
    end = command("endround")
    support = build_action_support(LegalCommandSet([move, end], complete=True))
    active = DeterministicHLActivePolicy(lambda observation, supplied: move)
    reference = DeterministicHLReferencePolicy(lambda observation, supplied: end)
    results = []
    measured = TrajectoryKLAgent(
        active,
        reference,
        lambda observation: support,
        TrajectoryKLConfig.for_policy_information_gain("old", "new"),
        results.append,
    )
    measured.reset()
    assert measured.act({"round": 7}) == support.resolve(command_action_id(move))
    measured.observe_transition({"terminated": True})
    assert len(results) == 1
    assert results[0].status == "complete"
    assert results[0].to_dict()["episode"] == results[0].episode == 1
    assert results[0].trajectory_kl_episode is not None
    assert results[0].trajectory_kl_episode > 0


def test_seeded_python_entry_is_deterministic_and_restores_host_rng(tmp_path):
    output = tmp_path / "output.json"
    script = tmp_path / "asset.py"
    script.write_text(
        "import json, random, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps([random.randint(0, 9), random.randint(0, 9)]))\n",
        encoding="utf-8",
    )
    before = random.getstate()
    run_seeded_script(script, 1234, [str(output)])
    first = json.loads(output.read_text(encoding="utf-8"))
    run_seeded_script(script, 1234, [str(output)])
    second = json.loads(output.read_text(encoding="utf-8"))
    assert first == second
    assert random.getstate() == before


def test_protocol_format_fake_transcript_enumerates_every_runtime_action_type():
    raw = json.dumps(game_observation(), separators=(",", ":"))
    transcript = f"{len(raw):06d}{raw}"
    observation = decode_ai_observation(transcript)
    legal = enumerate_legal_commands(observation)
    support = build_action_support(legal)
    commands = [candidate.action for candidate in support.actions]
    assert {item["operation_type"] for item in commands} == {
        "move", "attack", "summon", "use", "endround", "surrender",
    }
    assert all(item["player"] == 0 and item["round"] == 7 for item in commands)


def test_init_transcript_enumerates_all_distinct_legal_decks():
    raw = json.dumps({"camp": 1}, separators=(",", ":"))
    support = build_action_support(
        enumerate_legal_commands(f"{len(raw):06d}{raw}")
    )
    commands = [candidate.action for candidate in support.actions]
    assert len(commands) == 840  # 4 artifacts * P(7 creatures, 3)
    assert {item["operation_type"] for item in commands} == {"init"}


def test_init_support_includes_every_ordered_legal_deck():
    legal = enumerate_legal_commands({"camp": 0})
    support = build_action_support(legal)
    assert len(support.actions) == 4 * len(tuple(permutations(range(7), 3))) == 840

    left = command(
        "init",
        artifacts=["HolyLight"],
        creatures=["Archer", "Swordsman", "Priest"],
    )
    left["round"] = 0
    right = command(
        "init",
        artifacts=["HolyLight"],
        creatures=["Priest", "Swordsman", "Archer"],
    )
    right["round"] = 0
    assert command_action_id(left) != command_action_id(right)
    assert command_action_id(left) in support.action_ids
    assert command_action_id(right) in support.action_ids


_SPECIAL_MOVE_BORDER = (
    (7, -8, 1),
    (8, -8, 0),
    (8, -7, -1),
    (-8, 7, 1),
    (-8, 8, 0),
    (-7, 8, -1),
)


@pytest.mark.parametrize(
    ("target", "start"),
    [
        ((7, -8, 1), (6, -7, 1)),
        ((8, -8, 0), (7, -8, 1)),
        ((8, -7, -1), (7, -6, -1)),
        ((-8, 7, 1), (-7, 6, 1)),
        ((-8, 8, 0), (-8, 7, 1)),
        ((-7, 8, -1), (-6, 7, -1)),
    ],
)
def test_move_support_excludes_every_special_mapborder_position(target, start):
    observation = game_observation()
    observation["map"]["units"] = [
        unit(30, 0, 0, start, can_atk=False, max_move=1),
    ]
    moves = {
        tuple(item["operation_parameters"]["position"])
        for item in enumerate_legal_commands(observation).commands
        if item["operation_type"] == "move"
    }
    assert target not in moves


def test_holy_light_keeps_in_map_positions_that_only_movement_forbids():
    uses = {
        tuple(item["operation_parameters"]["target"])
        for item in enumerate_legal_commands(game_observation(artifact_type=0)).commands
        if item["operation_type"] == "use"
    }
    assert set(_SPECIAL_MOVE_BORDER) <= uses


def test_move_support_matches_ground_flying_and_same_layer_obstacles():
    ground = game_observation()
    ground["map"]["units"] = [
        unit(30, 0, 0, (-7, -4, 11), can_atk=False, max_move=1),
    ]
    ground_moves = {
        tuple(item["operation_parameters"]["position"])
        for item in enumerate_legal_commands(ground).commands
        if item["operation_type"] == "move"
    }
    assert (-6, -5, 11) not in ground_moves  # abyss blocks ground movement

    flying = game_observation()
    flying["map"]["units"] = [
        unit(
            30, 0, 2, (-7, -4, 11), can_atk=False,
            flying=True, max_move=1,
        ),
    ]
    flying_moves = {
        tuple(item["operation_parameters"]["position"])
        for item in enumerate_legal_commands(flying).commands
        if item["operation_type"] == "move"
    }
    assert (-6, -5, 11) in flying_moves  # flying may cross abyss

    occupied = game_observation()
    occupied["map"]["units"] = [
        unit(30, 0, 0, (2, -2, 0), can_atk=False, max_move=1),
        unit(31, 0, 1, (3, -2, -1), can_atk=False, can_move=False),
        unit(
            32, 0, 2, (2, -3, 1), can_atk=False, can_move=False,
            flying=True,
        ),
    ]
    occupied_moves = {
        tuple(item["operation_parameters"]["position"])
        for item in enumerate_legal_commands(occupied).commands
        if item["operation_type"] == "move"
        and item["operation_parameters"]["mover"] == 30
    }
    assert (3, -2, -1) not in occupied_moves
    assert (2, -3, 1) in occupied_moves


def test_move_support_can_end_on_enemy_obstruct_but_does_not_cross_it():
    observation = game_observation()
    observation["map"]["units"] = [
        unit(30, 0, 0, (2, -4, 2), can_atk=False, max_move=2),
        unit(31, 1, 1, (2, -2, 0), can_atk=False, can_move=False),
        unit(32, 0, 1, (1, -3, 2), can_atk=False, can_move=False),
        unit(33, 0, 1, (3, -4, 1), can_atk=False, can_move=False),
    ]
    moves = {
        tuple(item["operation_parameters"]["position"])
        for item in enumerate_legal_commands(observation).commands
        if item["operation_type"] == "move"
        and item["operation_parameters"]["mover"] == 30
    }
    assert (2, -3, 1) in moves  # enemy control-zone hex is a legal endpoint
    assert (3, -3, 0) not in moves  # the blocked alternatives force this through it


def test_multitarget_and_multiposition_actions_have_distinct_identities():
    support = build_action_support(enumerate_legal_commands(game_observation()))
    attacks = [
        item for item in support.actions
        if item.action["operation_type"] == "attack"
        and item.action["operation_parameters"]["attacker"] == 3
    ]
    moves = [
        item for item in support.actions
        if item.action["operation_type"] == "move"
        and item.action["operation_parameters"]["mover"] == 3
    ]
    assert {item.action["operation_parameters"]["target"] for item in attacks} == {4, 5}
    assert len({item.action_id for item in attacks}) == 2
    assert len(moves) > 1
    assert len({item.action_id for item in moves}) == len(moves)


def test_observation_enumerator_fails_closed_for_missing_state_or_infinite_support():
    missing = game_observation()
    missing["map"].pop("barracks")
    with pytest.raises(IncompleteActionSupportError, match="barracks"):
        enumerate_legal_commands(missing)
    with pytest.raises(IncompleteActionSupportError, match="WindBlessing"):
        enumerate_legal_commands(game_observation(artifact_type=3))


def test_isolated_policy_query_does_not_mutate_source_or_inputs():
    class StatefulSelector:
        def __init__(self):
            self.calls = 0

        def __call__(self, observation, support):
            self.calls += 1
            observation["round"] = 999
            return support.action_ids[0]

    selector = StatefulSelector()
    policy = IsolatedDeterministicPolicy(selector)
    observation = game_observation()
    before = json.dumps(observation, sort_keys=True)
    support = build_action_support(enumerate_legal_commands(observation))
    decision = policy.decide_with_distribution(observation, support)
    assert decision.action_id in support.action_ids
    assert selector.calls == 0
    assert json.dumps(observation, sort_keys=True) == before


def test_manifest_refuses_authoritative_ready_status_for_uncontrolled_test_opponents():
    manifest = research_protocol_manifest()
    assert manifest["status"] == "BLOCKED_NOT_AUTHORITATIVE"
    assert manifest["frozen_test"]["authoritative_ready"] is False
    assert FROZEN_OPPONENT_DETERMINISM["rank02"]["qualified"] is False
    assert FROZEN_OPPONENT_DETERMINISM["rank10"]["qualified"] is False
    assert FROZEN_OPPONENT_DETERMINISM["rank16"]["qualified"] is False


@pytest.mark.parametrize("malformed", [
    command("init", artifacts=["HolyLight"]),
    command("move", mover=3, position=[1, 2, 3]),
    command("attack", attacker=3, target=True),
    command("summon", type="Archer", level=0, position=[0, 1, -1]),
    command("use", target=[0, 1, -1]),
    command("endround", unexpected=1),
    command("surrender", unexpected=1),
])
def test_canonical_command_rejects_illegal_parameter_boundaries(malformed):
    with pytest.raises(ValueError):
        canonical_command(malformed)


def test_duplicate_commands_are_not_a_complete_support():
    end = command("endround")
    with pytest.raises(ValueError, match="duplicate"):
        build_action_support(LegalCommandSet([end, end], complete=True))


@pytest.mark.parametrize("camp", [False, True, 0.0, 1.0])
def test_strict_integer_init_observation_rejects_bool_and_float_camp(camp):
    with pytest.raises(IncompleteActionSupportError, match="camp"):
        enumerate_legal_commands({"camp": camp})


@pytest.mark.parametrize("camp", [False, True, 0.0, 1.0])
def test_strict_integer_runtime_observation_rejects_bool_and_float_camp(camp):
    observation = game_observation()
    observation["camp"] = camp
    with pytest.raises(IncompleteActionSupportError, match="camp"):
        enumerate_legal_commands(observation)


@pytest.mark.parametrize("player", [False, True, 0.0, 1.0])
def test_strict_integer_canonical_command_rejects_bool_and_float_player(player):
    candidate = command("endround")
    candidate["player"] = player
    with pytest.raises(ValueError, match="player"):
        canonical_command(candidate)


@pytest.mark.parametrize("level", [False, True, 1.0])
def test_strict_integer_canonical_summon_rejects_bool_and_float_level(level):
    candidate = command("summon", type="Archer", level=level, position=[0, 1, -1])
    with pytest.raises(ValueError, match="level"):
        canonical_command(candidate)


@pytest.mark.parametrize(
    ("field_index", "invalid"),
    [
        (0, False), (0, 3.0),
        (1, False), (1, 0.0),
        (2, False), (2, 0.0),
    ],
)
def test_strict_integer_runtime_unit_identity_and_enums_fail_closed(
    field_index, invalid
):
    observation = game_observation()
    observation["map"]["units"][0][field_index] = invalid
    with pytest.raises(IncompleteActionSupportError, match="unit"):
        enumerate_legal_commands(observation)


@pytest.mark.parametrize("owner", [False, True, 0.0, 1.0])
def test_strict_integer_barracks_owner_fails_closed(owner):
    observation = game_observation()
    observation["map"]["barracks"][0] = owner
    with pytest.raises(IncompleteActionSupportError, match="barracks"):
        enumerate_legal_commands(observation)


@pytest.mark.parametrize(
    ("field_index", "invalid", "label"),
    [
        (0, False, "artifact id"), (0, 0.0, "artifact id"),
        (1, False, "artifact type"), (1, 0.0, "artifact type"),
        (5, False, "artifact state"), (5, 0.0, "artifact state"),
        (6, False, "artifact target type"), (6, 0.0, "artifact target type"),
    ],
)
def test_strict_integer_artifact_identity_and_enums_fail_closed(
    field_index, invalid, label
):
    observation = game_observation()
    observation["players"][0][0][0][field_index] = invalid
    with pytest.raises(IncompleteActionSupportError, match=label):
        enumerate_legal_commands(observation)


@pytest.mark.parametrize(
    ("field_index", "invalid", "label"),
    [
        (0, False, "capacity type"), (0, 0.0, "capacity type"),
        (1, False, "available count"), (1, 1.0, "available count"),
    ],
)
def test_strict_integer_creature_capacity_fields_fail_closed(
    field_index, invalid, label
):
    observation = game_observation()
    observation["players"][0][3][0][field_index] = invalid
    with pytest.raises(IncompleteActionSupportError, match=label):
        enumerate_legal_commands(observation)


@pytest.mark.parametrize(
    ("operation", "parameters", "label"),
    [
        ("move", {"mover": 3.0, "position": [0, 1, -1]}, "mover"),
        ("attack", {"attacker": False, "target": 4}, "attacker"),
        ("attack", {"attacker": 3, "target": 4.0}, "target"),
        ("use", {"card": True, "target": 4}, "card"),
        ("use", {"card": 0, "target": 4.0}, "target"),
    ],
)
def test_strict_integer_canonical_command_ids_reject_bool_and_float(
    operation, parameters, label
):
    with pytest.raises(ValueError, match=label):
        canonical_command(command(operation, **parameters))

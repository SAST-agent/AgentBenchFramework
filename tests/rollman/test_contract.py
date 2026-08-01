from pathlib import Path

import yaml


AGENTBENCH_ROOT = Path("/Users/qingle/Code/SAST/AgentBench")


def test_frozen_backend_contract_matches_reviewed_rollman_values():
    from agentbench_frame.games.rollman.contract import RollmanContract

    contract = RollmanContract.from_agentbench(AGENTBENCH_ROOT)

    assert contract.map_sizes == (41, 32, 22)
    assert contract.max_rounds == (500, 400, 300)
    assert contract.skill_durations == (8, 8, 8, 8, 2)
    assert contract.portal_activation_rounds == (60, 50)
    assert contract.directions == {
        "STAY": 0,
        "UP": 1,
        "LEFT": 2,
        "DOWN": 3,
        "RIGHT": 4,
    }
    assert contract.spaces == {
        "WALL": 0,
        "EMPTY": 1,
        "REGULAR_BEAN": 2,
        "BONUS_BEAN": 3,
        "SPEED_BEAN": 4,
        "MAGNET_BEAN": 5,
        "SHIELD_BEAN": 6,
        "DOUBLE_BEAN": 7,
        "FROZE_BEAN": 8,
        "PORTAL": 9,
    }
    assert contract.events == {
        "EATEN_BY_GHOST": 0,
        "SHIELD_DESTROYED": 1,
        "FINISH_LEVEL": 2,
        "TIMEOUT": 3,
    }
    assert contract.score_constants["EATEN_BY_GHOST"] == -60
    assert contract.score_constants["EAT_ALL_BEANS"] == 50
    assert contract.score_constants["ROUND_BONUS_GAMMA"] == 0.43


def test_decision_space_is_complete_direction_support_without_inferred_intent():
    from agentbench_frame.games.rollman.contract import asset_path

    decision = yaml.safe_load(asset_path("decision_space.yaml").read_text(encoding="utf-8"))
    rollman = decision["roles"]["rollman"]

    assert rollman["decision_hook"] == "submitted_direction"
    assert [action["id"] for action in rollman["actions"]] == [0, 1, 2, 3, 4]
    assert all(action["protocol_legal"] for action in rollman["actions"])
    assert rollman["measurement"]["deterministic_policy"] is True
    assert rollman["measurement"]["distribution_source"] == "epsilon_measurement_channel"
    assert "target_item" not in str(decision)
    assert "tactical_hypothesis" not in str(decision)


def test_rule_asset_names_all_frozen_replay_fields_and_source_precedence():
    from agentbench_frame.games.rollman.contract import asset_path

    rules = asset_path("rules.md").read_text(encoding="utf-8")

    for field in (
        "pacman_step_block",
        "pacman_coord",
        "pacman_skills",
        "ghosts_step_block",
        "ghosts_coord",
        "score",
        "events",
        "portal_available",
        "StopReason",
    ):
        assert f"`{field}`" in rules
    assert "冻结后端" in rules
    assert "[8, 8, 8, 8, 2]" in rules
    assert "10 轮" not in rules


def test_rule_and_decision_assets_distinguish_sdk_object_from_replay_fields():
    from agentbench_frame.games.rollman.contract import asset_path

    rules = asset_path("rules.md").read_text(encoding="utf-8")
    decision = yaml.safe_load(
        asset_path("decision_space.yaml").read_text(encoding="utf-8")
    )

    assert "`GameState`" in rules
    assert "`pacman_pos`" in rules
    assert "`ghosts_pos`" in rules
    assert "`pacman_score`" in rules
    assert "`ghosts_score`" in rules
    assert "`gamestate_to_statedict()`" in rules
    interface = decision["policy_interface"]
    assert interface["input_type"] == "core.gamedata.GameState"
    assert interface["normalized_state_mapping"]["pacman_coord"] == "pacman_pos"
    assert interface["normalized_state_mapping"]["ghosts_coord"] == "ghosts_pos"

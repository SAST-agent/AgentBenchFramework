"""Miracle 决策空间的稳定动作契约。"""

from agentbench_frame.miracle.decision_space import Action, action_mask


def _obs(*, mana=0, units=None, capacities=None, miracles=None):
    return {
        "map": {
            "units": units or [],
            "miracles": miracles or [30, 30],
            "barracks": [-1, -1, -1, -1],
        },
        "players": [
            [[], mana, 12, capacities or [], []],
            [[], 0, 12, [], []],
        ],
        "round": 4,
        "camp": 0,
    }


def test_action_signature_is_stable_across_parameter_order():
    left = Action("move", {"mover": 7, "position": [1, -1, 0]})
    right = Action("move", {"position": [1, -1, 0], "mover": 7})

    assert left.signature() == right.signature()


def test_action_mask_always_contains_endround_and_surrender():
    actions = action_mask(_obs(), camp=0)

    assert Action("endround", {}) in actions
    assert Action("surrender", {}) in actions


def test_action_mask_enumerates_affordable_summons_for_deck_capacity():
    # type 0 = Archer；available_count=1；mana 只够 level 1。
    actions = action_mask(_obs(mana=2, capacities=[[0, 1, []]]), camp=0)

    summons = [a for a in actions if a.type == "summon"]
    assert len(summons) == 5
    assert {tuple(a.params["position"]) for a in summons} == {
        (-8, 6, 2), (-7, 6, 1), (-6, 6, 0), (-6, 7, -1), (-6, 8, -2),
    }
    assert {a.params["type"] for a in summons} == {"Archer"}
    assert {a.params["level"] for a in summons} == {1}


def test_action_mask_enumerates_attacks_in_range():
    mine = [9, 0, 1, 2, 2, 2, 2, [1, 1], 3, 2, [0, 0, 0], 1, 0, 0, 0, 0, 1, 1]
    enemy = [12, 1, 0, 2, 1, 2, 2, [3, 4], 3, 4, [1, -1, 0], 1, 0, 1, 0, 0, 1, 1]

    attacks = [a for a in action_mask(_obs(units=[mine, enemy]), 0) if a.type == "attack"]

    assert Action("attack", {"attacker": 9, "target": 12}) in attacks

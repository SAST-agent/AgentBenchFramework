import math

import pytest

from agentbench_frame.doto.decision_space import (
    ActionValidationError,
    canonicalize_action,
    action_mask,
    parse_observation,
    support_pair,
    termination_from_raw,
)


def raw_action(point=(1, 2)):
    return {
        "flag": 0,
        "move": [list(point)] + [[-1, -1]] * 4,
        "shoot": [[-1, -1]] * 5,
        "meteor": [[-1, -1]] * 5,
        "flash": [False] * 5,
    }


def observation(*, death=-1, fireball=0, meteor=0, meteor_num=1, flash=0, flash_num=1, carrier=-1):
    humans = []
    for human_id in range(10):
        humans.append([human_id, 10.0, 10.0, 100, meteor_num, meteor, flash_num, flash, fireball, death, 0])
    raw = {
        "frame": 1,
        "humans": humans,
        "fireballs": [],
        "meteors": [],
        "balls": [[20, 20, -1, 0], [30, 30, carrier, 1]],
        "scores": [0, 0],
        "bonus": [0, 0],
    }
    map_data = {"width": 40, "height": 40, "walls": [[0] * 40 for _ in range(40)]}
    return parse_observation(raw, faction=0, map_data=map_data)


def test_action_requires_exactly_five_humans():
    invalid = raw_action()
    invalid["move"] = []
    with pytest.raises(ActionValidationError, match="move must contain 5"):
        canonicalize_action(invalid)


def test_integer_and_float_coordinates_canonicalize_equal():
    assert canonicalize_action(raw_action((1, 2))) == canonicalize_action(raw_action((1.0, 2.0)))


def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ActionValidationError, match="finite"):
        canonicalize_action(raw_action((math.nan, 2)))


def test_dead_human_can_only_noop():
    action = canonicalize_action(raw_action((10.1, 10.0)))
    mask = action_mask(observation(death=4), action)
    assert mask.humans[0]["move"] == "human_dead"


def test_move_distance_and_wall_are_checked():
    far = canonicalize_action(raw_action((11.0, 10.0)))
    assert action_mask(observation(), far).humans[0]["move"] == "move_out_of_range"

    obs = observation()
    obs.map_data["walls"][10][10] = 1
    near = canonicalize_action(raw_action((10.1, 10.0)))
    assert action_mask(obs, near).humans[0]["move"] == "destination_is_wall"


def test_flash_rejects_carrier_and_meteor_checks_range():
    flash_raw = raw_action((20, 10))
    flash_raw["flash"][0] = True
    assert action_mask(observation(carrier=0), canonicalize_action(flash_raw)).humans[0]["flash"] == "carrier_cannot_flash"

    meteor_raw = raw_action((-1, -1))
    meteor_raw["meteor"][0] = [39, 39]
    assert action_mask(observation(), canonicalize_action(meteor_raw)).humans[0]["meteor"] == "meteor_out_of_range"


def test_support_pair_has_one_or_two_dirac_atoms():
    left = canonicalize_action(raw_action((10.1, 10)))
    same = canonicalize_action(raw_action((10.1, 10.0)))
    right = canonicalize_action(raw_action((9.9, 10)))
    assert support_pair(left, same) == (left,)
    assert support_pair(left, right) == (left, right)


def test_terminal_frame_is_explicit():
    assert termination_from_raw({"frame": -1}).terminated is True
    assert termination_from_raw({"frame": 3}).terminated is False

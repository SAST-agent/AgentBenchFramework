import math


def _case(*steps, terminal_support=None, public_summary=None, state_id="replay-a:0:P0"):
    return {
        "state_id": state_id,
        "steps": list(steps),
        "terminal_support": terminal_support or [[0, -1, -1], [11, 4, 5]],
        "public_summary": public_summary or {
            "round_index": 1,
            "role": "P0",
            "coins": {"self": 50, "enemy": 50},
        },
    }


def _step(selected, support=None):
    return {
        "selected": selected,
        "support": support or [[0, -1, -1], [11, 4, 5]],
    }


def test_identical_deterministic_atomic_policies_have_zero_kl():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    probe = {"cases": [_case(_step([11, 4, 5]))]}

    result = compare_probe_outputs(probe, probe, epsilon=0.05)

    assert result.status == "complete"
    assert result.decision_count == 1
    assert result.changed_action_count == 0
    assert result.details["mean_kl_nats_per_decision"] == 0.0
    assert result.details["state_examples"] == [
        {
            "state_id": "replay-a:0:P0",
            "public_summary": {
                "round_index": 1,
                "role": "P0",
                "coins": {"self": 50, "enemy": 50},
            },
            "parent_selected": [11, 4, 5],
            "candidate_selected": [11, 4, 5],
        }
    ]


def test_different_atomic_actions_have_positive_epsilon_smoothed_kl():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    parent = {"cases": [_case(_step([0, -1, -1]))]}
    candidate = {"cases": [_case(_step([11, 4, 5]))]}

    result = compare_probe_outputs(parent, candidate, epsilon=0.05)

    assert result.changed_action_count == 1
    assert result.decision_count == 1
    assert math.isfinite(result.details["mean_kl_nats_per_decision"])
    assert result.details["mean_kl_nats_per_decision"] > 0
    assert result.details["changed_examples"] == [
        {
            "state_id": "replay-a:0:P0",
            "step_index": 0,
            "parent_selected": [0, -1, -1],
            "candidate_selected": [11, 4, 5],
        }
    ]


def test_extra_bundle_atom_is_compared_against_hold_at_same_frozen_state():
    from agentbench_frame.games.antwar2.measurement import compare_probe_outputs

    parent = {
        "cases": [
            _case(
                _step([11, 4, 5]),
                terminal_support=[[0, -1, -1], [31, -1, -1]],
            )
        ]
    }
    candidate = {
        "cases": [
            _case(
                _step([11, 4, 5]),
                _step([31, -1, -1], [[0, -1, -1], [31, -1, -1]]),
                terminal_support=[[0, -1, -1]],
            )
        ]
    }

    result = compare_probe_outputs(parent, candidate, epsilon=0.05)

    assert result.decision_count == 2
    assert result.changed_action_count == 1
    assert result.details["state_count"] == 1


def test_parent_occupancy_summary_reports_raw_ranges_and_atomic_counts():
    from agentbench_frame.games.antwar2.measurement import summarize_probe_occupancy

    first = _case(
        _step([0, -1, -1]),
        state_id="reference-0:0:P0",
        public_summary={
            "round_index": 1,
            "role": "P0",
            "coins": {"self": 5, "enemy": 50},
            "tower_count": {"self": 2, "enemy": 0},
        },
    )
    second = _case(
        _step([11, 4, 5]),
        state_id="reference-0:20:P0",
        public_summary={
            "round_index": 21,
            "role": "P0",
            "coins": {"self": 41, "enemy": 42},
            "tower_count": {"self": 3, "enemy": 2},
        },
    )

    value = summarize_probe_occupancy({"cases": [first, second]})

    assert value["state_count"] == 2
    assert value["roles"]["P0"]["observed_ranges"]["self_coins"] == {
        "min": 5,
        "max": 41,
    }
    assert value["roles"]["P0"]["observed_ranges"]["self_tower_count"] == {
        "min": 2,
        "max": 3,
    }
    assert value["roles"]["P0"]["first_atomic_action_counts"] == [
        {"atom": [0, -1, -1], "count": 1},
        {"atom": [11, 4, 5], "count": 1},
    ]
    assert value["roles"]["P0"]["legal_operation_type_state_counts"] == [
        {"operation_type": 0, "state_count": 2},
        {"operation_type": 11, "state_count": 2},
    ]
    assert value["state_examples"][0]["legal_operation_types"] == [0, 11]
    assert value["state_examples"][0]["legal_atomic_actions"] == [
        [0, -1, -1],
        [11, 4, 5],
    ]


def test_parent_occupancy_examples_preserve_rare_parent_action_types():
    from agentbench_frame.games.antwar2.measurement import summarize_probe_occupancy

    cases = []
    for index in range(20):
        step = (
            _step(
                [11, 8, 9],
                support=[[0, -1, -1], [11, 4, 5], [11, 8, 9]],
            )
            if index == 10
            else _step([0, -1, -1])
        )
        cases.append(
            _case(step, state_id=f"reference-0:{index:03d}:P0")
        )

    value = summarize_probe_occupancy({"cases": cases}, max_examples=4)

    assert any(
        example["parent_selected"] == [11, 8, 9]
        for example in value["state_examples"]
    )
    rare = next(
        example
        for example in value["state_examples"]
        if example["parent_selected"] == [11, 8, 9]
    )
    assert [11, 4, 5] in rare["legal_atomic_actions"]
    assert [11, 8, 9] in rare["legal_atomic_actions"]


def test_terminal_replay_snapshot_is_not_a_valid_decision_state():
    from agentbench_frame.games.antwar2.policy_probe import _is_terminal_state

    assert _is_terminal_state({"winner": 0, "camps": [6, 0]})
    assert _is_terminal_state({"winner": 1, "camps": [0, 4]})
    assert not _is_terminal_state({"winner": -1, "camps": [50, 50]})


def test_frozen_occupancy_sampling_is_deterministic_and_excludes_terminal():
    from agentbench_frame.games.antwar2.policy_probe import _sample_records

    replay = [
        {"round_state": {"winner": -1}, "round": index}
        for index in range(10)
    ] + [{"round_state": {"winner": 0}, "round": 10}]

    first = _sample_records(replay, max_states=4)
    second = _sample_records(replay, max_states=4)

    assert first == second
    assert len(first) == 4
    assert all(record[1]["round_state"]["winner"] == -1 for record in first)
    assert first[0][0] == 0
    assert first[-1][0] == 9

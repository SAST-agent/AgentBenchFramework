import pytest

from agentbench_frame.generals.measurement import (
    BehaviorChange,
    DecisionClassSummary,
    ProbeState,
    classify_macro_action,
    evaluate_behavior_gate,
    measure_action_disagreement,
    measure_occupancy_shift,
    summarize_decision_classes,
)


def test_action_disagreement_compares_same_probe_states():
    probes = (
        ProbeState("s1", {"round": 1}, ((8,),)),
        ProbeState("s2", {"round": 2}, ((1, 0, 0, 4, 2), (8,))),
    )
    measured = measure_action_disagreement(
        probes,
        new_actions={"s1": ((8,),), "s2": ((8,),)},
    )
    assert measured.trace == (0.0, 1.0)
    assert measured.mean == 0.5


def test_measurement_reports_policy_kl_unavailable():
    measured = BehaviorChange(trace=(0.0,), mean=0.0)
    assert measured.policy_kl is None
    assert measured.policy_kl_status == "complete_macro_action_distribution_unavailable"


def test_occupancy_is_separate_and_nonnegative():
    value = measure_occupancy_shift(("a", "a", "b"), ("a", "c"), smoothing=1e-12)
    assert value >= 0


def test_disagreement_rejects_missing_probe():
    with pytest.raises(ValueError, match="exactly"):
        measure_action_disagreement((ProbeState("s", {}, ((8,),)),), {})


def state_with_main(position=(4, 5), seat=0):
    return {
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": seat,
                "position": list(position),
            }
        }
    }


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (((8,),), "end_only"),
        (((1, 4, 5, 4, 3), (8,)), "main_army_move"),
        (((1, 4, 6, 4, 3), (8,)), "non_main_army_move"),
        (((2, 0, 4), (8,)), "general_move"),
        (((3, 0, 1), (8,)), "general_upgrade"),
        (((4, 0, 1), (8,)), "skill"),
        (((5, 1), (8,)), "technology"),
        (((6, 1, 3, 4), (8,)), "super_weapon"),
        (((7, 4, 5), (8,)), "recruit"),
        (((99,), (8,)), "other"),
        ((), "other"),
        (((8,), (8,)), "other"),
    ],
)
def test_macro_action_classifies_observable_first_command(action, expected):
    assert classify_macro_action(state_with_main(), 0, action) == expected


def test_decision_summary_counts_each_probe_once():
    probes = (
        ProbeState("main", state_with_main(), ((8,),)),
        ProbeState("front", state_with_main(), ((8,),)),
        ProbeState("end", state_with_main(), ((8,),)),
    )

    summary = summarize_decision_classes(
        probes,
        {
            "main": ((1, 4, 5, 4, 2), (8,)),
            "front": ((1, 4, 6, 4, 2), (8,)),
            "end": ((8,),),
        },
    )

    assert summary.total == 3
    assert summary.counts["main_army_move"] == 1
    assert summary.counts["non_main_army_move"] == 1
    assert summary.counts["end_only"] == 1
    assert summary.rates["non_main_army_move"] == pytest.approx(1 / 3)


def passing_summary():
    return DecisionClassSummary(
        total=10,
        counts={"non_main_army_move": 2},
        rates={"non_main_army_move": 0.2},
    )


def test_behavior_gate_passes_only_when_every_predeclared_condition_passes():
    gate = evaluate_behavior_gate(
        provider_completed=True,
        protected_files_unchanged=True,
        tests_passed=True,
        action_disagreement=0.25,
        decision_classes=passing_summary(),
        validation_case_count=6,
        valid_validation_case_count=6,
        dense_deltas={
            "completed_rounds_survived": 4.0,
            "terminal_territory_share": 0.01,
        },
    )

    assert gate.passed is True
    assert all(gate.conditions.values())
    assert gate.improved_dense_metrics == (
        "completed_rounds_survived",
        "terminal_territory_share",
    )


@pytest.mark.parametrize(
    ("overrides", "failed_condition"),
    [
        ({"action_disagreement": 0.0}, "behavior_changed"),
        (
            {
                "decision_classes": DecisionClassSummary(
                    total=10,
                    counts={"non_main_army_move": 0},
                    rates={"non_main_army_move": 0.0},
                )
            },
            "frontline_move_observed",
        ),
        ({"valid_validation_case_count": 5}, "validation_complete"),
        (
            {
                "dense_deltas": {
                    "completed_rounds_survived": 0.0,
                    "terminal_territory_share": -0.01,
                }
            },
            "dense_progress_observed",
        ),
    ],
)
def test_behavior_gate_rejects_failed_condition(overrides, failed_condition):
    inputs = {
        "provider_completed": True,
        "protected_files_unchanged": True,
        "tests_passed": True,
        "action_disagreement": 0.25,
        "decision_classes": passing_summary(),
        "validation_case_count": 6,
        "valid_validation_case_count": 6,
        "dense_deltas": {"terminal_territory_share": 0.01},
    }
    inputs.update(overrides)

    gate = evaluate_behavior_gate(**inputs)

    assert gate.passed is False
    assert gate.conditions[failed_condition] is False

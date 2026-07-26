import pytest

from agentbench_frame.generals.measurement import (
    BehaviorChange,
    ProbeState,
    measure_action_disagreement,
    measure_occupancy_shift,
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

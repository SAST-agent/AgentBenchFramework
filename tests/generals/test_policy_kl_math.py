from agentbench_frame.generals.policy_kl_math import (
    compute_controlled_policy_kl,
)


EPSILONS = ("0.001", "0.01", "0.05", "0.1")


def test_computation_preserves_transition_state_epsilon_order():
    result = compute_controlled_policy_kl(
        versions=("v6", "v7"),
        reference_state_ids=("s0", "s1"),
        counts={"s0": 13, "s1": 1050},
        actions={
            ("v6", "s0"): ((5, 1), (8,)),
            ("v7", "s0"): ((5, 1), (8,)),
            ("v6", "s1"): ((5, 1), (8,)),
            ("v7", "s1"): ((5, 2), (8,)),
        },
        missing_actions={},
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        action_space_spec_id="spec",
    )

    assert [
        (fact["measurement_state_id"], fact["epsilon"])
        for fact in result.facts
    ] == [
        (state_id, epsilon)
        for state_id in ("s0", "s1")
        for epsilon in EPSILONS
    ]
    assert result.facts[0]["kl_nats_decimal"] == "0"
    assert result.facts[5]["kl_nats_decimal"] == (
        "11.436158164117011925846995005703779814217598025922366770430132745974673222046936"
    )
    transition = result.metric["transitions"][0]
    assert transition["mean_kl_nats_decimal"] == (
        "5.718079082058505962923497502851889907108799012961183385215066372987336611023468"
    )
    assert transition["coverage"] == {"complete": 2, "total": 2}
    assert transition["action_disagreement_rate"] == 0.5
    assert result.metric["direction"] == "new||old"
    assert result.complete is True


def test_missing_action_preserves_four_null_facts_and_null_aggregate():
    result = compute_controlled_policy_kl(
        versions=("v6", "v7"),
        reference_state_ids=("s0", "s1"),
        counts={"s0": 13, "s1": 1050},
        actions={
            ("v6", "s0"): ((5, 1), (8,)),
            ("v7", "s0"): ((5, 1), (8,)),
            ("v6", "s1"): ((5, 1), (8,)),
        },
        missing_actions={("v7", "s1"): "v7 probe was nondeterministic"},
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        action_space_spec_id="spec",
    )

    missing = [
        fact
        for fact in result.facts
        if fact["measurement_state_id"] == "s1"
    ]
    assert len(missing) == 4
    assert all(fact["status"] == "missing" for fact in missing)
    assert all(fact["kl_nats"] is None for fact in missing)
    assert all(
        fact["missing_reasons"] == ["v7 probe was nondeterministic"]
        for fact in missing
    )
    transition = result.metric["transitions"][0]
    assert transition["mean_kl_nats"] is None
    assert transition["coverage"] == {"complete": 1, "total": 2}
    assert transition["action_disagreement_rate"] is None
    assert result.complete is False


def test_missing_support_is_reported_without_indexing_the_count_map():
    result = compute_controlled_policy_kl(
        versions=("v6", "v7"),
        reference_state_ids=("s0",),
        counts={},
        actions={
            ("v6", "s0"): ((5, 1), (8,)),
            ("v7", "s0"): ((5, 2), (8,)),
        },
        missing_actions={},
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        action_space_spec_id="spec",
    )

    assert all(fact["support_size"] is None for fact in result.facts)
    assert all(
        fact["missing_reasons"] == ["missing exact support count"]
        for fact in result.facts
    )
    assert result.metric["support_size"] == {
        "complete": 0,
        "total": 1,
        "minimum": None,
        "maximum": None,
    }

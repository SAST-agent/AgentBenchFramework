from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import load_expanded_kl_config
from agentbench_frame.generals.engine import OfficialGeneralsEngine


ASSET_ROOT = Path(os.environ.get(
    "AGENTBENCH_ASSET_ROOT",
    "/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets",
))
ENGINE_ROOT = ASSET_ROOT / "backend_sources/corpus/28_generals/logic/gamecode_logic"
MANIFEST = ASSET_ROOT / "backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.toml"
EXPECTED_SCENARIOS = {
    "contact": 2,
    "main_general_danger": 2,
    "large_stack_routing": 2,
    "economy_combat_conflict": 2,
    "counter_capture": 2,
    "mid_late_consolidation": 2,
}


def test_intervention_pack_is_exact_balanced_and_round_trips(tmp_path):
    from agentbench_frame.generals.intervention_states import (
        assert_intervention_scenario,
        build_intervention_state_pack,
    )

    config = load_expanded_kl_config(MANIFEST)
    pack = build_intervention_state_pack(ENGINE_ROOT, config)

    assert pack.measurement_id == "generals-policy-kl-expanded-v1"
    assert len(pack.states) == 12
    assert Counter(item.scenario for item in pack.states) == EXPECTED_SCENARIOS
    assert Counter(item.actor for item in pack.states) == {0: 6, 1: 6}
    assert len({item.measurement_state_id for item in pack.states}) == 12
    assert [item.state_key for item in pack.states] == [
        item.state_key for item in config.intervention_states
    ]
    for item in pack.states:
        restored = OfficialGeneralsEngine.from_measurement_state(
            ENGINE_ROOT,
            item.snapshot,
            tmp_path / f"{item.state_key}.jsonl",
        )
        assert restored.measurement_state(item.actor) == item.snapshot
        assert restored.state.winner == -1
        assert {general.player for general in restored.state.generals} == {0, 1}
        assert len(restored.state.generals) == 2
        assert assert_intervention_scenario(item) is None
        outcome = restored.apply_turn(item.actor, ((8,),))
        assert outcome.done is False
        assert item.assertion_receipt["scenario_predicate"] is True
        assert item.assertion_receipt["both_mains_present"] is True
        assert item.assertion_receipt["legal_end_macro"] is True


def test_intervention_pack_generation_is_byte_stable_and_hash_bound():
    from agentbench_frame.generals.intervention_states import (
        build_intervention_state_pack,
    )

    config = load_expanded_kl_config(MANIFEST)
    first = build_intervention_state_pack(ENGINE_ROOT, config)
    second = build_intervention_state_pack(ENGINE_ROOT, config)

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.sha256 == second.sha256
    assert first.sha256 == hashlib.sha256(first.canonical_bytes()).hexdigest()
    assert first.canonical_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("scenario", "required"),
    [
        ("contact", {"owned_army": 7, "hostile_army": 2}),
        ("main_general_danger", {"main_army": 13, "hostile_army": 20}),
        ("large_stack_routing", {"small_stack": 4, "large_stack": 30}),
        ("economy_combat_conflict", {"coins": 40, "owned_army": 7, "hostile_army": 2}),
        ("counter_capture", {"owned_army": 9, "hostile_army": 5}),
        ("mid_late_consolidation", {"round": 120, "stack_a": 18, "stack_b": 14}),
    ],
)
def test_intervention_recipe_receipts_freeze_requested_numbers(scenario, required):
    from agentbench_frame.generals.intervention_states import (
        build_intervention_state_pack,
    )

    config = load_expanded_kl_config(MANIFEST)
    states = [
        item for item in build_intervention_state_pack(ENGINE_ROOT, config).states
        if item.scenario == scenario
    ]
    assert len(states) == 2
    for state in states:
        for key, value in required.items():
            assert state.construction_receipt[key] == value


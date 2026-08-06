from pathlib import Path

from agentbench_frame.generals.ablation_v9 import AttributionPolicy
from agentbench_frame.generals.historical_policy import PolicyProbeResult
from agentbench_frame.generals.models import MatchResult, TurnRecord
from agentbench_frame.generals.measurement_state import measurement_state_id
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _state(
    *,
    seat=0,
    main_army=20,
    adjacent_enemy=0,
    contact=False,
    large_stack=0,
):
    cells = {
        "7,4": {"type": 0, "player": seat, "army": main_army, "general_id": 1},
        "7,10": {"type": 0, "player": 1 - seat, "army": 20, "general_id": 2},
    }
    if adjacent_enemy:
        cells["7,5"] = {
            "type": 0,
            "player": 1 - seat,
            "army": adjacent_enemy,
            "general_id": None,
        }
    if contact:
        cells["5,5"] = {"type": 0, "player": seat, "army": 7, "general_id": None}
        cells["5,6"] = {"type": 0, "player": 1 - seat, "army": 2, "general_id": None}
    if large_stack:
        cells["9,9"] = {
            "type": 0,
            "player": seat,
            "army": large_stack + 1,
            "general_id": None,
        }
    return {
        "round": 1,
        "my_seat": seat,
        "cells": cells,
        "generals": {
            "1": {"id": 1, "type": "main", "player": seat, "position": [7, 4]},
            "2": {"id": 2, "type": "main", "player": 1 - seat, "position": [7, 10]},
        },
        "coins": [40, 40],
    }


def _turn(step, state_id, state, action):
    return TurnRecord(
        step=step,
        round_number=step + 1,
        player=0,
        state_id_before=state_id,
        state_before=state,
        commands=tuple(tuple(item) for item in action),
        state_id_after=f"after-{state_id}",
        state_after=state,
    )


def _match(seed, *turns):
    return MatchResult(
        case_id=f"attribute9-high-opponent-s{seed}-p0",
        valid=True,
        winner=1,
        termination_type="normal",
        seed=seed,
        evaluated_seat=0,
        turns=tuple(turns),
        elapsed_time_s=1.0,
        engine_hash="engine",
    )


def _measurement(actor, marker):
    payload = {
        "schema": "generals-measurement-state-v1",
        "actor": actor,
        "state": {"round": marker},
    }
    return payload


def test_diagnostic_selection_is_bounded_prioritized_and_same_state_ready():
    from agentbench_frame.generals.attribution_diagnostics import (
        diagnostic_missing_reasons,
        select_diagnostic_states,
    )

    danger = _state(main_army=1, adjacent_enemy=10)
    contact = _state(contact=True)
    large = _state(large_stack=30)
    economy = _state(contact=True)
    v7 = _match(
        302101,
        _turn(0, "danger", danger, [[8]]),
        _turn(1, "contact", contact, [[1, 5, 5, 4, 6], [8]]),
        _turn(2, "terminal-v7", _state(), [[8]]),
    )
    v8 = _match(
        302101,
        _turn(0, "large", large, [[8]]),
        _turn(1, "economy", economy, [[3, 1, 1], [8]]),
        _turn(2, "terminal-v8", _state(), [[8]]),
    )
    snapshots = {}
    for version, match in (("v7", v7), ("v8", v8)):
        for turn in match.turns:
            version_offset = 0 if version == "v7" else 10
            snapshot = _measurement(turn.player, 100 + version_offset + turn.step)
            snapshots[(version, f"s{match.seed}-p0", turn.step)] = snapshot

    selected = select_diagnostic_states(
        (v7,),
        (v8,),
        measurement_states=snapshots,
        max_states=48,
    )

    assert len(selected) == 4
    assert [item.reason for item in selected[:2]] == [
        "first_main_general_danger",
        "first_enemy_contact",
    ]
    assert [item.source_version for item in selected] == ["v7", "v7", "v8", "v8"]
    assert len({item.measurement_state_id for item in selected}) == 4
    assert all(
        item.measurement_state_id == measurement_state_id(item.measurement_state)
        for item in selected
    )
    assert diagnostic_missing_reasons((v7,), (v8,)) == {
        "s302101-p0": ("first_action_divergence",)
    }


def test_diagnostic_selection_caps_two_states_per_source_and_four_per_pair():
    from agentbench_frame.generals.attribution_diagnostics import (
        select_diagnostic_states,
    )

    turns = tuple(
        _turn(index, f"state-{index}", _state(contact=True), [[8]])
        for index in range(6)
    )
    v7 = _match(302101, *turns)
    v8 = _match(302101, *turns)
    snapshots = {
        (version, "s302101-p0", turn.step): _measurement(0, 100 + turn.step)
        for version in ("v7", "v8")
        for turn in turns
    }

    selected = select_diagnostic_states(
        (v7,),
        (v8,),
        measurement_states=snapshots,
        max_states=48,
    )

    assert len(selected) <= 4
    assert sum(item.source_version == "v7" for item in selected) <= 2
    assert sum(item.source_version == "v8" for item in selected) <= 2
    assert all(
        item.replay_ref.startswith(
            "matches/A/" if item.source_version == "v7" else "matches/D/"
        )
        for item in selected
    )


def test_compare_same_state_actions_probes_every_policy_on_identical_hash(tmp_path):
    from agentbench_frame.generals.attribution_diagnostics import (
        DiagnosticState,
        compare_same_state_actions,
    )

    snapshot = _measurement(0, 7)
    state_id = measurement_state_id(snapshot)
    state = DiagnosticState(
        pair_id="s302101-p0",
        source_version="v7",
        reason="first_enemy_contact",
        round_number=7,
        actor=0,
        measurement_state_id=state_id,
        measurement_state=snapshot,
        replay_ref="matches/A/case/replay.jsonl",
    )
    policies = []
    for cell in ("A", "B", "C", "D"):
        source = tmp_path / cell
        source.mkdir()
        (source / "main.py").write_text("def agent(*args): return [[8]]\n")
        manifest = LocalWorkspaceSnapshotter().capture(source)
        policies.append(AttributionPolicy(
            cell=cell,
            policy_id=f"policy-{cell}",
            source=source,
            content_hash=manifest.content_hash,
            interventions=(),
            source_authority="fixture",
        ))

    def probe(policy, measurement_state, **kwargs):
        del measurement_state, kwargs
        action = ((8,),) if policy.version in {"policy-A", "policy-C"} else ((1, 1, 1, 4, 1), (8,))
        return PolicyProbeResult(
            version=policy.version,
            status="complete",
            deterministic=True,
            raw_actions=(action, action),
            elapsed_time_s=0.01,
            stdout="",
            stderr="",
        )

    probes = compare_same_state_actions(
        (state,),
        tuple(policies),
        engine_root=Path("/engine"),
        sdk_root=Path("/sdk"),
        probe_policy=probe,
        canonicalize_action=lambda _, action: tuple(tuple(item) for item in action),
    )

    assert len(probes) == 4
    assert {item.measurement_state_id for item in probes} == {state_id}
    assert [item.policy_cell for item in probes] == ["A", "B", "C", "D"]
    assert all(item.status == "complete" and item.legal for item in probes)
    assert probes[0].canonical_action != probes[1].canonical_action


def test_earliest_trajectory_divergence_requires_shared_pre_state():
    from agentbench_frame.generals.attribution_diagnostics import (
        earliest_trajectory_divergence,
    )

    shared = _state()
    v7 = _match(302101, _turn(0, "shared", shared, [[8]]))
    v8 = _match(302101, _turn(0, "shared", shared, [[1, 7, 4, 4, 1], [8]]))

    divergence = earliest_trajectory_divergence(v7, v8)

    assert divergence is not None
    assert divergence.pair_id == "s302101-p0"
    assert divergence.round_number == 1
    assert divergence.state_id == "shared"
    assert divergence.v7_action == ((8,),)
    assert divergence.v8_action[0][0] == 1

import copy
import json
import os

import pytest

from agentbench_frame.games.miracle import ifelse_policy_state_v1 as policy_state
from agentbench_frame.games.miracle.decision_kl_v1 import (
    build_trusted_action_support,
)


def _explicit_config(*, opening="FF"):
    values = {}
    for spec in policy_state.MIRACLE_CONFIG_SPECS:
        if spec.name == "MIRACLE_CAMP1_OPENING":
            values[spec.name] = opening
        elif spec.name == "MIRACLE_ARTIFACT":
            values[spec.name] = "InfernoFlame"
        elif spec.name == "MIRACLE_DECK":
            values[spec.name] = ["Priest", "Archer", "Swordsman"]
        else:
            values[spec.name] = False
    return policy_state.PolicyConfigV1.from_explicit(values)


def _observation(*, camp=0, round_number=0):
    return {
        "round": round_number,
        "camp": camp,
        "map": {"units": [], "barracks": [-1] * 4, "miracles": [30, 30]},
        "players": [
            [[[0, 2, 8, 6, 0, 0, 0, [-1, -1, -1]]], 2, 2,
             [[3, 3, []], [0, 3, []], [1, 6, []]], []],
            [[[1, 2, 8, 6, 0, 0, 0, [-1, -1, -1]]], 2, 2,
             [[3, 3, []], [0, 3, []], [1, 6, []]], []],
        ],
    }


def _init_command(*, camp=0, artifacts=None, creatures=None):
    return {
        "player": camp,
        "round": 0,
        "operation_type": "init",
        "operation_parameters": {
            "artifacts": ["InfernoFlame"] if artifacts is None else artifacts,
            "creatures": (
                ["Priest", "Archer", "Swordsman"]
                if creatures is None
                else creatures
            ),
        },
    }


def _install_init_selector(monkeypatch):
    monkeypatch.setattr(
        policy_state,
        "_select_legacy_command",
        lambda *args: policy_state.SelectedCommandV1(
            _init_command(), "turn_start", "init"
        ),
    )


def test_config_contract_freezes_all_62_inputs_and_has_stable_identity():
    assert len(policy_state.MIRACLE_CONFIG_SPECS) == 62
    config = _explicit_config()
    assert config.complete is True
    assert len(config.values) == 62
    assert config.sha256 == policy_state.PolicyConfigV1.from_explicit(
        dict(config.to_dict()["values"])
    ).sha256
    assert json.loads(config.canonical_bytes)["schema_version"] == (
        policy_state.POLICY_CONFIG_SCHEMA_VERSION
    )


def test_missing_or_unknown_config_is_not_formal_policy_evidence():
    values = dict(_explicit_config().to_dict()["values"])
    values.pop("MIRACLE_GATE_DEFENSE")
    with pytest.raises(ValueError, match="exactly all 62"):
        policy_state.PolicyConfigV1.from_explicit(values)

    historical = policy_state.PolicyConfigV1.historical_unknown(
        observed={
            "MIRACLE_ARTIFACT": "InfernoFlame",
            "MIRACLE_DECK": ["Priest", "Archer", "Swordsman"],
        }
    )
    assert historical.complete is False
    assert historical.values["MIRACLE_GATE_DEFENSE"] == "unknown"
    with pytest.raises(policy_state.IncompletePolicyEvidenceError):
        historical.require_complete()


def test_config_types_and_opening_version_are_strict():
    values = dict(_explicit_config().to_dict()["values"])
    values["MIRACLE_GATE_DEFENSE"] = 1
    with pytest.raises(TypeError, match="MIRACLE_GATE_DEFENSE"):
        policy_state.PolicyConfigV1.from_explicit(values)
    values = dict(_explicit_config().to_dict()["values"])
    values["MIRACLE_CAMP1_OPENING"] = "XX"
    with pytest.raises(ValueError, match="MIRACLE_CAMP1_OPENING"):
        policy_state.PolicyConfigV1.from_explicit(values)

    values = dict(_explicit_config().to_dict()["values"])
    values["MIRACLE_ARTIFACT"] = "AttackerInventedArtifact"
    with pytest.raises(ValueError, match="MIRACLE_ARTIFACT"):
        policy_state.PolicyConfigV1.from_explicit(values)

    values = dict(_explicit_config().to_dict()["values"])
    values["MIRACLE_DECK"] = ["Priest", "Archer", "AttackerInventedUnit"]
    with pytest.raises(ValueError, match="MIRACLE_DECK"):
        policy_state.PolicyConfigV1.from_explicit(values)


def test_memory_is_issuer_bound_serializable_and_reset_is_unique():
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    first = machine.reset(camp=0, episode_id="episode-1")
    second = machine.reset(camp=0, episode_id="episode-1")
    assert first.to_dict() == second.to_dict()
    assert first.schema_version == policy_state.POLICY_MEMORY_SCHEMA_VERSION
    assert first.phase == "turn_start"
    assert first.decision_step == 1
    json.dumps(first.to_dict(), allow_nan=False)

    copied = copy.copy(first)
    with pytest.raises(ValueError, match="issued"):
        copied.to_dict()
    with pytest.raises(TypeError, match="issued"):
        policy_state.PolicyMemoryV1()


def test_mutated_replaced_cross_episode_or_wrong_identity_memory_is_rejected():
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    object.__setattr__(memory, "decision_step", 9)
    with pytest.raises(ValueError, match="issued"):
        machine.validate_memory(memory)

    other_episode = machine.reset(camp=0, episode_id="episode-2")
    with pytest.raises(ValueError, match="episode"):
        machine.validate_memory(other_episode, episode_id="episode-1")

    other_context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v2", config=_explicit_config()
    )
    other_machine = policy_state.ExplicitIfElseStateMachineV1(other_context)
    with pytest.raises(ValueError, match="policy identity"):
        other_machine.validate_memory(
            machine.reset(camp=0, episode_id="episode-1")
        )


def test_pair_provider_recomputes_strict_one_hot_on_same_state_memory_and_support(
    monkeypatch,
):
    config = _explicit_config()
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=config
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = {"camp": 0}
    support = build_trusted_action_support(observation)

    def fake_select(version, observation, memory, config):
        assert memory.decision_step == 1
        assert observation["camp"] == 0
        command = {
            "player": 0,
            "round": 0,
            "operation_type": "init",
            "operation_parameters": {
                "artifacts": ["InfernoFlame"],
                "creatures": ["Priest", "Archer", "Swordsman"],
            },
        }
        return policy_state.SelectedCommandV1(command, "turn_start")

    monkeypatch.setattr(policy_state, "_select_legacy_command", fake_select)
    decision = machine.evaluate_pair(observation, memory, support)
    assert decision.old_action_id == decision.new_action_id
    assert set(decision.old_distribution) == set(support.action_ids)
    assert set(decision.new_distribution) == set(support.action_ids)
    assert sum(decision.old_distribution.values()) == 1.0
    assert sum(decision.new_distribution.values()) == 1.0
    assert set(decision.old_distribution.values()) <= {0.0, 1.0}
    assert set(decision.new_distribution.values()) <= {0.0, 1.0}
    assert decision.memory_before_sha256 == memory.sha256
    assert decision.m_after.decision_step == 2
    assert decision.m_after.previous_transition_sha256 == decision.transition_sha256


def test_provider_rejects_incomplete_or_forged_support_and_uploaded_distribution(
    monkeypatch,
):
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = _observation()
    support = build_trusted_action_support(observation)

    command = {
        "player": 0,
        "round": 0,
        "operation_type": "init",
        "operation_parameters": {
            "artifacts": ["InfernoFlame"],
            "creatures": ["Priest", "Archer", "Swordsman"],
        },
    }
    monkeypatch.setattr(
        policy_state,
        "_select_legacy_command",
        lambda *args: policy_state.SelectedCommandV1(command, "turn_start"),
    )
    forged = copy.deepcopy(support)
    object.__setattr__(forged, "schema_version", "forged")
    with pytest.raises(ValueError, match="support"):
        machine.evaluate_pair(observation, memory, forged)
    assert "distribution" not in machine.evaluate_pair.__annotations__


def test_transition_tip_rejects_replayed_or_skipped_memory(monkeypatch):
    _install_init_selector(monkeypatch)
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = {"camp": 0}
    support = build_trusted_action_support(observation)

    first = machine.evaluate_pair(observation, memory, support)
    with pytest.raises(ValueError, match="current transition tip"):
        machine.evaluate_pair(observation, memory, support)

    skipped = machine.reset(camp=0, episode_id="episode-2")
    object.__setattr__(skipped, "decision_step", 3)
    with pytest.raises(ValueError, match="issued"):
        machine.evaluate_pair(observation, skipped, support)

    second = machine.evaluate_pair(observation, first.m_after, support)
    assert second.decision_step == 2
    assert second.episode_id == "episode-1"
    assert second.memory_before_sha256 == first.m_after.sha256


def test_pair_decision_is_issuer_bound_and_snapshot_protected(monkeypatch):
    _install_init_selector(monkeypatch)
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = {"camp": 0}
    support = build_trusted_action_support(observation)
    decision = machine.evaluate_pair(observation, memory, support)

    machine.validate_decision(
        decision,
        episode_id="episode-1",
        decision_step=1,
        observation=observation,
        complete_action_support=support,
    )
    json.dumps(decision.to_dict(), allow_nan=False)

    copied = copy.copy(decision)
    with pytest.raises(ValueError, match="issued decision"):
        machine.validate_decision(
            copied,
            episode_id="episode-1",
            decision_step=1,
            observation=observation,
            complete_action_support=support,
        )

    object.__setattr__(decision, "old_action_id", "forged")
    with pytest.raises(ValueError, match="issued decision"):
        machine.validate_decision(
            decision,
            episode_id="episode-1",
            decision_step=1,
            observation=observation,
            complete_action_support=support,
        )

    with pytest.raises(TypeError, match="issued"):
        policy_state.PolicyPairDecisionV1()


def test_pair_decision_binds_context_step_and_distinct_provider_choices(monkeypatch):
    config = _explicit_config()
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=config
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = {"camp": 0}
    support = build_trusted_action_support(observation)
    seen = []

    def different_select(provider_identity, observation, supplied_memory, config):
        seen.append((provider_identity, supplied_memory))
        creatures = ["Priest", "Archer", "Swordsman"]
        if provider_identity == context.new_provider.sha256:
            creatures = ["Archer", "Priest", "Swordsman"]
        return policy_state.SelectedCommandV1(
            _init_command(creatures=creatures), "turn_start", "init"
        )

    monkeypatch.setattr(policy_state, "_select_legacy_command", different_select)
    decision = machine.evaluate_pair(observation, memory, support)
    assert decision.old_action_id != decision.new_action_id
    assert seen == [
        (context.old_provider.sha256, memory),
        (context.new_provider.sha256, memory),
    ]
    with pytest.raises(ValueError, match="episode"):
        machine.validate_decision(
            decision,
            episode_id="another-episode",
            decision_step=1,
            observation=observation,
            complete_action_support=support,
        )
    with pytest.raises(ValueError, match="step"):
        machine.validate_decision(
            decision,
            episode_id="episode-1",
            decision_step=2,
            observation=observation,
            complete_action_support=support,
        )


def test_invalid_state_transition_fields_fail_closed(monkeypatch):
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    memory = machine.reset(camp=0, episode_id="episode-1")
    observation = {"camp": 0}
    support = build_trusted_action_support(observation)
    monkeypatch.setattr(
        policy_state,
        "_select_legacy_command",
        lambda *args: policy_state.SelectedCommandV1(
            _init_command(),
            "move",
            "move.move",
            memory_updates={"current_unit_cursor": -1},
        ),
    )
    with pytest.raises(ValueError, match="current_unit_cursor"):
        machine.evaluate_pair(observation, memory, support)


def test_source_identity_detects_replacement(tmp_path):
    source_root = tmp_path / "policy"
    source_root.mkdir()
    for name in policy_state.LEGACY_SOURCE_FILES:
        (source_root / name).write_text(
            "{}\n" if name == "Data.json" else f"# {name}\n",
            encoding="utf-8",
            newline="\n",
        )
    source = policy_state.LegacyPolicySourceV1.open(source_root, version="v0")
    source.revalidate()
    (source_root / "main.py").write_text(
        "# replaced\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(policy_state.PolicySourceError, match="identity mismatch"):
        source.revalidate()


def test_source_loader_is_read_only_and_never_writes_bytecode(tmp_path):
    source_root = tmp_path / "policy"
    source_root.mkdir()
    for name in policy_state.LEGACY_SOURCE_FILES:
        payload = (
            "{}\n"
            if name == "Data.json"
            else "class IfElseAI:\n    pass\n"
            if name == "main.py"
            else f"# {name}\n"
        )
        (source_root / name).write_text(
            payload, encoding="utf-8", newline="\n"
        )
    source = policy_state.LegacyPolicySourceV1.open(source_root, version="v0")
    policy_state._LegacyIfElseRuntime(source)
    assert not (source_root / "__pycache__").exists()


def test_all_source_files_require_strict_utf8_without_bom(tmp_path):
    source_root = tmp_path / "policy"
    source_root.mkdir()
    for name in policy_state.LEGACY_SOURCE_FILES:
        (source_root / name).write_bytes(
            b"{}\n" if name == "Data.json" else f"# {name}\n".encode("utf-8")
        )
    (source_root / "Data.json").write_bytes(b"\xef\xbb\xbf{}\n")
    with pytest.raises(policy_state.PolicySourceError, match="BOM"):
        policy_state.LegacyPolicySourceV1.open(source_root, version="v0")


def test_trace_replacement_after_preflight_is_rejected_even_with_same_bytes(
    tmp_path, monkeypatch
):
    observation = {"camp": 0}
    encoded = json.dumps(observation, separators=(",", ":"), sort_keys=True)
    framed = f"{len(encoded.encode('utf-8')):06d}{encoded}"
    records = [
        {
            "kind": "judge_frame",
            "frame": {"listen": [0], "player": [0], "content": [framed]},
        },
        {"kind": "ai_operation", "player": 0, "operation": _init_command()},
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    _install_init_selector(monkeypatch)
    original_evaluate = machine.evaluate_pair

    def replace_after_evaluate(*args, **kwargs):
        decision = original_evaluate(*args, **kwargs)
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_bytes(trace.read_bytes())
        os.replace(replacement, trace)
        return decision

    monkeypatch.setattr(machine, "evaluate_pair", replace_after_evaluate)
    with pytest.raises(ValueError, match="changed during"):
        policy_state.audit_sequential_trace(
            trace,
            machine=machine,
            evaluated_camp=0,
            episode_id="trace-replacement",
        )


def test_sequential_audit_serializes_every_memory_before_and_pair_decision(
    tmp_path, monkeypatch
):
    observation = {"camp": 0}
    encoded = json.dumps(observation, separators=(",", ":"), sort_keys=True)
    framed = f"{len(encoded.encode('utf-8')):06d}{encoded}"
    records = [
        {
            "kind": "judge_frame",
            "frame": {"listen": [0], "player": [0], "content": [framed]},
        },
        {"kind": "ai_operation", "player": 0, "operation": _init_command()},
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    context = policy_state.PolicyComparisonIdentityV1.for_test(
        old_version="v0", new_version="v1", config=_explicit_config()
    )
    machine = policy_state.ExplicitIfElseStateMachineV1(context)
    _install_init_selector(monkeypatch)

    audit = policy_state.audit_sequential_trace(
        trace,
        machine=machine,
        evaluated_camp=0,
        episode_id="serialized-memory",
    )
    assert audit.complete is True
    payload = audit.to_dict()
    assert len(payload["decision_records"]) == 1
    record = payload["decision_records"][0]
    assert record["decision_step"] == 1
    assert record["m_before"] == audit.memory_before[0].to_dict()
    assert record["pair_decision"] == audit.decisions[0].to_dict()
    assert record["pair_decision"]["memory_before_sha256"] == audit.memory_before[0].sha256
    assert record["recorded_action_id"] == record["pair_decision"]["new_action_id"]

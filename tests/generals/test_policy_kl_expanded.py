from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agentbench_frame.generals.historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
)
from agentbench_frame.generals.intervention_states import (
    InterventionState,
    InterventionStatePack,
    PACK_SCHEMA,
)
from agentbench_frame.generals.macro_counter import ExactCountResult
from agentbench_frame.generals.models import (
    ExpandedPolicyKLConfig,
    InterventionStateSpec,
)
from agentbench_frame.generals.policy_kl_expanded import (
    EXPANDED_DOMAIN,
    LEGACY_DOMAIN,
    ExpandedPolicyKLError,
    GeneralsExpandedPolicyKLPipeline,
)
from agentbench_frame.generals.policy_kl_math import compute_controlled_policy_kl
from agentbench_frame.generals.policy_kl_reuse import (
    EXPECTED_ACTION_SPACE_SPEC_ID,
    VerifiedPolicyKLSource,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


EPSILONS = ("0.001", "0.01", "0.05", "0.1")
SCENARIOS = (
    "contact",
    "main_general_danger",
    "large_stack_routing",
    "economy_combat_conflict",
    "counter_capture",
    "mid_late_consolidation",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class FakeActionSpace:
    spec_id = EXPECTED_ACTION_SPACE_SPEC_ID

    @staticmethod
    def canonicalize(_state, action):
        return tuple(tuple(command) for command in action)


class IllegalActionSpace(FakeActionSpace):
    @staticmethod
    def canonicalize(_state, _action):
        raise ValueError("illegal official primitive")


class FakeCounter:
    def __init__(self, incomplete_index=None, approximate=False):
        self.calls = 0
        self.incomplete_index = incomplete_index
        self.approximate = approximate

    def __call__(self, _action_space, state, _cache):
        index = self.calls
        self.calls += 1
        if index == self.incomplete_index:
            return ExactCountResult(
                status="incomplete_wall_time",
                support_size=99 if self.approximate else None,
                expanded_states=10,
                legal_edges=20,
                cache_hits=0,
                elapsed_time_s=1.0,
                root_state_id=state.measurement_state_id,
                error="guard",
            )
        return ExactCountResult(
            status="complete",
            support_size=30 + int(state.variant),
            expanded_states=10,
            legal_edges=20,
            cache_hits=0,
            elapsed_time_s=0.01,
            root_state_id=state.measurement_state_id,
        )


class FakeProbe:
    def __init__(self, nondeterministic=False):
        self.calls = 0
        self.nondeterministic = nondeterministic

    def __call__(self, policy, state):
        self.calls += 1
        first = ((5, 1 + (int(policy.version[1:]) + state.actor) % 2), (8,))
        second = ((8,),) if self.nondeterministic and self.calls == 1 else first
        return PolicyProbeResult(
            version=policy.version,
            status="complete",
            deterministic=first == second,
            raw_actions=(first, second),
            elapsed_time_s=0.01,
            stdout="{}\n{}\n",
            stderr="",
            error=None,
        )


def make_pack() -> InterventionStatePack:
    states = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for actor in (0, 1):
            index = scenario_index * 2 + actor
            states.append(
                InterventionState(
                    state_key=f"{scenario}-{actor}",
                    scenario=scenario,
                    actor=actor,
                    variant=actor,
                    snapshot={
                        "schema": "generals-measurement-state-v1",
                        "actor": actor,
                        "state": {"index": index, "round": 80},
                    },
                    measurement_state_id=f"intervention-{index:02d}",
                    construction_receipt={"index": index},
                    assertion_receipt={"scenario_predicate": True},
                )
            )
    return InterventionStatePack(
        measurement_id="generals-policy-kl-expanded-v1",
        schema=PACK_SCHEMA,
        states=tuple(states),
    )


def make_target(tmp_path: Path):
    target = tmp_path / "target-v9"
    source = target / "versions/v9/source"
    source.mkdir(parents=True)
    (source / "main.py").write_text("VERSION = 9\n")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(manifest, target / "versions/v9/manifest.json")
    write_json(
        target / "summary.json",
        {
            "run_id": target.name,
            "status": "complete",
            "runnable": True,
        },
    )
    return target, manifest


def make_verified(tmp_path: Path) -> VerifiedPolicyKLSource:
    records = tuple(
        {
            "seed": 100 + index,
            "seat": index % 2,
            "decision_number": 2 if index < 6 else 10,
            "state_id": f"legacy-{index:02d}",
            "snapshot": {
                "schema": "generals-measurement-state-v1",
                "actor": index % 2,
                "state": {"index": index, "round": 20},
            },
        }
        for index in range(12)
    )
    counts = {record["state_id"]: 10 + index for index, record in enumerate(records)}
    actions = {
        (f"v{version}", record["state_id"]): (
            (5, 1 + (version + index) % 2),
            (8,),
        )
        for version in range(9)
        for index, record in enumerate(records)
    }
    computation = compute_controlled_policy_kl(
        versions=tuple(f"v{index}" for index in range(9)),
        reference_state_ids=tuple(record["state_id"] for record in records),
        counts=counts,
        actions=actions,
        missing_actions={},
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        action_space_spec_id=EXPECTED_ACTION_SPACE_SPEC_ID,
    )
    return VerifiedPolicyKLSource(
        root=tmp_path / "legacy",
        tree_hash="a" * 64,
        reference_records=records,
        counts=counts,
        actions=actions,
        prior_facts=computation.facts,
        reuse_events=(),
        kl_source_events={},
        prior_metric=computation.metric,
    )


def make_policies(tmp_path: Path, target_manifest):
    policies = []
    snapshotter = LocalWorkspaceSnapshotter()
    for index in range(10):
        source = tmp_path / "fake-policies" / f"v{index}"
        source.mkdir(parents=True)
        (source / "main.py").write_text(f"VERSION = {index}\n")
        manifest = snapshotter.capture(source)
        if index == 9:
            manifest = target_manifest
        policies.append(
            HistoricalPolicySource(
                version=f"v{index}",
                run_id=f"run-{index}",
                content_hash=manifest.content_hash,
                source=source,
                manifest=manifest,
            )
        )
    return tuple(policies)


def make_expanded_pipeline(
    tmp_path: Path,
    *,
    counter=None,
    probe=None,
    pack=None,
    expected_pack_hash=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pack = pack or make_pack()
    pack_path = tmp_path / "states.json"
    pack_path.write_bytes(pack.canonical_bytes())
    target, target_manifest = make_target(tmp_path)
    verified = make_verified(tmp_path)
    policies = list(make_policies(tmp_path, target_manifest))
    policies[-1] = HistoricalPolicySource(
        version="v9",
        run_id=target.name,
        content_hash=target_manifest.content_hash,
        source=target / "versions/v9/source",
        manifest=target_manifest,
    )
    config = ExpandedPolicyKLConfig(
        measurement_id=pack.measurement_id,
        source_measurement_id="generals-policy-kl-reference-v3",
        source_run_id="legacy",
        source_tree_hash="a" * 64,
        state_pack=Path("states.json"),
        epsilons=EPSILONS,
        primary_epsilon="0.01",
        intervention_states=tuple(
            InterventionStateSpec(
                state_key=item.state_key,
                scenario=item.scenario,
                actor=item.actor,
                variant=item.variant,
            )
            for item in pack.states
        ),
    )
    counter = counter or FakeCounter()
    probe = probe or FakeProbe()
    pipeline = GeneralsExpandedPolicyKLPipeline(
        reference=config,
        legacy_run_dir=verified.root,
        target_run_dir=target,
        expected_target_hash=target_manifest.content_hash,
        state_pack_path=pack_path,
        expected_state_pack_hash=(
            expected_pack_hash
            or hashlib.sha256(pack.canonical_bytes()).hexdigest()
        ),
        data_dir=tmp_path / "data",
        engine_root=tmp_path / "engine",
        engine_hash="e" * 64,
        sdk_root=tmp_path / "sdk",
        legacy_verifier=lambda *_args, **_kwargs: verified,
        materializer=lambda *_args, **_kwargs: (),
        pack_loader=lambda _path: pack,
        policy_resolver=lambda: tuple(policies),
        action_space_factory=lambda _run_dir: FakeActionSpace(),
        count_state=counter,
        probe_policy=probe,
    )
    return pipeline, counter, probe


def event_counts(path: Path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result = {}
    for record in records:
        key = (record["event_type"], record.get("domain_id"))
        result[key] = result.get(key, 0) + 1
    return result


def test_expanded_pipeline_reuses_legacy_and_measures_24_states(tmp_path):
    pipeline, counter, probe = make_expanded_pipeline(tmp_path)

    result = pipeline.run()

    assert result.status == "complete"
    assert result.summary["domains"][LEGACY_DOMAIN]["reference_state_count"] == 12
    expanded = result.summary["domains"][EXPANDED_DOMAIN]
    assert expanded["reference_state_count"] == 24
    assert len(expanded["transitions"]) == 9
    assert expanded["transitions"][-1]["version_after"] == "v9"
    assert counter.calls == 12
    assert probe.calls == 12 * 10 + 12
    counts = event_counts(result.run_dir / "events.jsonl")
    assert counts[("reference_state_selected", EXPANDED_DOMAIN)] == 24
    assert counts[("action_space_count", EXPANDED_DOMAIN)] == 24
    assert counts[("historical_policy_action", EXPANDED_DOMAIN)] == 240
    assert counts[("controlled_reference_policy_kl", EXPANDED_DOMAIN)] == 864
    assert counts[("controlled_reference_policy_kl", LEGACY_DOMAIN)] == 48


def test_incomplete_exact_count_nulls_aggregate_and_recovers_in_place(tmp_path):
    counter = FakeCounter(incomplete_index=3)
    pipeline, _, probe = make_expanded_pipeline(tmp_path, counter=counter)

    first = pipeline.run()

    assert first.status == "resumable_incomplete"
    assert all(
        item["mean_kl_nats"] is None
        for item in first.summary["domains"][EXPANDED_DOMAIN]["transitions"]
    )
    assert len(list((first.run_dir / "measurement/expanded-24/facts").glob("*.json"))) == 828
    counter.incomplete_index = None

    recovered = pipeline.recover(first.run_dir)

    assert recovered.run_dir == first.run_dir
    assert recovered.status == "complete"
    assert counter.calls == 13
    assert probe.calls == 132
    assert (recovered.run_dir / "measurement/expanded-24/per-state-kl.jsonl").is_file()


def test_rejects_altered_pack_hash_before_creating_a_run(tmp_path):
    pipeline, _, _ = make_expanded_pipeline(
        tmp_path,
        expected_pack_hash="0" * 64,
    )

    with pytest.raises(ExpandedPolicyKLError, match="pack hash"):
        pipeline.run()


def test_rejects_v9_target_hash_mismatch(tmp_path):
    pipeline, _, _ = make_expanded_pipeline(tmp_path)
    pipeline.expected_target_hash = "0" * 64

    with pytest.raises(ExpandedPolicyKLError, match="content hash"):
        pipeline.run()


def test_rejects_duplicate_intervention_state_id(tmp_path):
    pack = make_pack()
    states = list(pack.states)
    states[1] = replace(states[1], measurement_state_id=states[0].measurement_state_id)
    changed = replace(pack, states=tuple(states))
    pipeline, _, _ = make_expanded_pipeline(tmp_path, pack=changed)

    with pytest.raises(ExpandedPolicyKLError, match="coordinates"):
        pipeline.run()


@pytest.mark.parametrize("duplicate", ["state_key", "scenario_actor"])
def test_rejects_duplicate_intervention_coordinates(tmp_path, duplicate):
    pack = make_pack()
    states = list(pack.states)
    if duplicate == "state_key":
        states[1] = replace(states[1], state_key=states[0].state_key)
    else:
        states[1] = replace(
            states[1],
            scenario=states[0].scenario,
            actor=states[0].actor,
        )
    changed = replace(pack, states=tuple(states))
    pipeline, _, _ = make_expanded_pipeline(tmp_path, pack=changed)

    with pytest.raises(ExpandedPolicyKLError, match="coordinates"):
        pipeline.run()


def test_rejects_wrong_epsilon_contract(tmp_path):
    pipeline, _, _ = make_expanded_pipeline(tmp_path)
    pipeline.reference = replace(pipeline.reference, epsilons=("0.01",))

    with pytest.raises(ExpandedPolicyKLError, match="epsilon"):
        pipeline.run()


def test_rejects_approximate_or_nondeterministic_measurement(tmp_path):
    approximate = FakeCounter(incomplete_index=0, approximate=True)
    pipeline, _, _ = make_expanded_pipeline(tmp_path / "approx", counter=approximate)
    with pytest.raises(ExpandedPolicyKLError, match="approximate"):
        pipeline.run()

    pipeline, _, _ = make_expanded_pipeline(
        tmp_path / "probe",
        probe=FakeProbe(nondeterministic=True),
    )
    with pytest.raises(ExpandedPolicyKLError, match="nondeterministic"):
        pipeline.run()


def test_rejects_illegal_policy_action(tmp_path):
    pipeline, _, _ = make_expanded_pipeline(tmp_path)
    pipeline._action_space_factory = lambda _run_dir: IllegalActionSpace()

    with pytest.raises(ExpandedPolicyKLError, match="illegal"):
        pipeline.run()

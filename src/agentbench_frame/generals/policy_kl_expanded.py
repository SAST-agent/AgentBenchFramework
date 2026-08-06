"""Exact dual-domain policy-KL measurement for Generals v0 through v9."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import (
    LocalWorkspaceSnapshotter,
    WorkspaceManifest,
)

from .historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
    probe_historical_policy,
)
from .intervention_states import (
    INTERVENTION_STATE_PACK_SHA256,
    InterventionState,
    InterventionStatePack,
    load_intervention_state_pack,
)
from .macro_action_space import GeneralsMacroActionSpaceV1
from .macro_counter import ExactCountIncomplete, ExactCountResult, ExactMacroCounter
from .models import ExpandedPolicyKLConfig
from .policy_kl_math import CanonicalAction, compute_controlled_policy_kl
from .policy_kl_reuse import (
    EVENT_METADATA,
    EXPECTED_ACTION_SPACE_SPEC_ID,
    FROZEN_EPSILONS,
    FROZEN_PRIMARY_EPSILON,
    VerifiedPolicyKLSource,
    materialize_policy_kl_source,
    verify_policy_kl_domain,
)


VERSIONS = tuple(f"v{index}" for index in range(10))
LEGACY_VERSIONS = VERSIONS[:-1]
LEGACY_DOMAIN = "legacy-12"
EXPANDED_DOMAIN = "expanded-24"


class ExpandedPolicyKLError(ValueError):
    """The dual-domain exact measurement violated its frozen contract."""


@dataclass(frozen=True)
class ExpandedPolicyKLResult:
    run_dir: Path
    status: str
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpandedPolicyKLError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpandedPolicyKLError(f"JSON artifact must be an object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_exact(path: Path, value: Any) -> None:
    payload = value if isinstance(value, bytes) else _json_bytes(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ExpandedPolicyKLError(f"append-only artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _manifest(path: Path) -> WorkspaceManifest:
    raw = _read_json(path)
    try:
        return WorkspaceManifest(
            content_hash=str(raw["content_hash"]),
            files={str(key): str(value) for key, value in raw["files"].items()},
            changed_files=[str(item) for item in raw.get("changed_files", [])],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpandedPolicyKLError(f"invalid policy manifest {path}: {exc}") from exc


def _event_id(run_id: str, kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{kind}\0{key}".encode()).hexdigest()
    return f"evt_{digest[:32]}"


def _fact_key(fact: dict[str, Any]) -> str:
    return "\0".join(
        str(fact[name])
        for name in (
            "domain_id",
            "version_before",
            "version_after",
            "measurement_state_id",
            "epsilon",
        )
    )


class GeneralsExpandedPolicyKLPipeline:
    """Hash-reuse legacy-12 and measure the separate expanded-24 domain."""

    def __init__(
        self,
        *,
        reference: ExpandedPolicyKLConfig,
        legacy_run_dir: Path,
        target_run_dir: Path,
        expected_target_hash: str,
        state_pack_path: Path,
        data_dir: Path,
        engine_root: Path,
        engine_hash: str,
        sdk_root: Path,
        expected_state_pack_hash: str = INTERVENTION_STATE_PACK_SHA256,
        count_wall_time_s: float | None = None,
        count_max_states: int | None = None,
        legacy_verifier: Callable[..., VerifiedPolicyKLSource] = verify_policy_kl_domain,
        materializer: Callable[..., Any] = materialize_policy_kl_source,
        pack_loader: Callable[[Path], InterventionStatePack] = load_intervention_state_pack,
        policy_resolver: Callable[[], tuple[HistoricalPolicySource, ...]] | None = None,
        action_space_factory: Callable[[Path], Any] | None = None,
        count_state: Callable[[Any, InterventionState, Path], ExactCountResult] | None = None,
        probe_policy: Callable[[HistoricalPolicySource, InterventionState], PolicyProbeResult] | None = None,
    ):
        self.reference = reference
        self.legacy_run_dir = Path(legacy_run_dir).resolve()
        self.target_run_dir = Path(target_run_dir).resolve()
        self.expected_target_hash = expected_target_hash
        self.state_pack_path = Path(state_pack_path).resolve()
        self.data_dir = Path(data_dir)
        self.engine_root = Path(engine_root)
        self.engine_hash = engine_hash
        self.sdk_root = Path(sdk_root)
        self.expected_state_pack_hash = expected_state_pack_hash
        self.count_wall_time_s = count_wall_time_s
        self.count_max_states = count_max_states
        self._legacy_verifier = legacy_verifier
        self._materializer = materializer
        self._pack_loader = pack_loader
        self._policy_resolver = policy_resolver
        self._action_space_factory = action_space_factory
        self._count_state_hook = count_state
        self._probe_policy_hook = probe_policy

    def _verify_reference(self) -> None:
        if (
            self.reference.epsilons != FROZEN_EPSILONS
            or self.reference.primary_epsilon != FROZEN_PRIMARY_EPSILON
        ):
            raise ExpandedPolicyKLError("expanded epsilon or smoothing contract changed")

    def _run_config(self) -> dict[str, Any]:
        return {
            "measurement_id": self.reference.measurement_id,
            "source_measurement_id": self.reference.source_measurement_id,
            "source_run_id": self.reference.source_run_id,
            "source_tree_hash": self.reference.source_tree_hash,
            "target_run_id": self.target_run_dir.name,
            "target_content_hash": self.expected_target_hash,
            "state_pack_sha256": self.expected_state_pack_hash,
            "domains": [LEGACY_DOMAIN, EXPANDED_DOMAIN],
        }

    def _verify_legacy(self) -> VerifiedPolicyKLSource:
        return self._legacy_verifier(
            self.legacy_run_dir,
            expected_measurement_id=self.reference.source_measurement_id,
            expected_tree_hash=self.reference.source_tree_hash,
            expected_state_count=12,
            expected_versions=LEGACY_VERSIONS,
        )

    def _verify_pack(self) -> InterventionStatePack:
        try:
            payload = self.state_pack_path.read_bytes()
        except OSError as exc:
            raise ExpandedPolicyKLError(f"cannot read intervention state pack: {exc}") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.expected_state_pack_hash:
            raise ExpandedPolicyKLError("intervention state pack hash changed")
        pack = self._pack_loader(self.state_pack_path)
        if pack.canonical_bytes() != payload:
            raise ExpandedPolicyKLError("intervention state pack is not canonical")
        expected = tuple(
            (item.state_key, item.scenario, item.actor, item.variant)
            for item in self.reference.intervention_states
        )
        observed = tuple(
            (item.state_key, item.scenario, item.actor, item.variant)
            for item in pack.states
        )
        ids = tuple(item.measurement_state_id for item in pack.states)
        if (
            pack.measurement_id != self.reference.measurement_id
            or len(pack.states) != 12
            or observed != expected
            or len(set(ids)) != 12
            or len({item.state_key for item in pack.states}) != 12
            or len({(item.scenario, item.actor) for item in pack.states}) != 12
        ):
            raise ExpandedPolicyKLError("intervention state pack coordinates changed")
        return pack

    def _source_from_root(
        self,
        root: Path,
        version: str,
        run_id: str,
        expected_hash: str | None = None,
    ) -> HistoricalPolicySource:
        policy_root = root / "policies" / version
        manifest = _manifest(policy_root / "manifest.json")
        actual = LocalWorkspaceSnapshotter().capture(policy_root / "source")
        if (
            actual.content_hash != manifest.content_hash
            or actual.files != manifest.files
            or (expected_hash is not None and actual.content_hash != expected_hash)
        ):
            raise ExpandedPolicyKLError(f"{version} policy source changed")
        return HistoricalPolicySource(
            version=version,
            run_id=run_id,
            content_hash=manifest.content_hash,
            source=(policy_root / "source").resolve(),
            manifest=manifest,
        )

    def _verify_target(self) -> HistoricalPolicySource:
        summary = _read_json(self.target_run_dir / "summary.json")
        if (
            summary.get("status") != "complete"
            or summary.get("runnable") is not True
            or summary.get("run_id") != self.target_run_dir.name
        ):
            raise ExpandedPolicyKLError("target v9 run is not complete and runnable")
        root = self.target_run_dir / "versions" / "v9"
        manifest = _manifest(root / "manifest.json")
        actual = LocalWorkspaceSnapshotter().capture(root / "source")
        if (
            manifest.content_hash != self.expected_target_hash
            or actual.content_hash != self.expected_target_hash
            or actual.files != manifest.files
        ):
            raise ExpandedPolicyKLError("target v9 content hash mismatch")
        return HistoricalPolicySource(
            version="v9",
            run_id=self.target_run_dir.name,
            content_hash=self.expected_target_hash,
            source=(root / "source").resolve(),
            manifest=manifest,
        )

    def _resolve_policies(
        self,
        source: VerifiedPolicyKLSource,
        v9: HistoricalPolicySource,
    ) -> tuple[HistoricalPolicySource, ...]:
        if self._policy_resolver is not None:
            policies = self._policy_resolver()
        else:
            policies = tuple(
                self._source_from_root(
                    source.root,
                    version,
                    next(
                        str(record["run_id"])
                        for (item_version, _), record in self._action_records(source).items()
                        if item_version == version
                    ),
                )
                for version in LEGACY_VERSIONS
            ) + (v9,)
        if tuple(item.version for item in policies) != VERSIONS:
            raise ExpandedPolicyKLError("resolved policy order changed")
        if policies[-1].content_hash != self.expected_target_hash:
            raise ExpandedPolicyKLError("resolved v9 policy hash changed")
        return policies

    @staticmethod
    def _action_records(
        source: VerifiedPolicyKLSource,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        records = {}
        for version, state_id in source.actions:
            records[(version, state_id)] = _read_json(
                source.root / "policies" / version / f"{state_id}.json"
            )
        return records

    def _new_action_space(self, run_dir: Path) -> Any:
        if self._action_space_factory is not None:
            action_space = self._action_space_factory(run_dir)
        else:
            action_space = GeneralsMacroActionSpaceV1(
                self.engine_root,
                engine_hash=self.engine_hash,
                replay_path=run_dir / "action-space/official-probes.jsonl",
            )
        if action_space.spec_id != EXPECTED_ACTION_SPACE_SPEC_ID:
            raise ExpandedPolicyKLError("action-space specification changed")
        return action_space

    def _count_intervention(
        self,
        action_space: Any,
        state: InterventionState,
        cache_path: Path,
    ) -> ExactCountResult:
        if self._count_state_hook is not None:
            return self._count_state_hook(action_space, state, cache_path)
        counter = ExactMacroCounter(
            action_space,
            cache_path,
            max_wall_time_s=self.count_wall_time_s,
            max_expanded_states=self.count_max_states,
        )
        return counter.count(state.snapshot)

    def _probe_intervention(
        self,
        policy: HistoricalPolicySource,
        state: InterventionState,
    ) -> PolicyProbeResult:
        if self._probe_policy_hook is not None:
            return self._probe_policy_hook(policy, state)
        return probe_historical_policy(
            policy,
            state.snapshot,
            engine_root=self.engine_root,
            sdk_root=self.sdk_root,
            repeats=2,
            timeout_s=5.0,
        )

    @staticmethod
    def _existing_events(run_dir: Path) -> dict[str, dict[str, Any]]:
        path = run_dir / "events.jsonl"
        if not path.exists():
            return {}
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or event_id in result:
                raise ExpandedPolicyKLError("existing event stream changed")
            result[event_id] = record
        return result

    @staticmethod
    def _emit_once(
        run: Run,
        events: dict[str, dict[str, Any]],
        event_type: str,
        key: str,
        payload: dict[str, Any],
    ) -> None:
        event_id = _event_id(run.run_id, event_type, key)
        safe = dict(payload)
        safe.pop("run_id", None)
        existing = events.get(event_id)
        if existing is not None:
            actual = {name: value for name, value in existing.items() if name not in EVENT_METADATA}
            if (
                existing.get("event_type") != event_type
                or existing.get("run_id") != run.run_id
                or actual != safe
            ):
                raise ExpandedPolicyKLError(f"pre-existing event changed: {event_id}")
            return
        run.write(event_type, event_id=event_id, **safe)
        events[event_id] = {"event_type": event_type, "run_id": run.run_id, **safe}

    def _materialize_v9(self, policy: HistoricalPolicySource, run_dir: Path) -> None:
        target = run_dir / "policies/v9/source"
        snapshotter = LocalWorkspaceSnapshotter()
        if target.exists():
            actual = snapshotter.capture(target)
            if actual.content_hash != policy.content_hash or actual.files != policy.manifest.files:
                raise ExpandedPolicyKLError("materialized v9 policy changed")
        else:
            snapshotter.materialize_manifest(policy.source, target, policy.manifest)
        _write_exact(run_dir / "policies/v9/manifest.json", _json_bytes(asdict(policy.manifest)))

    def _count_states(
        self,
        run: Run,
        events: dict[str, dict[str, Any]],
        action_space: Any,
        source: VerifiedPolicyKLSource,
        pack: InterventionStatePack,
    ) -> tuple[dict[str, int], bool]:
        counts = dict(source.counts)
        for record in source.reference_records:
            state_id = str(record["state_id"])
            self._emit_once(
                run,
                events,
                "action_space_count",
                f"legacy:{state_id}",
                {
                    "domain_id": EXPANDED_DOMAIN,
                    "measurement_state_id": state_id,
                    "action_space_spec_id": action_space.spec_id,
                    "status": "complete",
                    "support_size": str(source.counts[state_id]),
                    "reuse_status": "hash_reused",
                },
            )
        complete = True
        cache = Path(run.run_dir) / "action-space/intervention-cache.sqlite"
        for state in pack.states:
            path = Path(run.run_dir) / "action-space/intervention-counts" / f"{state.measurement_state_id}.json"
            if path.exists():
                record = _read_json(path)
                if (
                    record.get("status") != "complete"
                    or record.get("action_space_spec_id") != action_space.spec_id
                    or record.get("measurement_state_id") != state.measurement_state_id
                    or not str(record.get("support_size", "")).isdecimal()
                ):
                    raise ExpandedPolicyKLError("pre-existing exact count changed")
                counts[state.measurement_state_id] = int(record["support_size"])
                result = None
            else:
                try:
                    result = self._count_intervention(action_space, state, cache)
                except ExactCountIncomplete as exc:
                    result = exc.result
                if result.support_size is not None and result.status != "complete":
                    raise ExpandedPolicyKLError("approximate support count is forbidden")
                if result.status == "complete" and result.support_size is not None:
                    record = {
                        "domain_id": EXPANDED_DOMAIN,
                        "measurement_state_id": state.measurement_state_id,
                        "state_key": state.state_key,
                        "action_space_spec_id": action_space.spec_id,
                        "status": "complete",
                        "support_size": str(result.support_size),
                        "expanded_states": result.expanded_states,
                        "legal_edges": result.legal_edges,
                        "cache_hits": result.cache_hits,
                        "elapsed_time_s": result.elapsed_time_s,
                        "error": None,
                    }
                    _write_exact(path, record)
                    counts[state.measurement_state_id] = result.support_size
                else:
                    complete = False
                    attempt = {
                        "domain_id": EXPANDED_DOMAIN,
                        "measurement_state_id": state.measurement_state_id,
                        "state_key": state.state_key,
                        "action_space_spec_id": action_space.spec_id,
                        "status": result.status,
                        "support_size": None,
                        "expanded_states": result.expanded_states,
                        "legal_edges": result.legal_edges,
                        "cache_hits": result.cache_hits,
                        "elapsed_time_s": result.elapsed_time_s,
                        "error": result.error,
                    }
                    digest = hashlib.sha256(_json_bytes(attempt)).hexdigest()
                    _write_exact(path.parent / "attempts" / f"{state.measurement_state_id}-{digest}.json", attempt)
            if state.measurement_state_id in counts:
                self._emit_once(
                    run,
                    events,
                    "action_space_count",
                    f"intervention:{state.measurement_state_id}",
                    record,
                )
        return counts, complete

    def _probe_one(
        self,
        run: Run,
        action_space: Any,
        policy: HistoricalPolicySource,
        state: InterventionState,
        group: str,
    ) -> CanonicalAction:
        path = Path(run.run_dir) / "expanded-policy-actions" / group / policy.version / f"{state.measurement_state_id}.json"
        if path.exists():
            record = _read_json(path)
            raw = record.get("canonical_action")
            if (
                record.get("status") != "complete"
                or record.get("deterministic") is not True
                or record.get("version") != policy.version
                or record.get("content_hash") != policy.content_hash
                or record.get("measurement_state_id") != state.measurement_state_id
                or record.get("action_space_spec_id") != action_space.spec_id
                or not isinstance(raw, list)
            ):
                raise ExpandedPolicyKLError("pre-existing policy action changed")
            return tuple(tuple(command) for command in raw)
        result = self._probe_intervention(policy, state)
        if result.status != "complete" or result.deterministic is not True or len(result.raw_actions) < 2:
            raise ExpandedPolicyKLError(
                f"{policy.version} policy action is nondeterministic or incomplete"
            )
        try:
            canonical = action_space.canonicalize(state.snapshot, result.raw_actions[0])
        except Exception as exc:
            raise ExpandedPolicyKLError(
                f"{policy.version} policy action is illegal: {exc}"
            ) from exc
        record = {
            "domain_id": EXPANDED_DOMAIN,
            "state_group": group,
            "version": policy.version,
            "run_id": policy.run_id,
            "content_hash": policy.content_hash,
            "measurement_state_id": state.measurement_state_id,
            "status": "complete",
            "deterministic": True,
            "raw_actions": [[list(command) for command in action] for action in result.raw_actions],
            "canonical_action": [list(command) for command in canonical],
            "elapsed_time_s": result.elapsed_time_s,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None,
            "action_space_spec_id": action_space.spec_id,
        }
        _write_exact(path, record)
        return canonical

    @staticmethod
    def _legacy_as_intervention(record: dict[str, Any]) -> InterventionState:
        return InterventionState(
            state_key=str(record["state_id"]),
            scenario="legacy_reference",
            actor=int(record["seat"]),
            variant=int(record["decision_number"]),
            snapshot=record["snapshot"],
            measurement_state_id=str(record["state_id"]),
            construction_receipt={},
            assertion_receipt={},
        )

    def _actions(
        self,
        run: Run,
        events: dict[str, dict[str, Any]],
        action_space: Any,
        source: VerifiedPolicyKLSource,
        pack: InterventionStatePack,
        policies: tuple[HistoricalPolicySource, ...],
    ) -> dict[tuple[str, str], CanonicalAction]:
        actions = dict(source.actions)
        policy_by_version = {item.version: item for item in policies}
        legacy_states = tuple(self._legacy_as_intervention(item) for item in source.reference_records)
        for state in legacy_states:
            actions[("v9", state.measurement_state_id)] = self._probe_one(
                run, action_space, policy_by_version["v9"], state, "legacy-12"
            )
        for state in pack.states:
            for policy in policies:
                actions[(policy.version, state.measurement_state_id)] = self._probe_one(
                    run, action_space, policy, state, "intervention-12"
                )
        all_states = legacy_states + pack.states
        for state in all_states:
            for policy in policies:
                action = actions[(policy.version, state.measurement_state_id)]
                self._emit_once(
                    run,
                    events,
                    "historical_policy_action",
                    f"{policy.version}:{state.measurement_state_id}",
                    {
                        "domain_id": EXPANDED_DOMAIN,
                        "reference_state_count": 24,
                        "state_group": (
                            "legacy-12" if state.scenario == "legacy_reference" else "intervention-12"
                        ),
                        "version": policy.version,
                        "policy_run_id": policy.run_id,
                        "content_hash": policy.content_hash,
                        "measurement_state_id": state.measurement_state_id,
                        "status": "complete",
                        "deterministic": True,
                        "canonical_action": [list(command) for command in action],
                        "action_space_spec_id": action_space.spec_id,
                        "reuse_status": (
                            "hash_reused"
                            if policy.version != "v9" and state.scenario == "legacy_reference"
                            else "new_probe"
                        ),
                    },
                )
        return actions

    def _reference_events(
        self,
        run: Run,
        events: dict[str, dict[str, Any]],
        source: VerifiedPolicyKLSource,
        pack: InterventionStatePack,
    ) -> None:
        for record in source.reference_records:
            state_id = str(record["state_id"])
            self._emit_once(
                run,
                events,
                "reference_state_selected",
                f"legacy:{state_id}",
                {
                    "domain_id": EXPANDED_DOMAIN,
                    "reference_state_count": 24,
                    "state_group": "legacy-12",
                    "measurement_state_id": state_id,
                    "reuse_status": "hash_reused",
                },
            )
        for state in pack.states:
            artifact = Path(run.run_dir) / "intervention-states" / f"{state.measurement_state_id}.json"
            _write_exact(artifact, asdict(state))
            self._emit_once(
                run,
                events,
                "reference_state_selected",
                f"intervention:{state.measurement_state_id}",
                {
                    "domain_id": EXPANDED_DOMAIN,
                    "reference_state_count": 24,
                    "state_group": "intervention-12",
                    "state_key": state.state_key,
                    "scenario": state.scenario,
                    "actor": state.actor,
                    "measurement_state_id": state.measurement_state_id,
                    "artifact_ref": str(artifact),
                },
            )

    def _persist_facts(
        self,
        run: Run,
        events: dict[str, dict[str, Any]],
        domain: str,
        facts: tuple[dict[str, Any], ...],
        *,
        emit_filter: Callable[[dict[str, Any]], bool],
        complete: bool,
    ) -> None:
        root = Path(run.run_dir) / "measurement" / domain
        augmented = []
        for raw in facts:
            fact = {
                **raw,
                "domain_id": domain,
                "reference_state_count": 12 if domain == LEGACY_DOMAIN else 24,
            }
            augmented.append(fact)
            if fact["status"] == "complete":
                key = _fact_key(fact)
                digest = hashlib.sha256(key.encode()).hexdigest()
                _write_exact(root / "facts" / f"{digest}.json", fact)
                if emit_filter(fact):
                    self._emit_once(
                        run,
                        events,
                        "controlled_reference_policy_kl",
                        key,
                        fact,
                    )
        if complete:
            payload = "".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
                for item in augmented
            ).encode("utf-8")
            _write_exact(root / "per-state-kl.jsonl", payload)

    def _execute(self, run: Run) -> ExpandedPolicyKLResult:
        self._verify_reference()
        run_dir = Path(run.run_dir)
        events = self._existing_events(run_dir)
        source = self._verify_legacy()
        pack = self._verify_pack()
        v9 = self._verify_target()
        self._materializer(source, run_dir)
        policies = self._resolve_policies(source, v9)
        self._materialize_v9(v9, run_dir)
        _write_exact(
            run_dir / "provenance/expanded-start.json",
            {
                "schema": "generals-expanded-policy-kl-start-v1",
                "run_id": run.run_id,
                "config": self._run_config(),
                "policy_history": [
                    {
                        "version": item.version,
                        "run_id": item.run_id,
                        "content_hash": item.content_hash,
                    }
                    for item in policies
                ],
            },
        )
        action_space = self._new_action_space(run_dir)
        self._reference_events(run, events, source, pack)
        counts, counts_complete = self._count_states(
            run, events, action_space, source, pack
        )
        actions = self._actions(run, events, action_space, source, pack, policies)
        legacy_ids = tuple(str(item["state_id"]) for item in source.reference_records)
        expanded_ids = legacy_ids + tuple(item.measurement_state_id for item in pack.states)
        legacy = compute_controlled_policy_kl(
            versions=VERSIONS,
            reference_state_ids=legacy_ids,
            counts={key: counts[key] for key in legacy_ids},
            actions=actions,
            missing_actions={},
            epsilons=self.reference.epsilons,
            primary_epsilon=self.reference.primary_epsilon,
            action_space_spec_id=action_space.spec_id,
        )
        expanded = compute_controlled_policy_kl(
            versions=VERSIONS,
            reference_state_ids=expanded_ids,
            counts=counts,
            actions=actions,
            missing_actions={},
            epsilons=self.reference.epsilons,
            primary_epsilon=self.reference.primary_epsilon,
            action_space_spec_id=action_space.spec_id,
        )
        self._persist_facts(
            run,
            events,
            LEGACY_DOMAIN,
            legacy.facts,
            emit_filter=lambda fact: fact["version_before"] == "v8",
            complete=legacy.complete,
        )
        self._persist_facts(
            run,
            events,
            EXPANDED_DOMAIN,
            expanded.facts,
            emit_filter=lambda _fact: True,
            complete=expanded.complete,
        )
        domains = {
            LEGACY_DOMAIN: {"domain_id": LEGACY_DOMAIN, **legacy.metric},
            EXPANDED_DOMAIN: {"domain_id": EXPANDED_DOMAIN, **expanded.metric},
        }
        for domain, metric in domains.items():
            _write_exact(
                run_dir / "measurement" / domain / "transition-summary.json",
                metric,
            ) if (legacy.complete if domain == LEGACY_DOMAIN else expanded.complete) else None
        status = (
            "complete"
            if counts_complete and legacy.complete and expanded.complete
            else "resumable_incomplete"
        )
        summary = run.finish(
            {
                "status": status,
                "measurement_id": self.reference.measurement_id,
                "source_measurement_id": self.reference.source_measurement_id,
                "source_run_id": self.reference.source_run_id,
                "target_run_id": self.target_run_dir.name,
                "target_content_hash": self.expected_target_hash,
                "state_pack_sha256": self.expected_state_pack_hash,
                "exact_support_required": True,
                "domains": domains,
            }
        )
        return ExpandedPolicyKLResult(run_dir, status, summary)

    def run(self) -> ExpandedPolicyKLResult:
        # Verify all immutable authority before creating a run directory.
        self._verify_reference()
        self._verify_legacy()
        self._verify_pack()
        self._verify_target()
        run = Run.start(
            game="28_generals",
            agent="generals-policy-kl-expanded",
            run_type="measurement",
            data_dir=str(self.data_dir),
            config=self._run_config(),
        )
        return self._execute(run)

    def recover(self, run_dir: Path) -> ExpandedPolicyKLResult:
        self._verify_reference()
        root = Path(run_dir).resolve()
        summary = _read_json(root / "summary.json")
        if summary.get("status") == "complete":
            raise ExpandedPolicyKLError("complete expanded measurement cannot be recovered")
        if summary.get("status") != "resumable_incomplete":
            raise ExpandedPolicyKLError("expanded measurement is not resumable")
        if summary.get("config") != self._run_config():
            raise ExpandedPolicyKLError("expanded recovery identity changed")
        run = Run.resume(root)
        return self._execute(run)

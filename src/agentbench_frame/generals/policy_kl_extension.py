"""Append-only controlled policy-KL extension from v6 to retained v7."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.quality import (
    KNOWN_EVENT_TYPES,
    inspect_event_file,
)

from .assets import (
    load_pilot_config,
    load_policy_kl_extension_config,
    require_valid_assets,
    resolve_assets,
)
from .historical_policy import (
    HistoricalPolicySource,
    resolve_historical_policies,
)
from .models import PolicyKLExtensionConfig
from .policy_kl_math import compute_controlled_policy_kl
from .policy_kl_pipeline import (
    GeneralsPolicyKLPipeline,
    PolicyKLMeasurementResult,
    ReferenceState,
)
from .policy_kl_reuse import (
    ArtifactReceipt,
    EVENT_METADATA,
    QUALITY_DEFECT_FIELDS,
    VerifiedPolicyKLSource,
    canonical_tree_hash,
    materialize_policy_kl_source,
    verify_policy_kl_source,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"append-only artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def extension_event_id(run_id: str, kind: str, key: str) -> str:
    value = hashlib.sha256(
        f"{run_id}\0{kind}\0{key}".encode("utf-8")
    ).hexdigest()
    return f"evt_{value[:32]}"


def _fact_key(fact: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(fact["version_before"]),
        str(fact["version_after"]),
        str(fact["measurement_state_id"]),
        str(fact["epsilon"]),
    )


class GeneralsPolicyKLExtensionPipeline(GeneralsPolicyKLPipeline):
    """Reuse verified v1 facts, execute only v7, and add v6→v7."""

    def __init__(
        self,
        *,
        config: Any,
        reference: PolicyKLExtensionConfig,
        assets: Any,
        data_dir: Path,
        source_run_dir: Path,
    ):
        super().__init__(
            config=config,
            reference=reference,
            assets=assets,
            data_dir=data_dir,
        )
        self.source_run_dir = Path(source_run_dir).resolve()

    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        reference_manifest_path: Path,
        source_run_dir: Path,
        data_dir: Path,
    ) -> "GeneralsPolicyKLExtensionPipeline":
        config = load_pilot_config(manifest_path)
        reference = load_policy_kl_extension_config(
            reference_manifest_path,
            config,
        )
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        return cls(
            config=config,
            reference=reference,
            assets=assets,
            data_dir=data_dir,
            source_run_dir=source_run_dir,
        )

    def _verify_source(self) -> VerifiedPolicyKLSource:
        return verify_policy_kl_source(
            self.source_run_dir,
            self.reference,
        )

    def _run_config(self) -> dict[str, Any]:
        return {
            "measurement_id": self.reference.measurement_id,
            "source_measurement_id": self.reference.source_measurement_id,
            "source_run_id": self.reference.source_run_id,
            "reuse_mode": "verified_v1_domain_probe_only_v7",
        }

    def _start_receipt(self, run: Run) -> dict[str, Any]:
        return {
            "schema": "generals-policy-kl-extension-start-v1",
            "run_id": run.run_id,
            "game": run.meta.game,
            "agent": run.meta.agent,
            "run_type": run.meta.run_type,
            "created": run.meta.created,
            "git_commit": run.meta.git_commit,
            "started_at": run.meta.started_at,
            "config": self._run_config(),
            "measurement_id": self.reference.measurement_id,
            "source_measurement_id": self.reference.source_measurement_id,
            "source_run_id": self.reference.source_run_id,
            "source_tree_hash": self.reference.source_tree_hash,
            "source_run_dir": str(self.source_run_dir),
            "policy_history": [
                asdict(item) for item in self.reference.history
            ],
        }

    def _write_start_receipt(self, run: Run) -> None:
        _write_exact(
            Path(run.run_dir) / "provenance/extension-start-receipt.json",
            _json_bytes(self._start_receipt(run)),
        )

    def _validate_recovery_identity(self, run: Run) -> None:
        if (
            Path(run.run_dir).name != run.run_id
            or run.meta.game != "28_generals"
            or run.meta.agent != "generals-policy-kl"
            or run.meta.run_type != "measurement"
            or run.config != self._run_config()
        ):
            raise ValueError("failed extension run identity changed")
        receipt = _read_json(
            Path(run.run_dir)
            / "provenance/extension-start-receipt.json"
        )
        if receipt != self._start_receipt(run):
            raise ValueError("failed extension start receipt changed")

    def _materialize_source(
        self,
        source: VerifiedPolicyKLSource,
        run_dir: Path,
    ) -> tuple[ArtifactReceipt, ...]:
        return materialize_policy_kl_source(source, run_dir)

    def _resolve_v7(self, run_dir: Path) -> HistoricalPolicySource:
        policies = resolve_historical_policies(
            (self.reference.history[-1],),
            self.data_dir,
            run_dir,
        )
        if len(policies) != 1 or policies[0].version != "v7":
            raise ValueError("extension resolver must return only v7")
        return policies[0]

    @staticmethod
    def _existing_events(run_dir: Path) -> dict[str, dict[str, Any]]:
        path = run_dir / "events.jsonl"
        if not path.exists():
            return {}
        result = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed existing event at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"existing event at line {line_number} must be an object"
                )
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(
                    f"existing event at line {line_number} is missing event_id"
                )
            if event_id in result:
                raise ValueError(f"duplicate event_id in recovery: {event_id}")
            result[event_id] = record
        return result

    @staticmethod
    def _emit_once(
        run: Run,
        emitted: dict[str, dict[str, Any]],
        event_type: str,
        *,
        kind: str,
        key: str,
        payload: dict[str, Any],
    ) -> None:
        event_id = extension_event_id(run.run_id, kind, key)
        safe_payload = dict(payload)
        safe_payload.pop("run_id", None)
        if event_id in emitted:
            existing = emitted[event_id]
            identity = {
                "event_id": event_id,
                "event_type": event_type,
                "event": event_type,
                "run_id": run.run_id,
            }
            if any(
                existing.get(name) != value
                for name, value in identity.items()
            ):
                raise ValueError(
                    f"existing event identity changed: {event_id}"
                )
            existing_payload = {
                name: value
                for name, value in existing.items()
                if name not in EVENT_METADATA
            }
            if existing_payload != safe_payload:
                raise ValueError(
                    f"existing event payload changed: {event_id}"
                )
            emitted.pop(event_id)
            return
        run.write(event_type, event_id=event_id, **safe_payload)

    def _validate_unconsumed_existing_events(
        self,
        run: Run,
        remaining: dict[str, dict[str, Any]],
    ) -> None:
        for event_id, record in remaining.items():
            event_type = record.get("event_type", record.get("event"))
            if event_type not in KNOWN_EVENT_TYPES:
                # The final event-quality gate reports unknown types.
                continue
            if event_type not in {"pipeline_error", "pipeline_resumed"}:
                raise ValueError(
                    "unexpected pre-existing scientific event: "
                    f"{event_id} ({event_type})"
                )
            if (
                record.get("event") != event_type
                or record.get("run_id") != run.run_id
                or record.get("schema_version") != "1.0"
            ):
                raise ValueError(
                    f"existing lifecycle event identity changed: {event_id}"
                )
            payload = {
                name: value
                for name, value in record.items()
                if name not in EVENT_METADATA
            }
            if event_type == "pipeline_error":
                if (
                    set(payload) != {"stage", "error"}
                    or payload.get("stage")
                    not in {
                        "policy_kl_v7_extension",
                        "policy_kl_v7_extension_recovery",
                    }
                    or not isinstance(payload.get("error"), str)
                    or not payload["error"]
                ):
                    raise ValueError(
                        f"existing lifecycle event payload changed: {event_id}"
                    )
            elif (
                set(payload) != {"recovery_status", "source_run_id"}
                or not isinstance(payload.get("recovery_status"), str)
                or payload.get("source_run_id")
                != self.reference.source_run_id
            ):
                raise ValueError(
                    f"existing lifecycle event payload changed: {event_id}"
                )

    @staticmethod
    def _validate_event_projection(
        run: Run,
        *,
        artifact_receipt_count: int,
    ) -> None:
        run.writer.flush()
        records = list(
            GeneralsPolicyKLExtensionPipeline._existing_events(
                Path(run.run_dir)
            ).values()
        )
        expected_counts = {
            "policy_kl_source_verified": 1,
            "policy_kl_artifact_reused": artifact_receipt_count,
            "reference_state_selected": 12,
            "action_space_count": 12,
            "historical_policy_action": 96,
            "controlled_reference_policy_kl": 336,
        }
        counts = Counter(record.get("event_type") for record in records)
        for event_type, expected in expected_counts.items():
            if counts[event_type] != expected:
                raise ValueError(
                    f"{event_type} event count changed: "
                    f"expected {expected}, got {counts[event_type]}"
                )
        coordinate_fields = {
            "policy_kl_artifact_reused": ("destination_relative_path",),
            "reference_state_selected": ("measurement_state_id",),
            "action_space_count": ("measurement_state_id",),
            "historical_policy_action": (
                "version",
                "measurement_state_id",
            ),
            "controlled_reference_policy_kl": (
                "version_before",
                "version_after",
                "measurement_state_id",
                "epsilon",
            ),
        }
        for event_type, fields in coordinate_fields.items():
            matching = [
                record
                for record in records
                if record.get("event_type") == event_type
            ]
            coordinates = [
                tuple(record.get(field) for field in fields)
                for record in matching
            ]
            if any(any(value is None for value in item) for item in coordinates):
                raise ValueError(f"{event_type} coordinate is incomplete")
            if len(set(coordinates)) != len(coordinates):
                raise ValueError(f"{event_type} coordinates are not unique")

    def _write_reference_spec(
        self,
        run_dir: Path,
        source: VerifiedPolicyKLSource,
    ) -> None:
        payload = {
            "measurement_id": self.reference.measurement_id,
            "source_measurement_id": self.reference.source_measurement_id,
            "source_run_id": self.reference.source_run_id,
            "source_tree_hash": source.tree_hash,
            "opponent_id": self.reference.opponent_id,
            "seeds": list(self.reference.seeds),
            "seats": list(self.reference.seats),
            "decision_numbers": list(self.reference.decision_numbers),
            "state_ids": [
                record["state_id"] for record in source.reference_records
            ],
        }
        _write_exact(
            run_dir / "benchmark/reference-state-spec.json",
            _json_bytes(payload),
        )

    def _emit_source_receipts(
        self,
        run: Run,
        run_dir: Path,
        source: VerifiedPolicyKLSource,
        receipts: tuple[ArtifactReceipt, ...],
        emitted: dict[str, dict[str, Any]],
    ) -> None:
        self._emit_once(
            run,
            emitted,
            "policy_kl_source_verified",
            kind="source",
            key=source.tree_hash,
            payload={
                "measurement_id": self.reference.measurement_id,
                "source_measurement_id": self.reference.source_measurement_id,
                "source_run_id": self.reference.source_run_id,
                "source_tree_hash": source.tree_hash,
                "source_artifact_count": len(receipts),
                "verified": True,
            },
        )
        for receipt in receipts:
            self._emit_once(
                run,
                emitted,
                "policy_kl_artifact_reused",
                kind="artifact",
                key=receipt.destination_relative_path,
                payload={
                    **asdict(receipt),
                    "source_run_id": self.reference.source_run_id,
                    "artifact_ref": str(
                        run_dir / receipt.destination_relative_path
                    ),
                    "verified": True,
                },
            )

    def _emit_imported_non_kl_events(
        self,
        run: Run,
        run_dir: Path,
        source: VerifiedPolicyKLSource,
        emitted: dict[str, dict[str, Any]],
    ) -> None:
        for record in source.reuse_events:
            event_type = str(record["event_type"])
            payload = {
                key: value
                for key, value in record.items()
                if key not in EVENT_METADATA
            }
            if event_type == "historical_policy_action":
                payload["policy_run_id"] = record["run_id"]
            if event_type == "reference_state_selected":
                state_id = record["measurement_state_id"]
                payload["artifact_ref"] = str(
                    run_dir / "reference/states" / f"{state_id}.json"
                )
            payload.update(
                {
                    "source_event_id": record["event_id"],
                    "source_run_id": self.reference.source_run_id,
                    "reuse_status": "verified_reuse",
                    "materialization_measurement_id": (
                        self.reference.measurement_id
                    ),
                }
            )
            self._emit_once(
                run,
                emitted,
                event_type,
                kind="reuse-event",
                key=str(record["event_id"]),
                payload=payload,
            )

    @staticmethod
    def _references(
        source: VerifiedPolicyKLSource,
    ) -> tuple[ReferenceState, ...]:
        return tuple(
            ReferenceState(
                seed=int(record["seed"]),
                seat=int(record["seat"]),
                decision_number=int(record["decision_number"]),
                state_id=str(record["state_id"]),
                snapshot=record["snapshot"],
            )
            for record in source.reference_records
        )

    @staticmethod
    def _validate_v7_observation(
        record: dict[str, Any],
        *,
        policy: HistoricalPolicySource,
        reference: ReferenceState,
        action_space: Any,
    ) -> tuple[tuple[int, ...], ...] | None:
        identity = {
            "version": "v7",
            "run_id": policy.run_id,
            "content_hash": policy.content_hash,
            "measurement_state_id": reference.state_id,
            "action_space_spec_id": action_space.spec_id,
        }
        if any(record.get(name) != value for name, value in identity.items()):
            raise ValueError("saved v7 policy observation changed")
        if record.get("status") != "complete":
            if record.get("canonical_action") is not None:
                raise ValueError("saved v7 policy observation changed")
            return None
        raw_actions = record.get("raw_actions")
        saved_action = record.get("canonical_action")
        if (
            record.get("deterministic") is not True
            or not isinstance(raw_actions, list)
            or len(raw_actions) != 2
            or not all(isinstance(item, list) for item in raw_actions)
            or not isinstance(saved_action, list)
            or not saved_action
        ):
            raise ValueError("saved v7 policy observation changed")
        try:
            first = action_space.canonicalize(
                reference.snapshot,
                raw_actions[0],
            )
            second = action_space.canonicalize(
                reference.snapshot,
                raw_actions[1],
            )
            saved = tuple(tuple(command) for command in saved_action)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValueError("saved v7 policy observation changed") from exc
        if not first or first != second or first != saved:
            raise ValueError("saved v7 policy observation changed")
        return saved

    def _probe_v7(
        self,
        run: Run,
        run_dir: Path,
        action_space: Any,
        references: tuple[ReferenceState, ...],
        emitted: dict[str, dict[str, Any]],
    ) -> tuple[
        dict[tuple[str, str], tuple[tuple[int, ...], ...]],
        dict[tuple[str, str], str],
    ]:
        policy = self._resolve_v7(run_dir)
        actions = {}
        missing = {}
        for reference in references:
            key = (policy.version, reference.state_id)
            artifact = (
                run_dir
                / "policies"
                / policy.version
                / f"{reference.state_id}.json"
            )
            if artifact.exists():
                record = _read_json(artifact)
            else:
                result = self._probe_policy(policy, reference)
                canonical = None
                status = result.status
                error = result.error
                if status == "complete":
                    try:
                        canonical = action_space.canonicalize(
                            reference.snapshot,
                            result.raw_actions[0],
                        )
                    except Exception as exc:
                        status = "illegal_action"
                        error = f"{type(exc).__name__}: {exc}"
                record = {
                    "version": policy.version,
                    "run_id": policy.run_id,
                    "content_hash": policy.content_hash,
                    "measurement_state_id": reference.state_id,
                    "status": status,
                    "deterministic": result.deterministic,
                    "raw_actions": [
                        [list(command) for command in action]
                        for action in result.raw_actions
                    ],
                    "canonical_action": (
                        [list(command) for command in canonical]
                        if canonical is not None
                        else None
                    ),
                    "elapsed_time_s": result.elapsed_time_s,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": error,
                    "action_space_spec_id": action_space.spec_id,
                }
                _write_exact(artifact, _json_bytes(record))
            canonical_action = self._validate_v7_observation(
                record,
                policy=policy,
                reference=reference,
                action_space=action_space,
            )
            if canonical_action is not None:
                actions[key] = canonical_action
            else:
                missing[key] = str(
                    record.get("error") or record.get("status")
                )
            event_payload = dict(record)
            event_payload["policy_run_id"] = event_payload.pop("run_id")
            event_payload.update(
                {
                    "source_event_id": None,
                    "source_run_id": None,
                    "reuse_status": "new",
                    "artifact_ref": str(artifact),
                }
            )
            self._emit_once(
                run,
                emitted,
                "historical_policy_action",
                kind="v7-action",
                key=reference.state_id,
                payload=event_payload,
            )
        return actions, missing

    def _write_measurement(
        self,
        run: Run,
        run_dir: Path,
        source: VerifiedPolicyKLSource,
        action_space: Any,
        references: tuple[ReferenceState, ...],
        v7_actions: dict[tuple[str, str], tuple[tuple[int, ...], ...]],
        missing: dict[tuple[str, str], str],
        emitted: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        actions = {**source.actions, **v7_actions}
        versions = tuple(item.version for item in self.reference.history)
        computation = compute_controlled_policy_kl(
            versions=versions,
            reference_state_ids=tuple(
                reference.state_id for reference in references
            ),
            counts=source.counts,
            actions=actions,
            missing_actions=missing,
            epsilons=self.reference.epsilons,
            primary_epsilon=self.reference.primary_epsilon,
            action_space_spec_id=action_space.spec_id,
        )
        prior_count = len(source.prior_facts)
        if computation.facts[:prior_count] != source.prior_facts:
            raise ValueError("recomputed v0-v6 KL facts differ from source")
        if (
            computation.metric["transitions"][:6]
            != source.prior_metric["transitions"]
        ):
            raise ValueError("recomputed v0-v6 aggregates differ from source")

        for fact in computation.facts:
            key = _fact_key(fact)
            source_event = source.kl_source_events.get(key)
            if source_event is None:
                payload = {
                    **fact,
                    "source_event_id": None,
                    "source_run_id": None,
                    "reuse_status": "new",
                }
                event_kind = "new-kl"
                event_key = "\0".join(key)
            else:
                payload = {
                    **fact,
                    "source_event_id": source_event["event_id"],
                    "source_run_id": self.reference.source_run_id,
                    "reuse_status": "verified_reuse",
                    "materialization_measurement_id": (
                        self.reference.measurement_id
                    ),
                }
                event_kind = "reuse-kl"
                event_key = str(source_event["event_id"])
            self._emit_once(
                run,
                emitted,
                "controlled_reference_policy_kl",
                kind=event_kind,
                key=event_key,
                payload=payload,
            )

        measurement_dir = run_dir / "measurement"
        per_state = "".join(
            json.dumps(fact, sort_keys=True, ensure_ascii=False) + "\n"
            for fact in computation.facts
        ).encode("utf-8")
        _write_exact(measurement_dir / "per-state-kl.jsonl", per_state)
        _write_exact(
            measurement_dir / "transition-summary.json",
            _json_bytes(computation.metric),
        )
        return computation.metric, computation.complete

    def _finish_extension(
        self,
        run: Run,
        *,
        status: str,
        source_hash_before: str | None,
        source_hash_after: str | None,
        metric: dict[str, Any] | None,
        source_receipt_sha256: str | None = None,
        error: str | None = None,
    ) -> PolicyKLMeasurementResult:
        run.writer.flush()
        prefinish_quality = inspect_event_file(
            Path(run.run_dir) / "events.jsonl"
        ).to_dict()
        quality_defects = {
            name: prefinish_quality.get(name, 0)
            for name in QUALITY_DEFECT_FIELDS
            if prefinish_quality.get(name, 0)
        }
        if status in {"complete", "incomplete_policy_measurement"} and (
            quality_defects or prefinish_quality.get("warnings")
        ):
            status = "failed_event_quality"
            error = (
                "event quality prevents scientific finalization: "
                f"{quality_defects or prefinish_quality.get('warnings')}"
            )
        summary = run.finish(
            {
                "status": status,
                "measurement_id": self.reference.measurement_id,
                "source_measurement_id": self.reference.source_measurement_id,
                "source_run_id": self.reference.source_run_id,
                "source_tree_hash_before": source_hash_before,
                "source_tree_hash_after": source_hash_after,
                "source_receipt_sha256": source_receipt_sha256,
                "reuse_mode": "verified_v1_domain_probe_only_v7",
                "policy_history": [
                    asdict(item) for item in self.reference.history
                ],
                "controlled_reference_policy_kl": metric,
                "error": error,
            }
        )
        run_dir = Path(run.run_dir)
        quality_path = run_dir / "quality.json"
        quality_payload = _json_bytes(summary["event_quality"])
        if quality_path.exists():
            quality_path.write_bytes(quality_payload)
        else:
            _write_exact(quality_path, quality_payload)
        return PolicyKLMeasurementResult(status, run_dir, summary)

    def _execute(
        self,
        run: Run,
    ) -> PolicyKLMeasurementResult:
        run_dir = Path(run.run_dir)
        emitted = self._existing_events(run_dir)
        source = self._verify_source()
        receipts = self._materialize_source(source, run_dir)
        self._write_reference_spec(run_dir, source)
        action_space = self._new_action_space(run_dir)
        if action_space.spec_payload() != _read_json(
            run_dir / "benchmark/action-space-spec.json"
        ):
            raise ValueError("extension action-space specification changed")
        self._emit_source_receipts(
            run,
            run_dir,
            source,
            receipts,
            emitted,
        )
        self._emit_imported_non_kl_events(
            run,
            run_dir,
            source,
            emitted,
        )
        references = self._references(source)
        v7_actions, missing = self._probe_v7(
            run,
            run_dir,
            action_space,
            references,
            emitted,
        )
        metric, complete = self._write_measurement(
            run,
            run_dir,
            source,
            action_space,
            references,
            v7_actions,
            missing,
            emitted,
        )
        self._validate_unconsumed_existing_events(run, emitted)
        self._validate_event_projection(
            run,
            artifact_receipt_count=len(receipts),
        )
        source_hash_after = canonical_tree_hash(self.source_run_dir)
        if source_hash_after != source.tree_hash:
            raise ValueError("source run changed during extension")
        receipt = {
            "measurement_id": self.reference.measurement_id,
            "source_measurement_id": self.reference.source_measurement_id,
            "source_run_id": self.reference.source_run_id,
            "source_tree_hash_before": source.tree_hash,
            "source_tree_hash_after": source_hash_after,
            "source_run_dir": str(self.source_run_dir),
            "artifact_receipt_count": len(receipts),
            "verified": True,
        }
        receipt_path = run_dir / "provenance/source-run-receipt.json"
        _write_exact(receipt_path, _json_bytes(receipt))
        return self._finish_extension(
            run,
            status=(
                "complete" if complete else "incomplete_policy_measurement"
            ),
            source_hash_before=source.tree_hash,
            source_hash_after=source_hash_after,
            metric=metric,
            source_receipt_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
        )

    def run(self) -> PolicyKLMeasurementResult:
        run = Run.start(
            game="28_generals",
            agent="generals-policy-kl",
            run_type="measurement",
            data_dir=str(self.data_dir),
            config=self._run_config(),
        )
        self._write_start_receipt(run)
        try:
            return self._execute(run)
        except Exception as exc:
            run.write(
                "pipeline_error",
                stage="policy_kl_v7_extension",
                error=f"{type(exc).__name__}: {exc}",
            )
            source_hash_after = None
            try:
                source_hash_after = canonical_tree_hash(self.source_run_dir)
            except Exception:
                pass
            return self._finish_extension(
                run,
                status="failed",
                source_hash_before=self.reference.source_tree_hash,
                source_hash_after=source_hash_after,
                metric=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def recover(self, failed_run: Path) -> PolicyKLMeasurementResult:
        run_dir = Path(failed_run).resolve()
        summary_path = run_dir / "summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else None
        if summary is not None:
            if summary.get("status") == "complete":
                raise ValueError("cannot recover a complete extension run")
            if summary.get("measurement_id") != self.reference.measurement_id:
                raise ValueError("failed extension measurement_id changed")
            if summary.get("source_run_id") != self.reference.source_run_id:
                raise ValueError("failed extension source run changed")
        run = Run.resume(run_dir)
        try:
            self._validate_recovery_identity(run)
        except Exception:
            run.writer.close()
            raise
        run.write(
            "pipeline_resumed",
            recovery_status=(
                summary.get("status") if summary is not None else "interrupted"
            ),
            source_run_id=self.reference.source_run_id,
        )
        try:
            return self._execute(run)
        except Exception as exc:
            run.write(
                "pipeline_error",
                stage="policy_kl_v7_extension_recovery",
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._finish_extension(
                run,
                status="failed",
                source_hash_before=self.reference.source_tree_hash,
                source_hash_after=canonical_tree_hash(self.source_run_dir),
                metric=None,
                error=f"{type(exc).__name__}: {exc}",
            )

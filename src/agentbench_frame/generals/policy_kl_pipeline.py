"""Append-only controlled-reference policy-KL measurement for Generals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
from typing import Any

from agentbench_frame.eval.information_gain import (
    uniform_smoothed_deterministic_kl,
)
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run

from .assets import (
    load_pilot_config,
    load_policy_kl_reference_config,
    prepare_opponents,
    require_valid_assets,
    resolve_assets,
)
from .historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
    probe_historical_policy,
    resolve_historical_policies,
)
from .macro_action_space import GeneralsMacroActionSpaceV1
from .macro_counter import (
    ExactCountIncomplete,
    ExactCountResult,
    ExactMacroCounter,
)
from .match import GeneralsMatchRunner
from .measurement_state import measurement_state_id
from .models import (
    AssetLayout,
    MatchCase,
    PilotConfig,
    PolicyKLReferenceConfig,
)


@dataclass(frozen=True)
class ReferenceState:
    seed: int
    seat: int
    decision_number: int
    state_id: str
    snapshot: dict[str, Any]

    @classmethod
    def from_snapshot(
        cls,
        *,
        seed: int,
        seat: int,
        decision_number: int,
        snapshot: dict[str, Any],
    ) -> "ReferenceState":
        if snapshot.get("actor") != seat:
            raise ValueError("reference snapshot actor does not match seat")
        return cls(
            seed=int(seed),
            seat=int(seat),
            decision_number=int(decision_number),
            state_id=measurement_state_id(snapshot),
            snapshot=snapshot,
        )


@dataclass(frozen=True)
class PolicyKLMeasurementResult:
    status: str
    run_dir: Path
    summary: dict[str, Any]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class GeneralsPolicyKLPipeline:
    """Collect, count, probe, and aggregate one frozen measurement domain."""

    def __init__(
        self,
        *,
        config: PilotConfig,
        reference: PolicyKLReferenceConfig,
        assets: AssetLayout,
        data_dir: Path,
        count_wall_time_s: float = 3600,
        count_max_states: int = 5_000_000,
    ):
        self.config = config
        self.reference = reference
        self.assets = assets
        self.data_dir = Path(data_dir)
        self.count_wall_time_s = float(count_wall_time_s)
        self.count_max_states = int(count_max_states)
        self._prepared = None
        self._sdk_root: Path | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        reference_manifest_path: Path,
        data_dir: Path,
        count_wall_time_s: float = 3600,
        count_max_states: int = 5_000_000,
    ) -> "GeneralsPolicyKLPipeline":
        config = load_pilot_config(manifest_path)
        reference = load_policy_kl_reference_config(
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
            count_wall_time_s=count_wall_time_s,
            count_max_states=count_max_states,
        )

    def _ensure_prepared(self, run_dir: Path):
        if self._prepared is None:
            prepared = prepare_opponents(
                self.assets,
                run_dir / "prepared-opponents",
                Path(sys.executable),
            )
            self._prepared = {
                item.agent_id: item for item in prepared
            }
            self._sdk_root = self._prepared[
                "popular-rank16-xiaoaojianghu-v1"
            ].cwd
        return self._prepared

    def _collect_reference_states(
        self,
        run: Run,
        run_dir: Path,
    ) -> tuple[ReferenceState, ...]:
        opponents = self._ensure_prepared(run_dir)
        strongest = opponents[self.reference.opponent_id]
        runner = GeneralsMatchRunner(
            self.assets,
            benchmark_id=self.reference.measurement_id,
        )
        selected = []
        wanted = set(self.reference.decision_numbers)
        for seed in self.reference.seeds:
            artifact_dir = run_dir / "reference" / "matches" / f"s{seed}"
            result = runner.run(
                MatchCase(
                    case_id=f"reference-selfplay-s{seed}",
                    seed=seed,
                    evaluated_seat=0,
                    opponent_id=self.reference.opponent_id,
                    opponent_tier="high",
                ),
                (strongest, strongest),
                artifact_dir,
                capture_measurement_states=True,
            )
            if not result.valid:
                raise ValueError(
                    f"reference self-play failed for seed {seed}: "
                    f"{result.termination_type}: {result.error}"
                )
            records = [
                json.loads(line)
                for line in (
                    artifact_dir / "measurement-state.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            for seat in self.reference.seats:
                by_decision = {
                    int(item["seat_decision_number"]): item
                    for item in records
                    if int(item["player"]) == seat
                    and int(item["seat_decision_number"]) in wanted
                }
                if set(by_decision) != wanted:
                    missing = sorted(wanted - set(by_decision))
                    raise ValueError(
                        f"reference self-play seed {seed} seat {seat} "
                        f"is missing decisions {missing}"
                    )
                for decision in self.reference.decision_numbers:
                    record = by_decision[decision]
                    reference = ReferenceState.from_snapshot(
                        seed=seed,
                        seat=seat,
                        decision_number=decision,
                        snapshot=record["snapshot"],
                    )
                    if reference.state_id != record["measurement_state_id"]:
                        raise ValueError(
                            "captured reference state hash does not match"
                        )
                    selected.append(reference)
        return tuple(selected)

    def _new_action_space(
        self,
        run_dir: Path,
    ) -> GeneralsMacroActionSpaceV1:
        return GeneralsMacroActionSpaceV1(
            self.assets.engine_root,
            engine_hash=self.assets.engine_hash,
            replay_path=run_dir / "action-space" / "official-probes.jsonl",
        )

    def _count_reference_state(
        self,
        action_space: GeneralsMacroActionSpaceV1,
        reference: ReferenceState,
        cache_path: Path,
    ) -> ExactCountResult:
        counter = ExactMacroCounter(
            action_space,
            cache_path,
            max_wall_time_s=self.count_wall_time_s,
            max_expanded_states=self.count_max_states,
        )
        return counter.count(reference.snapshot)

    def _resolve_policies(
        self,
        run_dir: Path,
    ) -> tuple[HistoricalPolicySource, ...]:
        return resolve_historical_policies(
            self.reference.history,
            self.data_dir,
            run_dir,
        )

    def _probe_policy(
        self,
        policy: HistoricalPolicySource,
        reference: ReferenceState,
    ) -> PolicyProbeResult:
        if self._sdk_root is None:
            self._ensure_prepared(Path(policy.source).parents[2])
        assert self._sdk_root is not None
        return probe_historical_policy(
            policy,
            reference.snapshot,
            engine_root=self.assets.engine_root,
            sdk_root=self._sdk_root,
            repeats=2,
            timeout_s=5.0,
        )

    def _persist_new_reference_states(
        self,
        run: Run,
        run_dir: Path,
        references: tuple[ReferenceState, ...],
    ) -> None:
        expected = (
            len(self.reference.seeds)
            * len(self.reference.seats)
            * len(self.reference.decision_numbers)
        )
        if len(references) != expected:
            raise ValueError(
                f"reference collector returned {len(references)} states, "
                f"expected {expected}"
            )
        ids = [item.state_id for item in references]
        if len(set(ids)) != len(ids):
            raise ValueError("reference collector returned duplicate state IDs")
        coordinates = {
            (item.seed, item.seat, item.decision_number)
            for item in references
        }
        expected_coordinates = {
            (seed, seat, decision)
            for seed in self.reference.seeds
            for seat in self.reference.seats
            for decision in self.reference.decision_numbers
        }
        if coordinates != expected_coordinates:
            raise ValueError("reference collector coordinates do not match spec")

        for reference in references:
            state_path = (
                run_dir
                / "reference"
                / "states"
                / f"{reference.state_id}.json"
            )
            _write_json(state_path, asdict(reference))
            run.write(
                "reference_state_selected",
                measurement_id=self.reference.measurement_id,
                seed=reference.seed,
                seat=reference.seat,
                decision_number=reference.decision_number,
                measurement_state_id=reference.state_id,
                artifact_ref=str(state_path),
            )
        _write_json(
            run_dir / "benchmark" / "reference-state-spec.json",
            {
                "measurement_id": self.reference.measurement_id,
                "opponent_id": self.reference.opponent_id,
                "seeds": list(self.reference.seeds),
                "seats": list(self.reference.seats),
                "decision_numbers": list(self.reference.decision_numbers),
                "state_ids": ids,
            },
        )

    def _load_reference_states(
        self,
        run_dir: Path,
    ) -> tuple[ReferenceState, ...]:
        spec = _read_json(
            run_dir / "benchmark" / "reference-state-spec.json"
        )
        if spec.get("measurement_id") != self.reference.measurement_id:
            raise ValueError("recovery reference measurement_id changed")
        references = []
        for state_id in spec["state_ids"]:
            raw = _read_json(
                run_dir / "reference" / "states" / f"{state_id}.json"
            )
            reference = ReferenceState(
                seed=int(raw["seed"]),
                seat=int(raw["seat"]),
                decision_number=int(raw["decision_number"]),
                state_id=str(raw["state_id"]),
                snapshot=raw["snapshot"],
            )
            if (
                reference.state_id != state_id
                or measurement_state_id(reference.snapshot) != state_id
            ):
                raise ValueError("recovery reference state hash changed")
            references.append(reference)
        return tuple(references)

    @staticmethod
    def _count_record(
        reference: ReferenceState,
        result: ExactCountResult,
        spec_id: str,
    ) -> dict[str, Any]:
        return {
            "measurement_state_id": reference.state_id,
            "seed": reference.seed,
            "seat": reference.seat,
            "decision_number": reference.decision_number,
            "action_space_spec_id": spec_id,
            "status": result.status,
            "support_size": (
                str(result.support_size)
                if result.support_size is not None
                else None
            ),
            "expanded_states": result.expanded_states,
            "legal_edges": result.legal_edges,
            "cache_hits": result.cache_hits,
            "elapsed_time_s": result.elapsed_time_s,
            "error": result.error,
        }

    def _count_references(
        self,
        run: Run,
        run_dir: Path,
        action_space: Any,
        references: tuple[ReferenceState, ...],
    ) -> tuple[dict[str, int], ExactCountIncomplete | None]:
        counts: dict[str, int] = {}
        cache = run_dir / "action-space" / "cache.sqlite"
        for reference in references:
            count_path = (
                run_dir
                / "action-space"
                / "counts"
                / f"{reference.state_id}.json"
            )
            if count_path.exists():
                existing = _read_json(count_path)
                if (
                    existing.get("status") == "complete"
                    and existing.get("action_space_spec_id")
                    == action_space.spec_id
                ):
                    counts[reference.state_id] = int(
                        existing["support_size"]
                    )
                    continue
            try:
                result = self._count_reference_state(
                    action_space,
                    reference,
                    cache,
                )
                stopped = None
            except ExactCountIncomplete as exc:
                result = exc.result
                stopped = exc
            record = self._count_record(
                reference,
                result,
                action_space.spec_id,
            )
            _write_json(count_path, record)
            run.write("action_space_count", **record)
            if result.support_size is not None:
                counts[reference.state_id] = result.support_size
            if stopped is not None:
                return counts, stopped
        return counts, None

    def _probe_references(
        self,
        run: Run,
        run_dir: Path,
        action_space: Any,
        references: tuple[ReferenceState, ...],
    ) -> tuple[
        dict[tuple[str, str], tuple[tuple[int, ...], ...]],
        dict[tuple[str, str], str],
    ]:
        policies = self._resolve_policies(run_dir)
        if tuple(item.version for item in policies) != tuple(
            item.version for item in self.reference.history
        ):
            raise ValueError("resolved historical policy order changed")
        actions = {}
        missing = {}
        for policy in policies:
            for reference in references:
                key = (policy.version, reference.state_id)
                artifact = (
                    run_dir
                    / "policies"
                    / policy.version
                    / f"{reference.state_id}.json"
                )
                if artifact.exists():
                    existing = _read_json(artifact)
                    if existing.get("status") == "complete":
                        actions[key] = tuple(
                            tuple(command)
                            for command in existing["canonical_action"]
                        )
                        continue
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
                _write_json(artifact, record)
                run.write("historical_policy_action", **record)
                if canonical is None:
                    missing[key] = error or status
                else:
                    actions[key] = canonical
        return actions, missing

    def _measure_kl(
        self,
        run: Run,
        run_dir: Path,
        action_space: Any,
        references: tuple[ReferenceState, ...],
        counts: dict[str, int],
        actions: dict[tuple[str, str], tuple[tuple[int, ...], ...]],
        missing_actions: dict[tuple[str, str], str],
    ) -> tuple[dict[str, Any], bool]:
        versions = tuple(item.version for item in self.reference.history)
        facts = []
        transition_values: dict[
            tuple[str, str, str],
            list[Decimal],
        ] = {}
        transition_disagreements: dict[
            tuple[str, str, str],
            int,
        ] = {}
        for before, after in zip(versions, versions[1:]):
            for reference in references:
                old = actions.get((before, reference.state_id))
                new = actions.get((after, reference.state_id))
                reasons = []
                if reference.state_id not in counts:
                    reasons.append("missing exact support count")
                if old is None:
                    reasons.append(
                        missing_actions.get(
                            (before, reference.state_id),
                            f"missing {before} policy action",
                        )
                    )
                if new is None:
                    reasons.append(
                        missing_actions.get(
                            (after, reference.state_id),
                            f"missing {after} policy action",
                        )
                    )
                for epsilon in self.reference.epsilons:
                    key = (before, after, epsilon)
                    if reasons:
                        decimal_value = None
                        display_value = None
                        status = "missing"
                    else:
                        value = uniform_smoothed_deterministic_kl(
                            new,
                            old,
                            counts[reference.state_id],
                            epsilon,
                            precision=80,
                        )
                        decimal_value = str(value)
                        display_value = float(value)
                        status = "complete"
                        transition_values.setdefault(key, []).append(value)
                        transition_disagreements[key] = (
                            transition_disagreements.get(key, 0)
                            + int(new != old)
                        )
                    fact = {
                        "version_before": before,
                        "version_after": after,
                        "measurement_state_id": reference.state_id,
                        "support_size": (
                            str(counts[reference.state_id])
                            if reference.state_id in counts
                            else None
                        ),
                        "epsilon": epsilon,
                        "kl_nats_decimal": decimal_value,
                        "kl_nats": display_value,
                        "actions_equal": (
                            new == old
                            if new is not None and old is not None
                            else None
                        ),
                        "status": status,
                        "missing_reasons": sorted(set(reasons)),
                        "action_space_spec_id": action_space.spec_id,
                    }
                    facts.append(fact)
                    run.write("controlled_reference_policy_kl", **fact)

        measurement_dir = run_dir / "measurement"
        measurement_dir.mkdir(parents=True, exist_ok=True)
        (measurement_dir / "per-state-kl.jsonl").write_text(
            "".join(
                json.dumps(
                    fact,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
                for fact in facts
            ),
            encoding="utf-8",
        )

        total = len(references)
        transitions = []
        all_primary_complete = True
        for before, after in zip(versions, versions[1:]):
            sensitivity = {}
            for epsilon in self.reference.epsilons:
                key = (before, after, epsilon)
                values = transition_values.get(key, [])
                coverage = {"complete": len(values), "total": total}
                if len(values) == total:
                    with localcontext() as context:
                        context.prec = 80
                        mean = +(sum(values, Decimal(0)) / Decimal(total))
                    mean_decimal = str(mean)
                    mean_display = float(mean)
                    disagreement_rate = (
                        transition_disagreements.get(key, 0) / total
                    )
                else:
                    mean_decimal = None
                    mean_display = None
                    disagreement_rate = None
                sensitivity[epsilon] = {
                    "mean_kl_nats_decimal": mean_decimal,
                    "mean_kl_nats": mean_display,
                    "coverage": coverage,
                    "action_disagreement_rate": disagreement_rate,
                }
            primary = sensitivity[self.reference.primary_epsilon]
            if primary["coverage"]["complete"] != total:
                all_primary_complete = False
            transitions.append(
                {
                    "version_before": before,
                    "version_after": after,
                    "mean_kl_nats_decimal": primary[
                        "mean_kl_nats_decimal"
                    ],
                    "mean_kl_nats": primary["mean_kl_nats"],
                    "coverage": primary["coverage"],
                    "action_disagreement_rate": primary[
                        "action_disagreement_rate"
                    ],
                    "sensitivity": sensitivity,
                }
            )
        support_values = list(counts.values())
        metric = {
            "metric": "controlled_reference_policy_kl",
            "direction": "new||old",
            "primary_epsilon": self.reference.primary_epsilon,
            "epsilons": list(self.reference.epsilons),
            "reference_state_count": total,
            "action_space_spec_id": action_space.spec_id,
            "support_size": {
                "complete": len(support_values),
                "total": total,
                "minimum": (
                    str(min(support_values)) if support_values else None
                ),
                "maximum": (
                    str(max(support_values)) if support_values else None
                ),
            },
            "transitions": transitions,
            "scientific_scope": (
                "deterministic policy disagreement weighted by exact "
                "uniform-smoothed support scale; not epistemic information gain"
            ),
        }
        _write_json(
            measurement_dir / "transition-summary.json",
            metric,
        )
        return metric, all_primary_complete

    @staticmethod
    def _finish(
        run: Run,
        *,
        status: str,
        metric: dict[str, Any] | None,
        measurement_id: str,
        error: str | None = None,
    ) -> PolicyKLMeasurementResult:
        summary = run.finish(
            {
                "status": status,
                "measurement_id": measurement_id,
                "controlled_reference_policy_kl": metric,
                "error": error,
            }
        )
        run_dir = Path(run.run_dir)
        _write_json(
            run_dir / "quality.json",
            summary["event_quality"],
        )
        return PolicyKLMeasurementResult(status, run_dir, summary)

    def _execute(
        self,
        run: Run,
        *,
        recovery: bool,
    ) -> PolicyKLMeasurementResult:
        run_dir = Path(run.run_dir)
        action_space = self._new_action_space(run_dir)
        action_spec_path = (
            run_dir / "benchmark" / "action-space-spec.json"
        )
        action_spec = action_space.spec_payload()
        if recovery:
            if _read_json(action_spec_path) != action_spec:
                raise ValueError("recovery action-space spec changed")
            references = self._load_reference_states(run_dir)
        else:
            _write_json(action_spec_path, action_spec)
            references = self._collect_reference_states(run, run_dir)
            self._persist_new_reference_states(run, run_dir, references)

        counts, stopped = self._count_references(
            run,
            run_dir,
            action_space,
            references,
        )
        if stopped is not None:
            run.write(
                "pipeline_error",
                stage="exact_count",
                status=stopped.result.status,
                error=stopped.result.error,
                measurement_state_id=stopped.result.root_state_id,
            )
            return self._finish(
                run,
                status="incomplete_exact_count",
                metric=None,
                measurement_id=self.reference.measurement_id,
                error=stopped.result.error,
            )

        actions, missing = self._probe_references(
            run,
            run_dir,
            action_space,
            references,
        )
        metric, complete = self._measure_kl(
            run,
            run_dir,
            action_space,
            references,
            counts,
            actions,
            missing,
        )
        return self._finish(
            run,
            status=(
                "complete"
                if complete
                else "incomplete_policy_measurement"
            ),
            metric=metric,
            measurement_id=self.reference.measurement_id,
        )

    def run(self) -> PolicyKLMeasurementResult:
        run = Run.start(
            game="28_generals",
            agent="generals-policy-kl",
            run_type="measurement",
            data_dir=str(self.data_dir),
            config={
                "measurement_id": self.reference.measurement_id,
                "primary_epsilon": self.reference.primary_epsilon,
                "count_wall_time_s": self.count_wall_time_s,
                "count_max_states": self.count_max_states,
            },
        )
        try:
            return self._execute(run, recovery=False)
        except Exception as exc:
            run.write(
                "pipeline_error",
                stage="pipeline",
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._finish(
                run,
                status="failed",
                metric=None,
                measurement_id=self.reference.measurement_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    def recover(self, failed_run: Path) -> PolicyKLMeasurementResult:
        run_dir = Path(failed_run).resolve()
        summary = _read_json(run_dir / "summary.json")
        if summary.get("status") == "complete":
            raise ValueError("cannot recover a complete measurement run")
        if summary.get("measurement_id") != self.reference.measurement_id:
            raise ValueError("failed run measurement_id changed")
        run = Run.resume(run_dir)
        run.write(
            "pipeline_resumed",
            recovery_status=summary.get("status"),
            action_space_spec_id=_read_json(
                run_dir / "benchmark" / "action-space-spec.json"
            )["spec_id"],
        )
        try:
            return self._execute(run, recovery=True)
        except Exception as exc:
            run.write(
                "pipeline_error",
                stage="recovery",
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._finish(
                run,
                status="failed",
                metric=None,
                measurement_id=self.reference.measurement_id,
                error=f"{type(exc).__name__}: {exc}",
            )

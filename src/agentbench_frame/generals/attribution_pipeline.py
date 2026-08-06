"""Append-only scientific attribution for the two v8 policy interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run

from .ablation_v9 import AttributionPolicy, materialize_attribution_policies
from .attribution import AttributionReport, compute_paired_attribution
from .attribution_diagnostics import (
    DiagnosticProbe,
    DiagnosticState,
    compare_same_state_actions,
    diagnostic_missing_reasons,
    earliest_trajectory_divergence,
    select_diagnostic_states,
)
from .challenge_v9 import build_round9_attribution_cases
from .dense import build_dense_trace, summarize_dense_trace
from .evaluator import GeneralsEvaluation
from .historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
    probe_historical_policy,
)
from .measurement_state import measurement_state_id
from .models import PilotConfig, Round9ChallengeConfig


POLICY_ORDER = ("A", "B", "C", "D")
ESTIMANDS = ("B-A", "C-A", "D-B-C+A")


@dataclass(frozen=True)
class AttributionRunResult:
    run_dir: Path
    status: str
    policy_order: tuple[str, ...]
    case_count_per_policy: int
    valid_case_count: int
    diagnostic_state_count: int
    report_hash: str | None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _jsonable(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_id(seed: int, seat: int) -> str:
    return f"s{seed}-p{seat}"


def _score(outcome: str) -> float:
    return 1.0 if outcome == "win" else 0.5 if outcome == "draw" else 0.0


def _metric_terminal(summary: object, field: str) -> float | None:
    metric = getattr(summary, field)
    value = getattr(metric, "terminal")
    return float(value) if value is not None else None


def _report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Generals v8 Scientific Attribution",
        "",
        "This measurement run compares four paired policies on the same 12 cases.",
        "It does not open the formal benchmark and performs no coding-agent act.",
        "",
        "## Policies",
        "",
        "| Cell | Policy | Interventions | Source hash |",
        "|---|---|---|---|",
    ]
    for policy in payload["policies"]:
        interventions = ", ".join(policy["interventions"]) or "control"
        lines.append(
            f"| {policy['cell']} | {policy['policy_id']} | {interventions} | "
            f"`{policy['content_hash']}` |"
        )
    lines.extend(["", "## Paired effects", ""])
    for metric, effect in payload["attribution"]["metrics"].items():
        lines.append(
            f"- {metric}: B-A={effect['large_stack']}, C-A={effect['contact']}, "
            f"D-B-C+A={effect['interaction']}"
        )
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"Selected states: {len(payload['diagnostic_states'])}",
        f"Same-state probes: {len(payload['diagnostic_probes'])}",
        "",
    ])
    return "\n".join(lines)


class GeneralsAttributionPipeline:
    """Run A/B/C/D without a provider or access to the formal benchmark."""

    def __init__(
        self,
        *,
        config: PilotConfig,
        challenge: Round9ChallengeConfig,
        v7: HistoricalPolicySource,
        v8: HistoricalPolicySource,
        data_dir: Path,
        evaluator: Any,
        engine_root: Path,
        sdk_root: Path,
        probe_policy: Callable[..., PolicyProbeResult] = probe_historical_policy,
        canonicalize_action: Callable[
            [Mapping[str, Any], Sequence[Sequence[int]]],
            tuple[tuple[int, ...], ...],
        ] | None = None,
        bootstrap_seed: int = 9049,
        bootstrap_replicates: int = 10_000,
    ):
        self.config = config
        self.challenge = challenge
        self.v7 = v7
        self.v8 = v8
        self.data_dir = Path(data_dir)
        self.evaluator = evaluator
        self.engine_root = Path(engine_root)
        self.sdk_root = Path(sdk_root)
        self.probe_policy = probe_policy
        self.canonicalize_action = canonicalize_action
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_replicates = bootstrap_replicates

    def _start(self) -> Run:
        return Run.start(
            game="28_generals",
            agent="generals-scientific-attribution",
            run_type="measurement",
            data_dir=str(self.data_dir),
            config={
                "challenge_id": self.challenge.challenge_id,
                "budget_phase": "learning",
                "policy_parent_version": "v7",
                "iteration_predecessor_version": "v8",
                "formal_benchmark_opened": False,
            },
        )

    @staticmethod
    def _result(
        run: Run,
        *,
        status: str,
        case_count: int,
        valid_count: int = 0,
        diagnostic_count: int = 0,
        report_hash: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> AttributionRunResult:
        summary = {
            "status": status,
            "policy_order": list(POLICY_ORDER),
            "case_count_per_policy": case_count,
            "valid_case_count": valid_count,
            "diagnostic_state_count": diagnostic_count,
            "report_hash": report_hash,
            "coding_agent_act_count": 0,
            "formal_benchmark_opened": False,
            "attribution": {
                "estimands": list(ESTIMANDS),
            },
        }
        if extra:
            summary.update(_jsonable(extra))
        run.finish(summary)
        return AttributionRunResult(
            run_dir=Path(run.run_dir),
            status=status,
            policy_order=POLICY_ORDER,
            case_count_per_policy=case_count,
            valid_case_count=valid_count,
            diagnostic_state_count=diagnostic_count,
            report_hash=report_hash,
        )

    def _measurement_states(
        self,
        run_dir: Path,
        evaluations: Mapping[str, GeneralsEvaluation],
    ) -> dict[tuple[str, str, int], Mapping[str, Any]]:
        states: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        for cell, version in (("A", "v7"), ("D", "v8")):
            for match in evaluations[cell].matches:
                path = run_dir / "matches" / cell / match.case_id / "measurement-state.jsonl"
                try:
                    records = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                    ]
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"cannot read measurement states for {cell}/{match.case_id}: {exc}"
                    ) from exc
                pair = _pair_id(match.seed, match.evaluated_seat)
                for record in records:
                    snapshot = record.get("snapshot")
                    if not isinstance(snapshot, Mapping):
                        raise ValueError("measurement-state record is missing snapshot")
                    state_id = measurement_state_id(snapshot)
                    if state_id != record.get("measurement_state_id"):
                        raise ValueError("measurement-state hash changed")
                    step = int(record["global_step"])
                    key = (version, pair, step)
                    if key in states:
                        raise ValueError("duplicate measurement-state coordinate")
                    states[key] = snapshot
        return states

    @staticmethod
    def _validate_pairs(
        cases: Sequence[Any],
        evaluations: Mapping[str, GeneralsEvaluation],
    ) -> None:
        expected = {
            case.case_id: (case.seed, case.first_player)
            for case in cases
        }
        for cell in POLICY_ORDER:
            evaluation = evaluations[cell]
            if len(evaluation.results) != len(cases) or len(evaluation.matches) != len(cases):
                raise ValueError(f"cell {cell} does not contain every attribution case")
            if {item.case_id for item in evaluation.results} != set(expected):
                raise ValueError(f"cell {cell} result pair IDs changed")
            for match in evaluation.matches:
                coordinate = expected.get(match.case_id)
                if coordinate != (match.seed, match.evaluated_seat):
                    raise ValueError(f"cell {cell} match pair IDs changed")

    @staticmethod
    def _metrics(
        evaluations: Mapping[str, GeneralsEvaluation],
    ) -> tuple[dict[str, dict[str, dict[str, float | None]]], dict[str, Any]]:
        names = (
            "outcome_score",
            "completed_rounds_survived",
            "territory_margin_terminal",
            "army_margin_terminal",
            "coin_margin_terminal",
            "net_main_pressure_terminal",
        )
        cells = {
            name: {cell: {} for cell in POLICY_ORDER}
            for name in names
        }
        dense_payload: dict[str, Any] = {}
        for cell in POLICY_ORDER:
            evaluation = evaluations[cell]
            result_by_case = {item.case_id: item for item in evaluation.results}
            dense_payload[cell] = {}
            for match in evaluation.matches:
                pair = _pair_id(match.seed, match.evaluated_seat)
                summary = summarize_dense_trace(match, build_dense_trace(match))
                dense_payload[cell][pair] = _jsonable(summary)
                cells["outcome_score"][cell][pair] = _score(
                    result_by_case[match.case_id].outcome
                )
                cells["completed_rounds_survived"][cell][pair] = float(
                    summary.completed_rounds_survived
                )
                for metric, field in (
                    ("territory_margin_terminal", "territory_margin"),
                    ("army_margin_terminal", "army_margin"),
                    ("coin_margin_terminal", "coin_margin"),
                    ("net_main_pressure_terminal", "net_main_pressure"),
                ):
                    cells[metric][cell][pair] = _metric_terminal(summary, field)
        return cells, dense_payload

    def run(self) -> AttributionRunResult:
        run = self._start()
        run_dir = Path(run.run_dir)
        cases = build_round9_attribution_cases(self.config, self.challenge)
        try:
            policies = materialize_attribution_policies(
                self.v7,
                self.v8,
                run_dir / "policies",
            )
        except ValueError as exc:
            run.write("pipeline_error", status="source_hash_drift", error=str(exc))
            return self._result(
                run,
                status="source_hash_drift",
                case_count=len(cases),
                extra={"error": str(exc)},
            )

        for policy in policies:
            policy_event = _jsonable(policy)
            policy_event["source"] = str(policy.source.relative_to(run_dir))
            run.write(
                "attribution_policy_materialized",
                **policy_event,
            )

        evaluations: dict[str, GeneralsEvaluation] = {}
        valid_count = 0
        for policy in policies:
            evaluation = self.evaluator.evaluate(
                workspace=policy.source,
                version=policy.cell,
                phase="attribution",
                run=run,
                cases=cases,
            )
            evaluations[policy.cell] = evaluation
            for result in evaluation.results:
                valid_count += int(result.valid)
                run.write(
                    "attribution_game_result",
                    policy_cell=policy.cell,
                    policy_id=policy.policy_id,
                    case_id=result.case_id,
                    outcome=result.outcome,
                    valid=result.valid,
                    error=result.error,
                    metadata=dict(result.metadata),
                )

        if valid_count != len(cases) * len(POLICY_ORDER):
            return self._result(
                run,
                status="incomplete",
                case_count=len(cases),
                valid_count=valid_count,
                extra={"error": "all 48 attribution games must be valid"},
            )
        try:
            self._validate_pairs(cases, evaluations)
            metric_cells, dense_payload = self._metrics(evaluations)
            attribution = compute_paired_attribution(
                metric_cells,
                bootstrap_seed=self.bootstrap_seed,
                bootstrap_replicates=self.bootstrap_replicates,
            )
        except ValueError as exc:
            run.write("pipeline_error", status="invalid_attribution_pairs", error=str(exc))
            return self._result(
                run,
                status="invalid_attribution_pairs",
                case_count=len(cases),
                valid_count=valid_count,
                extra={"error": str(exc)},
            )

        for metric, effect in attribution.metrics.items():
            run.write(
                "attribution_factorial_effect",
                metric=metric,
                estimands={
                    "B-A": effect.large_stack,
                    "C-A": effect.contact,
                    "D-B-C+A": effect.interaction,
                },
                bootstrap_intervals=_jsonable(effect.intervals),
                complete_pair_count=effect.complete_pair_count,
            )

        a_matches = evaluations["A"].matches
        d_matches = evaluations["D"].matches
        for left, right in zip(a_matches, d_matches):
            divergence = earliest_trajectory_divergence(left, right)
            if divergence is not None:
                run.write("trajectory_divergence", **_jsonable(divergence))
        try:
            measurement_states = self._measurement_states(run_dir, evaluations)
            diagnostic_states = select_diagnostic_states(
                a_matches,
                d_matches,
                measurement_states=measurement_states,
                max_states=self.challenge.diagnostic_max_states,
            )
            if len(diagnostic_states) > self.challenge.diagnostic_max_states:
                raise ValueError("diagnostic state count exceeds frozen maximum")
            for state in diagnostic_states:
                run.write("diagnostic_state_selected", **_jsonable(state))
            probes = compare_same_state_actions(
                diagnostic_states,
                policies,
                engine_root=self.engine_root,
                sdk_root=self.sdk_root,
                probe_policy=self.probe_policy,
                canonicalize_action=self.canonicalize_action,
            )
            for probe in probes:
                run.write("diagnostic_policy_probe", **_jsonable(probe))
            if len(probes) != len(diagnostic_states) * 4 or any(
                probe.status != "complete"
                or not probe.deterministic
                or probe.legal is not True
                for probe in probes
            ):
                raise ValueError("all diagnostic probes must be deterministic and legal")
        except ValueError as exc:
            run.write("pipeline_error", status="invalid_diagnostic_probe", error=str(exc))
            return self._result(
                run,
                status="invalid_diagnostic_probe",
                case_count=len(cases),
                valid_count=valid_count,
                extra={"error": str(exc)},
            )

        evidence = {
            "challenge_id": self.challenge.challenge_id,
            "cases": [asdict(case) for case in cases],
            "dense_episode_summaries": dense_payload,
            "diagnostic_states": diagnostic_states,
            "diagnostic_probes": probes,
            "diagnostic_missing_reasons": diagnostic_missing_reasons(
                a_matches, d_matches
            ),
        }
        report_payload = {
            "schema": "generals-scientific-attribution-v1",
            "status": "complete",
            "formal_benchmark_opened": False,
            "coding_agent_act_count": 0,
            "policies": policies,
            "attribution": attribution,
            "diagnostic_states": diagnostic_states,
            "diagnostic_probes": probes,
            "artifacts": {
                "evidence": "diagnosis/evidence.json",
                "events": "events.jsonl",
                "policy_root": "policies",
            },
        }
        report_path = run_dir / "diagnosis/report.json"
        _atomic_json(run_dir / "diagnosis/evidence.json", evidence)
        _atomic_json(report_path, report_payload)
        _atomic_text(run_dir / "diagnosis/report.md", _report_markdown(_jsonable(report_payload)))
        report_hash = _file_hash(report_path)
        run.write(
            "attribution_report_frozen",
            report_hash=report_hash,
            report_ref="diagnosis/report.json",
            evidence_ref="diagnosis/evidence.json",
        )
        run.writer.flush()
        quality = inspect_event_file(run_dir / "events.jsonl")
        defects = (
            quality.malformed_lines
            + quality.invalid_events
            + quality.unknown_event_types
            + quality.duplicate_event_ids
            + quality.missing_event_ids
            + quality.missing_run_ids
        )
        if defects:
            return self._result(
                run,
                status="invalid_event_quality",
                case_count=len(cases),
                valid_count=valid_count,
                diagnostic_count=len(diagnostic_states),
                extra={"error": "event quality defects", "quality": quality.to_dict()},
            )
        return self._result(
            run,
            status="complete",
            case_count=len(cases),
            valid_count=valid_count,
            diagnostic_count=len(diagnostic_states),
            report_hash=report_hash,
            extra={
                "attribution": {
                    "estimands": list(ESTIMANDS),
                    "bootstrap_seed": attribution.bootstrap_seed,
                    "bootstrap_replicates": attribution.bootstrap_replicates,
                    "metrics": _jsonable(attribution.metrics),
                },
                "report_artifacts": {
                    "json": "diagnosis/report.json",
                    "markdown": "diagnosis/report.md",
                    "evidence": "diagnosis/evidence.json",
                },
            },
        )

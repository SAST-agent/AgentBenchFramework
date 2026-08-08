"""Fail-closed capability gate for a Generals API leaderboard harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist
import tomllib
from typing import Any, Mapping

from .assets import AssetValidationError
from .models import LeaderboardQualificationConfig, PilotConfig


QUALIFICATION_ID = "generals-api-leaderboard-qualification-v1"
QUALIFICATION_SCHEMA = "generals-api-leaderboard-qualification-receipt-v1"
QUALIFICATION_RESULT_SCHEMA = "generals-api-leaderboard-qualification-result-v1"
QUALIFICATION_SEEDS = (
    304101,
    304202,
    304303,
    304404,
    304505,
    304606,
    304707,
    304808,
    304909,
    304999,
    305101,
    305202,
    305303,
    305404,
    305505,
    305606,
    305707,
    305808,
    305909,
    305999,
)
QUALIFICATION_SEATS = (0, 1)
QUALIFICATION_BUDGETS = (1, 2, 4, 8, 16, 32)
QUALIFICATION_BUDGET_FIELDS = (
    "coding_agent_acts",
    "total_tokens",
    "wall_time_s",
    "learning_episodes",
    "cost_usd",
)
QUALIFICATION_REPLICATES = 3
MINIMUM_QUALIFYING_REPLICATES = 2
CONFIDENCE_METHOD = "wilson-one-sided"
CONFIDENCE_LEVEL = 0.95
SUPERIORITY_THRESHOLD = 0.5
MINIMUM_WINS_PER_SEAT = 11
INITIAL_POLICY_VERSION = "v7"
INITIAL_POLICY_HASH = (
    "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
)

_RECEIPT_KEYS = {
    "schema",
    "qualification_id",
    "provider",
    "model",
    "model_revision",
    "harness_hash",
    "initial_policy_hash",
    "engine_sha256",
    "opponent_id",
    "opponent_tree_sha256",
    "replay_skill_sha256",
    "qualification_results_exposed_to_provider",
    "selection_uses_qualification_results",
    "replicates",
}
_CHECKPOINT_KEYS = {
    "budget",
    "policy_hash",
    "evaluation_status",
    "results",
}
_REPLICATE_KEYS = {
    "replicate_id",
    "run_id",
    "provider_trace_sha256",
    "checkpoints",
}
_RESULT_KEYS = {"case_id", "seed", "seat", "outcome", "valid"}
_OUTCOMES = {"win", "loss", "draw"}


@dataclass(frozen=True)
class QualificationCheckpoint:
    replicate_id: str
    coding_agent_acts: int
    status: str
    passed: bool
    policy_hash: str | None
    valid_games: int
    wins: int
    losses: int
    draws: int
    score: float | None
    win_rate: float | None
    win_rate_lower_bound: float | None
    per_seat_wins: Mapping[int, int]
    reasons: tuple[str, ...]
    budget: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LeaderboardQualificationResult:
    qualification_id: str
    status: str
    qualified: bool
    provider: str | None
    model: str | None
    model_revision: str | None
    harness_hash: str | None
    qualifying_replicates: tuple[str, ...]
    minimum_qualifying_replicates: int
    qualification_budget_acts: int | None
    checkpoints: tuple[QualificationCheckpoint, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoints"] = [asdict(item) for item in self.checkpoints]
        return payload


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise AssetValidationError(f"{field} must be an integer")
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise AssetValidationError(f"{field} must be an array of strings")
    return tuple(value)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_leaderboard_qualification_config(
    path: Path,
    pilot: PilotConfig,
    *,
    engine_sha256: str,
    opponent_tree_sha256: str,
    replay_skill_sha256: str,
) -> LeaderboardQualificationConfig:
    """Load the exact capability gate frozen for leaderboard publication."""

    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        config = LeaderboardQualificationConfig(
            qualification_id=str(raw["qualification_id"]),
            opponent_id=str(raw["opponent_id"]),
            evaluation_seeds=tuple(
                _strict_int(item, "evaluation seed")
                for item in raw["evaluation_seeds"]
            ),
            seats=tuple(
                _strict_int(item, "qualification seat")
                for item in raw["seats"]
            ),
            budget_axis=str(raw["budget_axis"]),
            budget_checkpoints=tuple(
                _strict_int(item, "budget checkpoint")
                for item in raw["budget_checkpoints"]
            ),
            replicates=_strict_int(raw["replicates"], "replicates"),
            minimum_qualifying_replicates=_strict_int(
                raw["minimum_qualifying_replicates"],
                "minimum qualifying replicates",
            ),
            confidence_method=str(raw["confidence_method"]),
            confidence_level=float(raw["confidence_level"]),
            superiority_threshold=float(raw["superiority_threshold"]),
            minimum_wins_per_seat=_strict_int(
                raw["minimum_wins_per_seat"], "minimum wins per seat"
            ),
            initial_policy_version=str(raw["initial_policy_version"]),
            initial_policy_hash=str(raw["initial_policy_hash"]),
            engine_sha256=str(raw["engine_sha256"]),
            opponent_tree_sha256=str(raw["opponent_tree_sha256"]),
            replay_skill_sha256=str(raw["replay_skill_sha256"]),
            required_budget_fields=_strings(
                raw["required_budget_fields"], "required budget fields"
            ),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise AssetValidationError(
            f"invalid leaderboard qualification manifest: {exc}"
        ) from exc

    highest = pilot.opponents[0]
    exact = (
        (config.qualification_id, QUALIFICATION_ID, "qualification_id"),
        (config.opponent_id, highest.opponent_id, "opponent_id"),
        (highest.tier, "high", "opponent tier"),
        (config.evaluation_seeds, QUALIFICATION_SEEDS, "evaluation seeds"),
        (config.seats, QUALIFICATION_SEATS, "seats"),
        (config.budget_axis, "coding_agent_acts", "budget axis"),
        (config.budget_checkpoints, QUALIFICATION_BUDGETS, "budget checkpoints"),
        (config.replicates, QUALIFICATION_REPLICATES, "replicates"),
        (
            config.minimum_qualifying_replicates,
            MINIMUM_QUALIFYING_REPLICATES,
            "minimum qualifying replicates",
        ),
        (config.confidence_method, CONFIDENCE_METHOD, "confidence method"),
        (config.confidence_level, CONFIDENCE_LEVEL, "confidence level"),
        (
            config.superiority_threshold,
            SUPERIORITY_THRESHOLD,
            "superiority threshold",
        ),
        (
            config.minimum_wins_per_seat,
            MINIMUM_WINS_PER_SEAT,
            "minimum wins per seat",
        ),
        (
            config.initial_policy_version,
            INITIAL_POLICY_VERSION,
            "initial policy version",
        ),
        (
            config.initial_policy_hash,
            INITIAL_POLICY_HASH,
            "initial policy hash",
        ),
        (config.engine_sha256, engine_sha256, "engine hash"),
        (
            config.opponent_tree_sha256,
            opponent_tree_sha256,
            "opponent tree hash",
        ),
        (
            config.replay_skill_sha256,
            replay_skill_sha256,
            "replay skill hash",
        ),
        (
            config.required_budget_fields,
            QUALIFICATION_BUDGET_FIELDS,
            "required budget fields",
        ),
    )
    for observed, expected, field in exact:
        if observed != expected:
            raise AssetValidationError(
                f"leaderboard qualification {field} changed"
            )
    if set(config.evaluation_seeds) & (
        set(pilot.evaluation_seeds) | set(pilot.learning_seeds)
    ):
        raise AssetValidationError(
            "leaderboard qualification seeds overlap the pilot"
        )
    for field, digest in (
        ("initial policy hash", config.initial_policy_hash),
        ("engine hash", config.engine_sha256),
        ("opponent tree hash", config.opponent_tree_sha256),
        ("replay skill hash", config.replay_skill_sha256),
    ):
        if not _is_digest(digest):
            raise AssetValidationError(f"{field} must be a lowercase SHA-256")
    return config


def load_qualification_receipt(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read qualification receipt: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("qualification receipt must be an object")
    return value


def wilson_lower_bound(wins: int, games: int, confidence_level: float) -> float:
    """Return a one-sided Wilson lower confidence bound for win probability."""

    if type(wins) is not int or type(games) is not int:
        raise ValueError("Wilson counts must be integers")
    if games <= 0 or wins < 0 or wins > games:
        raise ValueError("Wilson counts are outside their valid range")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence level must lie strictly between 0.5 and 1")
    z = NormalDist().inv_cdf(confidence_level)
    proportion = wins / games
    denominator = 1.0 + z * z / games
    center = proportion + z * z / (2.0 * games)
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * games))
        / games
    )
    return max(0.0, (center - spread) / denominator)


def _invalid_result(
    config: LeaderboardQualificationConfig,
    receipt: Mapping[str, Any],
    reasons: list[str],
) -> LeaderboardQualificationResult:
    return LeaderboardQualificationResult(
        qualification_id=config.qualification_id,
        status="invalid",
        qualified=False,
        provider=(
            receipt.get("provider")
            if isinstance(receipt.get("provider"), str)
            else None
        ),
        model=(
            receipt.get("model")
            if isinstance(receipt.get("model"), str)
            else None
        ),
        model_revision=(
            receipt.get("model_revision")
            if isinstance(receipt.get("model_revision"), str)
            else None
        ),
        harness_hash=(
            receipt.get("harness_hash")
            if isinstance(receipt.get("harness_hash"), str)
            else None
        ),
        qualifying_replicates=(),
        minimum_qualifying_replicates=config.minimum_qualifying_replicates,
        qualification_budget_acts=None,
        checkpoints=(),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _expected_cases(
    config: LeaderboardQualificationConfig,
) -> dict[str, tuple[int, int]]:
    return {
        f"qualify-{config.opponent_id}-s{seed}-p{seat}": (seed, seat)
        for seed in config.evaluation_seeds
        for seat in config.seats
    }


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def evaluate_leaderboard_qualification(
    config: LeaderboardQualificationConfig,
    receipt: Mapping[str, Any],
) -> LeaderboardQualificationResult:
    """Evaluate one frontier-model receipt without opening provider feedback."""

    invalid: list[str] = []
    if set(receipt) != _RECEIPT_KEYS:
        invalid.append("receipt_schema_fields_changed")
    identities = (
        (receipt.get("schema"), QUALIFICATION_SCHEMA, "schema_changed"),
        (
            receipt.get("qualification_id"),
            config.qualification_id,
            "qualification_id_changed",
        ),
        (
            receipt.get("initial_policy_hash"),
            config.initial_policy_hash,
            "initial_policy_hash_changed",
        ),
        (
            receipt.get("engine_sha256"),
            config.engine_sha256,
            "engine_hash_changed",
        ),
        (
            receipt.get("opponent_id"),
            config.opponent_id,
            "opponent_id_changed",
        ),
        (
            receipt.get("opponent_tree_sha256"),
            config.opponent_tree_sha256,
            "opponent_tree_hash_changed",
        ),
        (
            receipt.get("replay_skill_sha256"),
            config.replay_skill_sha256,
            "replay_skill_hash_changed",
        ),
        (
            receipt.get("qualification_results_exposed_to_provider"),
            False,
            "qualification_results_exposed_to_provider",
        ),
        (
            receipt.get("selection_uses_qualification_results"),
            False,
            "selection_uses_qualification_results",
        ),
    )
    for observed, expected, reason in identities:
        if observed != expected or type(observed) is not type(expected):
            invalid.append(reason)
    for field in ("provider", "model", "model_revision"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value.strip():
            invalid.append(f"{field}_missing")
    if not _is_digest(receipt.get("harness_hash")):
        invalid.append("harness_hash_invalid")
    raw_replicates = receipt.get("replicates")
    if not isinstance(raw_replicates, list):
        invalid.append("replicates_not_an_array")
    if invalid:
        return _invalid_result(config, receipt, invalid)

    assert isinstance(raw_replicates, list)
    expected_replicates = tuple(
        f"replicate-{index}" for index in range(1, config.replicates + 1)
    )
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    run_ids: set[str] = set()
    for raw in raw_replicates:
        if not isinstance(raw, Mapping) or set(raw) != _REPLICATE_KEYS:
            invalid.append("replicate_schema_invalid")
            continue
        replicate_id = raw.get("replicate_id")
        if not isinstance(replicate_id, str) or replicate_id in raw_by_id:
            invalid.append("replicate_id_invalid")
            continue
        run_id = raw.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or run_id in run_ids
        ):
            invalid.append("replicate_run_id_invalid")
        else:
            run_ids.add(run_id)
        if not _is_digest(raw.get("provider_trace_sha256")):
            invalid.append("provider_trace_hash_invalid")
        raw_by_id[replicate_id] = raw
    if tuple(sorted(raw_by_id)) != expected_replicates:
        invalid.append("replicate_set_changed")
    if invalid:
        return _invalid_result(config, receipt, invalid)

    expected_cases = _expected_cases(config)
    rows: list[QualificationCheckpoint] = []
    incomplete = False
    first_pass: dict[str, int] = {}
    for replicate_id in expected_replicates:
        raw_checkpoints = raw_by_id[replicate_id].get("checkpoints")
        if not isinstance(raw_checkpoints, list):
            invalid.append("checkpoints_not_an_array")
            continue
        by_budget: dict[int, Mapping[str, Any]] = {}
        for checkpoint in raw_checkpoints:
            if (
                not isinstance(checkpoint, Mapping)
                or set(checkpoint) != _CHECKPOINT_KEYS
            ):
                invalid.append("checkpoint_schema_invalid")
                continue
            budget = checkpoint.get("budget")
            if not isinstance(budget, Mapping) or set(budget) != set(
                config.required_budget_fields
            ):
                invalid.append("checkpoint_budget_schema_invalid")
                continue
            acts = budget.get("coding_agent_acts")
            if type(acts) is not int or acts in by_budget:
                invalid.append("checkpoint_act_invalid")
                continue
            by_budget[acts] = checkpoint
        unknown = set(by_budget) - set(config.budget_checkpoints)
        if unknown:
            invalid.append("checkpoint_budget_not_frozen")
            continue

        previous_budget: dict[str, float] | None = None
        for acts in config.budget_checkpoints:
            checkpoint = by_budget.get(acts)
            if checkpoint is None:
                incomplete = True
                rows.append(
                    QualificationCheckpoint(
                        replicate_id=replicate_id,
                        coding_agent_acts=acts,
                        status="incomplete",
                        passed=False,
                        policy_hash=None,
                        valid_games=0,
                        wins=0,
                        losses=0,
                        draws=0,
                        score=None,
                        win_rate=None,
                        win_rate_lower_bound=None,
                        per_seat_wins={seat: 0 for seat in config.seats},
                        reasons=("missing_checkpoint",),
                    )
                )
                continue
            budget = checkpoint["budget"]
            assert isinstance(budget, Mapping)
            budget_values: dict[str, float] = {}
            for field in config.required_budget_fields:
                value = budget.get(field)
                integer_field = field in {
                    "coding_agent_acts",
                    "total_tokens",
                    "learning_episodes",
                }
                if (
                    (integer_field and type(value) is not int)
                    or (not integer_field and not _number(value))
                    or float(value) < 0
                ):
                    invalid.append("checkpoint_budget_value_invalid")
                    continue
                budget_values[field] = float(value)
            if budget.get("coding_agent_acts") != acts:
                invalid.append("checkpoint_act_changed")
            if previous_budget is not None and any(
                budget_values.get(field, -1) < previous_budget.get(field, -1)
                for field in config.required_budget_fields
            ):
                invalid.append("checkpoint_budget_decreased")
            previous_budget = budget_values
            policy_hash = checkpoint.get("policy_hash")
            if not _is_digest(policy_hash):
                invalid.append("checkpoint_policy_hash_invalid")
            evaluation_status = checkpoint.get("evaluation_status")
            raw_results = checkpoint.get("results")
            if evaluation_status not in {"complete", "incomplete"}:
                invalid.append("checkpoint_evaluation_status_invalid")
                continue
            if not isinstance(raw_results, list):
                invalid.append("checkpoint_results_not_an_array")
                continue
            if evaluation_status == "incomplete":
                incomplete = True
                rows.append(
                    QualificationCheckpoint(
                        replicate_id=replicate_id,
                        coding_agent_acts=acts,
                        status="incomplete",
                        passed=False,
                        policy_hash=(
                            policy_hash
                            if isinstance(policy_hash, str)
                            else None
                        ),
                        valid_games=0,
                        wins=0,
                        losses=0,
                        draws=0,
                        score=None,
                        win_rate=None,
                        win_rate_lower_bound=None,
                        per_seat_wins={seat: 0 for seat in config.seats},
                        reasons=("evaluation_incomplete",),
                        budget=dict(budget),
                    )
                )
                continue

            per_case: dict[str, Mapping[str, Any]] = {}
            checkpoint_incomplete = False
            for result in raw_results:
                if not isinstance(result, Mapping) or set(result) != _RESULT_KEYS:
                    invalid.append("case_result_schema_invalid")
                    continue
                case_id = result.get("case_id")
                if not isinstance(case_id, str) or case_id in per_case:
                    invalid.append("case_result_id_invalid")
                    continue
                expected = expected_cases.get(case_id)
                if expected is None:
                    invalid.append("case_result_unknown")
                    continue
                seed = result.get("seed")
                seat = result.get("seat")
                if (
                    type(seed) is not int
                    or type(seat) is not int
                    or (seed, seat) != expected
                ):
                    invalid.append("case_result_identity_changed")
                if result.get("outcome") not in _OUTCOMES:
                    invalid.append("case_result_outcome_invalid")
                if type(result.get("valid")) is not bool:
                    invalid.append("case_result_validity_invalid")
                elif result.get("valid") is False:
                    checkpoint_incomplete = True
                per_case[case_id] = result
            if set(per_case) != set(expected_cases):
                checkpoint_incomplete = True
            if checkpoint_incomplete:
                incomplete = True
                rows.append(
                    QualificationCheckpoint(
                        replicate_id=replicate_id,
                        coding_agent_acts=acts,
                        status="incomplete",
                        passed=False,
                        policy_hash=(
                            policy_hash
                            if isinstance(policy_hash, str)
                            else None
                        ),
                        valid_games=sum(
                            item.get("valid") is True
                            for item in per_case.values()
                        ),
                        wins=0,
                        losses=0,
                        draws=0,
                        score=None,
                        win_rate=None,
                        win_rate_lower_bound=None,
                        per_seat_wins={seat: 0 for seat in config.seats},
                        reasons=("case_matrix_incomplete",),
                        budget=dict(budget),
                    )
                )
                continue
            wins = sum(item["outcome"] == "win" for item in per_case.values())
            losses = sum(item["outcome"] == "loss" for item in per_case.values())
            draws = sum(item["outcome"] == "draw" for item in per_case.values())
            games = len(expected_cases)
            per_seat = {
                seat: sum(
                    item["outcome"] == "win" and item["seat"] == seat
                    for item in per_case.values()
                )
                for seat in config.seats
            }
            lower = wilson_lower_bound(wins, games, config.confidence_level)
            passed = lower > config.superiority_threshold and all(
                per_seat[seat] >= config.minimum_wins_per_seat
                for seat in config.seats
            )
            if passed and replicate_id not in first_pass:
                first_pass[replicate_id] = acts
            rows.append(
                QualificationCheckpoint(
                    replicate_id=replicate_id,
                    coding_agent_acts=acts,
                    status="complete",
                    passed=passed,
                    policy_hash=(policy_hash if isinstance(policy_hash, str) else None),
                    valid_games=games,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                    score=(wins + 0.5 * draws) / games,
                    win_rate=wins / games,
                    win_rate_lower_bound=lower,
                    per_seat_wins=per_seat,
                    reasons=(() if passed else ("human_superiority_not_established",)),
                    budget=dict(budget),
                )
            )

    if invalid:
        return _invalid_result(config, receipt, invalid)
    qualifying = tuple(
        replicate_id
        for replicate_id in expected_replicates
        if replicate_id in first_pass
    )
    if incomplete:
        status = "incomplete"
        qualified = False
        reasons = ("qualification_matrix_incomplete",)
    elif len(qualifying) >= config.minimum_qualifying_replicates:
        status = "qualified"
        qualified = True
        reasons = ()
    else:
        status = "unqualified"
        qualified = False
        reasons = ("insufficient_qualifying_replicates",)
    qualification_budget = None
    if qualified:
        ordered_crossings = sorted(first_pass.values())
        qualification_budget = ordered_crossings[
            config.minimum_qualifying_replicates - 1
        ]
    return LeaderboardQualificationResult(
        qualification_id=config.qualification_id,
        status=status,
        qualified=qualified,
        provider=str(receipt["provider"]),
        model=str(receipt["model"]),
        model_revision=str(receipt["model_revision"]),
        harness_hash=str(receipt["harness_hash"]),
        qualifying_replicates=qualifying,
        minimum_qualifying_replicates=config.minimum_qualifying_replicates,
        qualification_budget_acts=qualification_budget,
        checkpoints=tuple(rows),
        reasons=reasons,
    )

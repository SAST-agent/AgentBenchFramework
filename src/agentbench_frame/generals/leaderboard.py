"""Deterministic cross-provider leaderboard aggregation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .leaderboard_qualification import (
    CONFIDENCE_LEVEL,
    LeaderboardQualificationResult,
    QualificationCheckpoint,
    QUALIFICATION_BUDGET_FIELDS,
    QUALIFICATION_BUDGETS,
    QUALIFICATION_ID,
    QUALIFICATION_REPLICATES,
    wilson_lower_bound,
)


LEADERBOARD_SCHEMA = "generals-api-leaderboard-v1"


def _require_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"qualification {field} is invalid")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"qualification {field} is invalid")
    return value


def _require_number_or_none(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise ValueError(f"qualification {field} is invalid")
    return float(value)


def _require_strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"qualification {field} is invalid")
    return tuple(value)


def _entry_identity(result: LeaderboardQualificationResult) -> tuple[str, str, str]:
    values = (result.provider, result.model, result.model_revision)
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("leaderboard provider identity is incomplete")
    return values  # type: ignore[return-value]


def _checkpoint_by_budget(
    result: LeaderboardQualificationResult,
) -> dict[int, tuple[QualificationCheckpoint, ...]]:
    by_budget: dict[int, list[QualificationCheckpoint]] = {}
    for checkpoint in result.checkpoints:
        if checkpoint.coding_agent_acts not in QUALIFICATION_BUDGETS:
            raise ValueError("leaderboard result contains an unfrozen budget")
        by_budget.setdefault(checkpoint.coding_agent_acts, []).append(checkpoint)
    if set(by_budget) != set(QUALIFICATION_BUDGETS) or any(
        len(rows) != QUALIFICATION_REPLICATES for rows in by_budget.values()
    ):
        raise ValueError(
            "leaderboard result does not contain the complete replicate matrix"
        )
    expected_replicates = tuple(
        f"replicate-{index}" for index in range(1, QUALIFICATION_REPLICATES + 1)
    )
    if any(
        tuple(sorted(checkpoint.replicate_id for checkpoint in rows))
        != expected_replicates
        for rows in by_budget.values()
    ):
        raise ValueError("leaderboard result replicate identities changed")
    return {
        acts: tuple(sorted(rows, key=lambda checkpoint: checkpoint.replicate_id))
        for acts, rows in by_budget.items()
    }


def _row(
    result: LeaderboardQualificationResult,
    checkpoints: Sequence[QualificationCheckpoint] | None,
) -> dict[str, Any]:
    provider, model, revision = _entry_identity(result)
    row: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "model_revision": revision,
        "qualification_status": result.status,
        "qualification_budget_acts": result.qualification_budget_acts,
        "qualifying_replicates": list(result.qualifying_replicates),
        "checkpoint_status": (
            "missing" if checkpoints is None else "complete"
        ),
        "replicate_statuses": (
            None
            if checkpoints is None
            else [checkpoint.status for checkpoint in checkpoints]
        ),
        "replicate_count": 0 if checkpoints is None else len(checkpoints),
        "rank": None,
        "valid_games": None,
        "wins": None,
        "losses": None,
        "draws": None,
        "score": None,
        "win_rate": None,
        "win_rate_lower_bound": None,
        "per_seat_wins": None,
        "budget": None,
        "budget_per_replicate": None,
    }
    if checkpoints is None or any(
        checkpoint.status != "complete" for checkpoint in checkpoints
    ):
        row["checkpoint_status"] = (
            "incomplete" if checkpoints is not None else "missing"
        )
        return row
    wins = sum(checkpoint.wins for checkpoint in checkpoints)
    losses = sum(checkpoint.losses for checkpoint in checkpoints)
    draws = sum(checkpoint.draws for checkpoint in checkpoints)
    games = wins + losses + draws
    per_seat_wins: dict[int, int] = {}
    for checkpoint in checkpoints:
        for seat, seat_wins in checkpoint.per_seat_wins.items():
            per_seat_wins[seat] = per_seat_wins.get(seat, 0) + seat_wins
    budget_rows = [checkpoint.budget for checkpoint in checkpoints]
    if any(budget is None for budget in budget_rows):
        row["checkpoint_status"] = "incomplete"
        return row
    assert all(budget is not None for budget in budget_rows)
    if any(
        set(budget) != set(QUALIFICATION_BUDGET_FIELDS)
        for budget in budget_rows
    ):
        row["checkpoint_status"] = "incomplete"
        return row
    budget_total = {
        field: sum(float(budget[field]) for budget in budget_rows)
        for field in QUALIFICATION_BUDGET_FIELDS
    }
    for field in ("coding_agent_acts", "total_tokens", "learning_episodes"):
        budget_total[field] = int(budget_total[field])
    row.update(
        {
            "valid_games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": (wins + 0.5 * draws) / games,
            "win_rate": wins / games,
            "win_rate_lower_bound": wilson_lower_bound(
                wins, games, CONFIDENCE_LEVEL
            ),
            "per_seat_wins": {
                str(seat): wins
                for seat, wins in sorted(per_seat_wins.items())
            },
            "budget": budget_total,
            "budget_per_replicate": [dict(budget) for budget in budget_rows],
        }
    )
    return row


def build_leaderboard(
    results: Sequence[LeaderboardQualificationResult],
    *,
    harness_hash: str | None = None,
) -> dict[str, Any]:
    """Aggregate verified provider results at every frozen act checkpoint.

    Numeric ranking is available only for complete checkpoint rows. The
    conservative Wilson lower bound is the primary key, followed by observed
    score and wins. This keeps a high-confidence result ahead of a noisier
    result with the same raw score while leaving incomplete data unranked.
    """

    if not results:
        raise ValueError("leaderboard requires at least one provider result")
    if any(
        result.status != "qualified" or not result.qualified
        for result in results
    ):
        raise ValueError("leaderboard publication requires qualified results")
    identities = [_entry_identity(result) for result in results]
    if len(identities) != len(set(identities)):
        raise ValueError("leaderboard contains duplicate provider identities")
    if any(result.qualification_id != QUALIFICATION_ID for result in results):
        raise ValueError("leaderboard qualification contract changed")
    observed_hashes = {result.harness_hash for result in results}
    if None in observed_hashes or len(observed_hashes) != 1:
        raise ValueError("leaderboard results were not produced by one harness")
    observed_hash = next(iter(observed_hashes))
    if harness_hash is not None and observed_hash != harness_hash:
        raise ValueError("leaderboard harness hash changed")

    by_result = {
        identity: _checkpoint_by_budget(result)
        for identity, result in zip(identities, results)
    }
    checkpoints: list[dict[str, Any]] = []
    for acts in QUALIFICATION_BUDGETS:
        rows = [
            _row(result, by_result[identity].get(acts))
            for identity, result in zip(identities, results)
        ]
        ranked = [
            row
            for row in rows
            if row["checkpoint_status"] == "complete"
            and row["win_rate_lower_bound"] is not None
            and row["score"] is not None
        ]
        ranked.sort(
            key=lambda row: (
                -float(row["win_rate_lower_bound"]),
                -float(row["score"]),
                -int(row["wins"]),
                row["provider"],
                row["model"],
                row["model_revision"],
            )
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        rows.sort(
            key=lambda row: (
                row["rank"] is None,
                row["rank"] if row["rank"] is not None else 10**9,
                row["provider"],
                row["model"],
            )
        )
        checkpoints.append({"coding_agent_acts": acts, "entries": rows})

    return {
        "schema": LEADERBOARD_SCHEMA,
        "qualification_id": QUALIFICATION_ID,
        "harness_hash": observed_hash,
        "provider_count": len(results),
        "budget_checkpoints": list(QUALIFICATION_BUDGETS),
        "checkpoints": checkpoints,
    }


def qualification_result_from_dict(
    payload: Mapping[str, Any],
) -> LeaderboardQualificationResult:
    """Parse the result JSON emitted by the qualification CLI."""

    required = {
        "qualification_id",
        "status",
        "qualified",
        "provider",
        "model",
        "model_revision",
        "harness_hash",
        "qualifying_replicates",
        "minimum_qualifying_replicates",
        "qualification_budget_acts",
        "checkpoints",
        "reasons",
    }
    if not required.issubset(payload):
        raise ValueError("qualification result fields are incomplete")
    if payload["status"] not in {
        "qualified",
        "unqualified",
        "incomplete",
        "invalid",
    } or type(payload["qualified"]) is not bool:
        raise ValueError("qualification result status is invalid")
    if type(payload["minimum_qualifying_replicates"]) is not int:
        raise ValueError("qualification replicate threshold is invalid")
    qualification_budget_acts = payload["qualification_budget_acts"]
    if qualification_budget_acts is not None and type(qualification_budget_acts) is not int:
        raise ValueError("qualification budget is invalid")
    for field in ("qualification_id", "provider", "model", "model_revision", "harness_hash"):
        value = payload[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"qualification {field} is invalid")
    checkpoints: list[QualificationCheckpoint] = []
    raw_checkpoints = payload["checkpoints"]
    if not isinstance(raw_checkpoints, list):
        raise ValueError("qualification checkpoints are not an array")
    for raw in raw_checkpoints:
        if not isinstance(raw, Mapping):
            raise ValueError("qualification checkpoint is not an object")
        checkpoint_fields = {
            "replicate_id",
            "coding_agent_acts",
            "status",
            "passed",
            "policy_hash",
            "valid_games",
            "wins",
            "losses",
            "draws",
            "score",
            "win_rate",
            "win_rate_lower_bound",
            "per_seat_wins",
            "reasons",
            "budget",
        }
        if set(raw) != checkpoint_fields:
            raise ValueError("qualification checkpoint fields are invalid")
        per_seat = raw.get("per_seat_wins")
        if not isinstance(per_seat, Mapping):
            raise ValueError("qualification per-seat wins are invalid")
        budget = raw.get("budget")
        if not isinstance(budget, Mapping) or set(budget) != set(
            QUALIFICATION_BUDGET_FIELDS
        ):
            raise ValueError("qualification checkpoint budget is invalid")
        if raw["status"] not in {"complete", "incomplete"}:
            raise ValueError("qualification checkpoint status is invalid")
        if type(raw["passed"]) is not bool:
            raise ValueError("qualification checkpoint pass flag is invalid")
        if raw["policy_hash"] is not None and not isinstance(
            raw["policy_hash"], str
        ):
            raise ValueError("qualification checkpoint policy hash is invalid")
        try:
            parsed_per_seat = {int(seat): _require_int(wins, "seat wins") for seat, wins in per_seat.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("qualification per-seat wins are invalid") from exc
        if set(parsed_per_seat) != {0, 1}:
            raise ValueError("qualification per-seat identities are invalid")
        parsed_budget: dict[str, object] = {}
        for field, value in budget.items():
            if field in {"coding_agent_acts", "total_tokens", "learning_episodes"}:
                parsed_budget[field] = _require_int(value, f"budget {field}")
            elif type(value) in {int, float}:
                parsed_budget[field] = value
            else:
                raise ValueError(f"qualification budget {field} is invalid")
        checkpoints.append(
            QualificationCheckpoint(
                replicate_id=(
                    _require_string(raw["replicate_id"], "replicate_id")
                ),
                coding_agent_acts=_require_int(
                    raw["coding_agent_acts"], "coding_agent_acts"
                ),
                status=raw["status"],
                passed=raw["passed"],
                policy_hash=raw.get("policy_hash"),
                valid_games=_require_int(raw["valid_games"], "valid_games"),
                wins=_require_int(raw["wins"], "wins"),
                losses=_require_int(raw["losses"], "losses"),
                draws=_require_int(raw["draws"], "draws"),
                score=_require_number_or_none(raw.get("score"), "score"),
                win_rate=_require_number_or_none(raw.get("win_rate"), "win_rate"),
                win_rate_lower_bound=_require_number_or_none(
                    raw.get("win_rate_lower_bound"), "win rate lower bound"
                ),
                per_seat_wins=parsed_per_seat,
                reasons=_require_strings(raw.get("reasons"), "reasons"),
                budget=parsed_budget,
            )
        )
    return LeaderboardQualificationResult(
        qualification_id=_require_string(
            payload["qualification_id"], "qualification_id"
        ),
        status=payload["status"],
        qualified=payload["qualified"],
        provider=payload.get("provider"),
        model=payload.get("model"),
        model_revision=payload.get("model_revision"),
        harness_hash=payload.get("harness_hash"),
        qualifying_replicates=_require_strings(
            payload["qualifying_replicates"], "qualifying replicates"
        ),
        minimum_qualifying_replicates=payload["minimum_qualifying_replicates"],
        qualification_budget_acts=qualification_budget_acts,
        checkpoints=tuple(checkpoints),
        reasons=_require_strings(payload["reasons"], "reasons"),
    )

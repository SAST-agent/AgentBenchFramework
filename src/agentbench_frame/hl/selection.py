"""Game-grounded linear successor selection without source-size penalties."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

from agentbench_frame.hl.match_record import MatchRecord


@dataclasses.dataclass(frozen=True)
class CandidateDiagnostics:
    version_id: str
    points: float
    mean_score_margin: float
    worst_score_margin: float
    max_level: int
    survival_decisions: int
    captures: int
    behavioral_novelty: float
    branch_index: int
    opponent_points: tuple[tuple[str, float], ...] = ()
    opponent_mean_score_margins: tuple[tuple[str, float], ...] = ()
    role_points: tuple[tuple[str, float], ...] = ()
    role_mean_score_margins: tuple[tuple[str, float], ...] = ()
    opponent_role_points: tuple[tuple[str, float], ...] = ()
    opponent_role_mean_score_margins: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_matches(
        cls,
        *,
        version_id: str,
        branch_index: int,
        matches: Iterable[Mapping[str, object]],
        behavioral_novelty: float = 0.0,
    ) -> "CandidateDiagnostics":
        normalized: list[dict[str, object]] = []
        for match in matches:
            if "schema_version" in match:
                try:
                    record = MatchRecord.from_mapping(match)
                except ValueError:
                    continue
                if not record.promotable:
                    continue
                assert record.result is not None
                assert record.points is not None
                assert record.dense_margin is not None
                normalized.append(
                    {
                        "opponent": record.opponent,
                        "role": record.candidate_role,
                        "result": record.result,
                        "points": record.points,
                        "margin": record.dense_margin,
                        "max_level": int(record.terminal_metrics.get("max_level", 0)),
                        "survival_decisions": int(
                            record.terminal_metrics.get("survival_decisions", 0)
                        ),
                        "captures": int(record.terminal_metrics.get("captures", 0)),
                    }
                )
                continue
            if (
                match.get("status", "complete") != "complete"
                or match.get("result") not in {"win", "draw", "loss"}
                or not isinstance(match.get("rollman_score"), (int, float))
                or not isinstance(match.get("ghosts_score"), (int, float))
            ):
                continue
            result = str(match["result"])
            normalized.append(
                {
                    "opponent": str(match.get("opponent") or ""),
                    "role": str(match.get("role") or "candidate"),
                    "result": result,
                    "points": (
                        1.0 if result == "win" else 0.5 if result == "draw" else 0.0
                    ),
                    "margin": float(match["rollman_score"])
                    - float(match["ghosts_score"]),
                    "max_level": int(match.get("max_level") or 0),
                    "survival_decisions": int(
                        match.get("game_agent_decisions") or 0
                    ),
                    "captures": int(match.get("captures") or 0),
                }
            )
        if not normalized:
            raise ValueError("candidate diagnostics require valid completed matches")
        point_values = [float(match["points"]) for match in normalized]
        margins = [float(match["margin"]) for match in normalized]

        def grouped(
            key,
        ) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
            keys = sorted({key(match) for match in normalized})
            points: list[tuple[str, float]] = []
            dense_margins: list[tuple[str, float]] = []
            for group_key in keys:
                rows = [match for match in normalized if key(match) == group_key]
                points.append(
                    (
                        group_key,
                        sum(float(match["points"]) for match in rows) / len(rows),
                    )
                )
                dense_margins.append(
                    (
                        group_key,
                        sum(float(match["margin"]) for match in rows) / len(rows),
                    )
                )
            return tuple(points), tuple(dense_margins)

        opponent_points, opponent_margins = grouped(
            lambda match: str(match["opponent"])
        )
        role_points, role_margins = grouped(lambda match: str(match["role"]))
        opponent_role_points, opponent_role_margins = grouped(
            lambda match: f"{match['opponent']}\x1f{match['role']}"
        )
        return cls(
            version_id=version_id,
            points=sum(point_values) / len(point_values),
            mean_score_margin=sum(margins) / len(margins),
            worst_score_margin=min(margins),
            max_level=max(int(match["max_level"]) for match in normalized),
            survival_decisions=max(
                int(match["survival_decisions"]) for match in normalized
            ),
            captures=sum(int(match["captures"]) for match in normalized),
            behavioral_novelty=float(behavioral_novelty),
            branch_index=int(branch_index),
            opponent_points=opponent_points,
            opponent_mean_score_margins=opponent_margins,
            role_points=role_points,
            role_mean_score_margins=role_margins,
            opponent_role_points=opponent_role_points,
            opponent_role_mean_score_margins=opponent_role_margins,
        )

    def key(self) -> tuple[float, ...]:
        return (
            self.points,
            self.mean_score_margin,
            self.worst_score_margin,
            float(self.max_level),
            float(self.survival_decisions),
            float(-self.captures),
            self.behavioral_novelty,
            float(-self.branch_index),
        )


@dataclasses.dataclass(frozen=True)
class SelectionDecision:
    search_parent_version_id: str
    promote_official_champion: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class SearchState:
    search_parent_version_id: str
    official_champion_version_id: str
    exploration_debt: int = 0


def select_linear_successor(
    parent: CandidateDiagnostics,
    candidates: Sequence[CandidateDiagnostics],
) -> SelectionDecision:
    if not candidates:
        return SelectionDecision(parent.version_id, False, "no_valid_candidate")
    parent_points = dict(
        parent.opponent_role_points or parent.opponent_points
    )
    parent_margins = dict(
        parent.opponent_role_mean_score_margins
        or parent.opponent_mean_score_margins
    )
    if len(parent_points) > 1:
        robust: list[tuple[CandidateDiagnostics, str]] = []
        for candidate in candidates:
            points = dict(
                candidate.opponent_role_points or candidate.opponent_points
            )
            margins = dict(
                candidate.opponent_role_mean_score_margins
                or candidate.opponent_mean_score_margins
            )
            if set(points) != set(parent_points) or set(margins) != set(
                parent_margins
            ):
                continue
            if any(points[key] < parent_points[key] for key in parent_points):
                continue
            if any(points[key] > parent_points[key] for key in parent_points):
                robust.append((candidate, "robust_target_points_progress"))
                continue
            if any(margins[key] < parent_margins[key] for key in parent_margins):
                continue
            if any(margins[key] > parent_margins[key] for key in parent_margins):
                robust.append((candidate, "robust_dense_progress"))
        if robust:
            best, reason = max(robust, key=lambda item: item[0].key())
            return SelectionDecision(best.version_id, False, reason)
        return SelectionDecision(parent.version_id, False, "no_progress")
    best = max(candidates, key=CandidateDiagnostics.key)
    if best.points > parent.points:
        return SelectionDecision(best.version_id, False, "target_points_progress")
    if best.points == parent.points and best.key()[1:] > parent.key()[1:]:
        return SelectionDecision(best.version_id, False, "dense_progress")
    return SelectionDecision(parent.version_id, False, "no_progress")


def choose_debt_recovery(
    state: SearchState,
    *,
    specialists: Sequence[CandidateDiagnostics],
) -> SelectionDecision:
    if specialists:
        best = max(specialists, key=CandidateDiagnostics.key)
        return SelectionDecision(
            best.version_id,
            False,
            "historical_target_specialist",
        )
    return SelectionDecision(
        state.official_champion_version_id,
        False,
        "official_champion_fallback",
    )

"""Game-grounded linear successor selection without source-size penalties."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence


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

    @classmethod
    def from_matches(
        cls,
        *,
        version_id: str,
        branch_index: int,
        matches: Iterable[Mapping[str, object]],
        behavioral_novelty: float = 0.0,
    ) -> "CandidateDiagnostics":
        valid = [
            match
            for match in matches
            if match.get("status", "complete") == "complete"
            and match.get("result") in {"win", "draw", "loss"}
            and isinstance(match.get("rollman_score"), (int, float))
            and isinstance(match.get("ghosts_score"), (int, float))
        ]
        if not valid:
            raise ValueError("candidate diagnostics require valid completed matches")
        point_values = [
            1.0 if match["result"] == "win" else 0.5 if match["result"] == "draw" else 0.0
            for match in valid
        ]
        margins = [
            float(match["rollman_score"]) - float(match["ghosts_score"])
            for match in valid
        ]
        opponents = sorted({str(match.get("opponent") or "") for match in valid})
        opponent_points = []
        opponent_margins = []
        for opponent in opponents:
            rows = [
                match
                for match in valid
                if str(match.get("opponent") or "") == opponent
            ]
            row_points = [
                1.0
                if match["result"] == "win"
                else 0.5
                if match["result"] == "draw"
                else 0.0
                for match in rows
            ]
            row_margins = [
                float(match["rollman_score"])
                - float(match["ghosts_score"])
                for match in rows
            ]
            opponent_points.append(
                (opponent, sum(row_points) / len(row_points))
            )
            opponent_margins.append(
                (opponent, sum(row_margins) / len(row_margins))
            )
        return cls(
            version_id=version_id,
            points=sum(point_values) / len(point_values),
            mean_score_margin=sum(margins) / len(margins),
            worst_score_margin=min(margins),
            max_level=max(int(match.get("max_level") or 0) for match in valid),
            survival_decisions=max(
                int(match.get("game_agent_decisions") or 0) for match in valid
            ),
            captures=sum(int(match.get("captures") or 0) for match in valid),
            behavioral_novelty=float(behavioral_novelty),
            branch_index=int(branch_index),
            opponent_points=tuple(opponent_points),
            opponent_mean_score_margins=tuple(opponent_margins),
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
    parent_points = dict(parent.opponent_points)
    parent_margins = dict(parent.opponent_mean_score_margins)
    if len(parent_points) > 1:
        robust: list[tuple[CandidateDiagnostics, str]] = []
        for candidate in candidates:
            points = dict(candidate.opponent_points)
            margins = dict(candidate.opponent_mean_score_margins)
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

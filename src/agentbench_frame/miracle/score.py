"""Miracle 版本对齐的性能分、gain 与预算 AUC。"""

from __future__ import annotations


def aggregate_score(episodes: list[dict], candidate_camp_by_episode: dict[str, int]) -> dict:
    measured_scores = []
    wins = 0
    counts = {"normal": 0, "timeout": 0, "failed": 0, "missing": 0}
    preserved = []
    for episode in episodes:
        row = dict(episode)
        episode_id = str(row.get("episode_id", ""))
        camp = candidate_camp_by_episode.get(episode_id)
        scores = row.get("scores")
        normal = row.get("terminated_by") == "normal" and not row.get("errors")
        if camp not in (0, 1) or not isinstance(scores, (list, tuple)) or len(scores) < 2:
            counts["missing"] += 1
            row["candidate_score"] = None
        elif normal:
            value = float(scores[camp])
            measured_scores.append(value)
            wins += int(row.get("winner") == camp)
            counts["normal"] += 1
            row["candidate_score"] = value
        elif row.get("terminated_by") == "timeout":
            counts["timeout"] += 1
            row["candidate_score"] = None
        else:
            counts["failed"] += 1
            row["candidate_score"] = None
        row["candidate_camp"] = camp
        preserved.append(row)
    total = len(episodes)
    measured = len(measured_scores)
    return {
        "mean_score": round(sum(measured_scores) / measured, 6) if measured else None,
        "win_rate": round(wins / measured, 6) if measured else None,
        "completion_rate": measured / total if total else 0.0,
        "counts": counts,
        "episodes": preserved,
    }


def trapezoid_auc(points: list[dict], x_key: str, y_key: str) -> dict:
    valid = sorted(
        (
            (float(point[x_key]), float(point[y_key]))
            for point in points
            if point.get(x_key) is not None and point.get(y_key) is not None
        ),
        key=lambda item: item[0],
    )
    if len(valid) < 2:
        return {"value": None, "reason": "insufficient_points", "n_points": len(valid)}
    value = sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(valid, valid[1:])
    )
    return {"value": round(value, 6), "reason": None, "n_points": len(valid)}


def build_score_curve(iterations: list[dict]) -> dict:
    ordered = sorted(iterations, key=lambda item: int(item["iteration"]))
    raw = ordered[0].get("score") if ordered else None
    points = []
    for item in ordered:
        budget = item.get("budget", {})
        evo = item.get("score")
        point = {
            "iteration": int(item["iteration"]),
            "version": item.get("version"),
            "status": item.get("status"),
            "raw": raw,
            "evo": evo,
            "gain": round(evo - raw, 6) if evo is not None and raw is not None else None,
            "win_rate": item.get("win_rate"),
            "completion_rate": item.get("completion_rate"),
            "rollouts": budget.get("rollouts", 0),
            "total_tokens": budget.get("total_tokens", 0),
            "episode_reads": budget.get("episode_reads", 0),
            "decision_reads": budget.get("decision_reads", 0),
            "wall_seconds": budget.get("wall_seconds", 0),
        }
        points.append(point)
    axes = {
        "iteration": "iteration",
        "rollout": "rollouts",
        "total_token": "total_tokens",
        "episode_read": "episode_reads",
        "wall_time": "wall_seconds",
    }
    return {
        "metric": "official_score",
        "points": points,
        "auc": {name: trapezoid_auc(points, key, "evo") for name, key in axes.items()},
    }

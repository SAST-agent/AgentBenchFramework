"""Candidate-oriented DOTO performance, gain curves, and budget AUC."""

from __future__ import annotations

from collections.abc import Mapping


def aggregate_episodes(rows: list[dict], candidate_factions: Mapping[str, int]) -> dict:
    differences, outcomes, preserved = [], [], []
    counts = {"normal": 0, "timeout": 0, "failed": 0, "missing": 0}
    for episode in rows:
        row = dict(episode)
        episode_id = str(row.get("episode_id", ""))
        faction = candidate_factions.get(episode_id)
        scores = row.get("scores")
        normal = row.get("terminated_by") == "normal" and not row.get("errors")
        valid_scores = (isinstance(scores, (list, tuple)) and len(scores) >= 2 and
                        all(isinstance(value, (int, float)) for value in scores[:2]))
        if faction not in (0, 1) or not valid_scores:
            counts["missing"] += 1
            diff = None
        elif normal:
            candidate, opponent = float(scores[faction]), float(scores[1 - faction])
            diff = candidate - opponent
            differences.append(diff)
            outcomes.append(1.0 if diff > 0 else 0.0 if diff < 0 else 0.5)
            counts["normal"] += 1
            row["candidate_score"], row["opponent_score"] = candidate, opponent
        else:
            counts["timeout" if row.get("terminated_by") == "timeout" else "failed"] += 1
            diff = None
        row["candidate_faction"] = faction
        row["score_diff"] = diff
        preserved.append(row)
    measured, total = len(differences), len(rows)
    return {"score": round(sum(differences) / measured, 6) if measured else None,
            "win_rate": round(sum(outcomes) / measured, 6) if measured else None,
            "completion_rate": measured / total if total else 0.0,
            "counts": counts, "episodes": preserved}


def trapezoid_auc(points: list[dict], x_key: str, y_key: str) -> dict:
    valid = sorted((float(point[x_key]), float(point[y_key])) for point in points
                   if point.get(x_key) is not None and point.get(y_key) is not None)
    if len(valid) < 2:
        return {"value": None, "reason": "insufficient_points", "n_points": len(valid)}
    if len({x for x, _ in valid}) < 2:
        return {"value": None, "reason": "insufficient_distinct_x", "n_points": len(valid)}
    value = sum((right_x - left_x) * (left_y + right_y) / 2
                for (left_x, left_y), (right_x, right_y) in zip(valid, valid[1:]))
    return {"value": round(value, 6), "reason": None, "n_points": len(valid)}


def build_score_curve(iterations: list[dict]) -> dict:
    ordered = sorted(iterations, key=lambda item: int(item["iteration"]))
    raw = ordered[0].get("score") if ordered else None
    points = []
    for item in ordered:
        budget, evo = item.get("budget", {}), item.get("score")
        points.append({
            "iteration": int(item["iteration"]), "version": item.get("version"),
            "status": item.get("status"), "raw": raw, "evo": evo,
            "gain": round(evo - raw, 6) if evo is not None and raw is not None else None,
            "win_rate": item.get("win_rate"), "completion_rate": item.get("completion_rate"),
            "rollouts": budget.get("rollouts", 0), "total_tokens": budget.get("total_tokens", 0),
            "episode_reads": budget.get("episode_reads", 0), "frame_reads": budget.get("frame_reads", 0),
            "compile_seconds": budget.get("compile_seconds", 0),
            "battle_seconds": budget.get("battle_seconds", 0),
            "api_seconds": budget.get("api_seconds", 0), "wall_seconds": budget.get("wall_seconds", 0),
        })
    axes = {"iteration": "iteration", "rollout": "rollouts", "total_token": "total_tokens",
            "episode_read": "episode_reads", "frame_read": "frame_reads", "wall_time": "wall_seconds"}
    return {"metric": "candidate_score_diff", "points": points,
            "auc": {name: trapezoid_auc(points, key, "evo") for name, key in axes.items()}}

from agentbench_frame.miracle.score import aggregate_score, build_score_curve, trapezoid_auc


def test_aggregate_score_is_seat_aligned_and_preserves_failures():
    episodes = [
        {"episode_id": "a", "scores": [10, 3], "winner": 0,
         "terminated_by": "normal", "errors": []},
        {"episode_id": "b", "scores": [4, 20], "winner": 1,
         "terminated_by": "normal", "errors": []},
        {"episode_id": "c", "scores": [0, 0], "winner": 1,
         "terminated_by": "host_error", "errors": ["bad"]},
    ]
    result = aggregate_score(episodes, {"a": 0, "b": 1, "c": 0})

    assert result["mean_score"] == 15.0
    assert result["win_rate"] == 1.0
    assert result["completion_rate"] == 2 / 3
    assert result["counts"] == {"normal": 2, "timeout": 0, "failed": 1, "missing": 0}
    assert len(result["episodes"]) == 3


def test_curve_uses_fixed_raw_and_keeps_negative_gain():
    curve = build_score_curve([
        {"iteration": 0, "version": "v0", "status": "baseline", "score": 10,
         "win_rate": 0.0, "completion_rate": 1.0, "budget": {}},
        {"iteration": 1, "version": "v1", "status": "accepted", "score": 20,
         "win_rate": 1.0, "completion_rate": 1.0, "budget": {}},
        {"iteration": 2, "version": "v2", "status": "accepted", "score": 5,
         "win_rate": 0.0, "completion_rate": 1.0, "budget": {}},
    ])

    assert [point["raw"] for point in curve["points"]] == [10, 10, 10]
    assert [point["gain"] for point in curve["points"]] == [0, 10, -5]


def test_trapezoid_auc_and_insufficient_points():
    measured = trapezoid_auc([
        {"x": 0, "y": 10}, {"x": 2, "y": 20}, {"x": 5, "y": 10},
    ], "x", "y")
    missing = trapezoid_auc([{"x": 0, "y": 10}], "x", "y")

    assert measured == {"value": 75.0, "reason": None, "n_points": 3}
    assert missing == {"value": None, "reason": "insufficient_points", "n_points": 1}

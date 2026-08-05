from agentbench_frame.doto.score import aggregate_episodes, build_score_curve, trapezoid_auc


def test_aggregate_is_candidate_oriented_and_preserves_failures():
    rows = [
        {"episode_id": "a", "scores": [8, 3], "winner": 0, "terminated_by": "normal", "errors": []},
        {"episode_id": "b", "scores": [2, 7], "winner": 1, "terminated_by": "normal", "errors": []},
        {"episode_id": "c", "scores": None, "winner": None, "terminated_by": "timeout", "errors": ["x"]},
    ]
    result = aggregate_episodes(rows, {"a": 0, "b": 1, "c": 0})
    assert result["score"] == 5.0
    assert result["win_rate"] == 1.0
    assert result["completion_rate"] == 2 / 3
    assert result["episodes"][1]["score_diff"] == 5.0
    assert result["episodes"][2]["score_diff"] is None


def test_draw_counts_as_half_win():
    result = aggregate_episodes([
        {"episode_id": "d", "scores": [4, 4], "winner": None,
         "terminated_by": "normal", "errors": []}], {"d": 0})
    assert result["win_rate"] == .5


def test_curve_keeps_failed_point_and_all_auc_axes():
    curve = build_score_curve([
        {"iteration": 0, "version": "v0", "status": "measured", "score": 1.0,
         "budget": {"rollouts": 2, "total_tokens": 0, "episode_reads": 0,
                    "frame_reads": 0, "wall_seconds": 1}},
        {"iteration": 1, "version": "bad", "status": "build_failed", "score": None,
         "budget": {"rollouts": 2, "total_tokens": 10, "episode_reads": 1,
                    "frame_reads": 2, "wall_seconds": 2}},
        {"iteration": 2, "version": "v2", "status": "measured", "score": 3.0,
         "budget": {"rollouts": 4, "total_tokens": 20, "episode_reads": 2,
                    "frame_reads": 4, "wall_seconds": 3}},
    ])
    assert curve["points"][1]["evo"] is None
    assert curve["points"][1]["gain"] is None
    assert curve["points"][2]["gain"] == 2.0
    assert set(curve["auc"]) == {"iteration", "rollout", "total_token", "episode_read", "frame_read", "wall_time"}


def test_auc_requires_two_distinct_x_values():
    result = trapezoid_auc([{"x": 1, "y": 2}, {"x": 1, "y": 3}], "x", "y")
    assert result["value"] is None
    assert result["reason"] == "insufficient_distinct_x"

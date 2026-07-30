import csv
from pathlib import Path


def _events():
    return [
        {"event_type": "version_created", "version_id": "v0", "act_id": "initial",
         "evaluation_status": "complete", "benchmark_score": 0.20},
        {"event_type": "candidate_selected", "iteration_id": "iter-000000",
         "version_id": "v0", "act_id": "initial"},
        {"event_type": "act_completed", "act_id": "act-1", "iteration_id": "iter-1",
         "total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
        {"event_type": "version_created", "version_id": "v1", "act_id": "act-1",
         "evaluation_status": "complete", "benchmark_score": 0.50},
        {"event_type": "candidate_selected", "iteration_id": "iter-1",
         "version_id": "v1", "act_id": "act-1"},
        {"event_type": "evaluation_completed", "version_id": "v1",
         "benchmark_score": 0.50, "wins": 2, "losses": 2, "draws": 0},
        {"event_type": "policy_kl_measured", "version_id": "v1",
         "local_policy_kl_trace": [0.1, 0.3]},
        {"event_type": "occupancy_measured", "version_id": "v1",
         "occupancy_shift": 0.25},
        {"event_type": "elo_updated", "version_id": "v1", "rating": 1516.0},
        {"event_type": "act_completed", "act_id": "act-2a", "iteration_id": "iter-2",
         "total_tokens": 60, "prompt_tokens": 50, "completion_tokens": 10},
        {"event_type": "act_completed", "act_id": "act-2b", "iteration_id": "iter-2",
         "total_tokens": 70, "prompt_tokens": 55, "completion_tokens": 15},
        {"event_type": "version_created", "version_id": "v2", "act_id": "act-2b",
         "evaluation_status": "incomplete", "benchmark_score": None},
        {"event_type": "candidate_selected", "iteration_id": "iter-2",
         "version_id": "v2", "act_id": "act-2b"},
    ]


def test_curve_rows_keep_performance_information_and_budget_separate():
    from agentbench_frame.hl.report import derive_curve_rows

    rows = derive_curve_rows(_events())

    assert len(rows) == 3
    assert rows[0]["raw_score"] == 0.20
    assert rows[1]["benchmark_score"] == 0.50
    assert rows[1]["gain"] == 0.30
    assert rows[1]["best_score_so_far"] == 0.50
    assert rows[1]["win_rate"] == 0.50
    assert rows[1]["rollman_elo"] == 1516.0
    assert rows[1]["mean_local_policy_kl"] == 0.20
    assert rows[1]["occupancy_shift"] == 0.25
    assert rows[1]["cumulative_total_tokens"] == 100
    assert rows[2]["benchmark_score"] is None
    assert rows[2]["gain"] is None
    assert rows[2]["cumulative_total_tokens"] == 230


def test_report_writes_csv_and_six_panel_raster_and_vector_plots(tmp_path):
    from agentbench_frame.hl.report import write_hl_report

    outputs = write_hl_report(_events(), tmp_path)

    assert outputs["curves_csv"].is_file()
    assert outputs["curves_png"].is_file()
    assert outputs["curves_svg"].is_file()
    assert outputs["curves_png"].stat().st_size > 10_000
    with outputs["curves_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["mean_local_policy_kl"] == "0.2"
    assert rows[2]["benchmark_score"] == ""


def test_match_table_preserves_each_game_and_invalid_status(tmp_path):
    from agentbench_frame.hl.report import write_hl_report

    events = _events() + [
        {"event_type": "match_completed", "match_id": "m1", "version_id": "v1",
         "opponent": "rank01", "seed": 1, "result": "win", "valid": True},
        {"event_type": "match_completed", "match_id": "m2", "version_id": "v1",
         "opponent": "rank02", "seed": 2, "result": None, "valid": False,
         "error": "process failed"},
    ]
    outputs = write_hl_report(events, tmp_path)

    with outputs["matches_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["match_id"] for row in rows] == ["m1", "m2"]
    assert rows[1]["valid"] == "False"
    assert rows[1]["result"] == ""

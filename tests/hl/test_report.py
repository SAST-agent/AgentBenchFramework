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


def test_no_change_proposal_records_zero_kl_and_zero_occupancy_shift():
    from agentbench_frame.hl.report import derive_curve_rows

    events = [
        {
            "event_type": "version_created",
            "version_id": "v0",
            "act_id": "origin",
            "evaluation_status": "complete",
            "benchmark_score": 0.25,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000000",
            "version_id": "v0",
            "act_id": "origin",
        },
        {
            "event_type": "policy_kl_measured",
            "version_id": "v0",
            "local_policy_kl_trace": [1.25],
        },
        {
            "event_type": "occupancy_measured",
            "version_id": "v0",
            "occupancy_shift": 9.0,
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000001",
            "parent_version_id": "v0",
            "selected_version_id": "v0",
            "candidate_version_ids": ["v1", "v2", "v3", "v4"],
        },
    ]

    rows = derive_curve_rows(events)

    assert [row["iteration"] for row in rows] == [0, 1]
    assert rows[1]["version_id"] == "v0"
    assert rows[1]["mean_local_policy_kl"] == 0.0
    assert rows[1]["occupancy_shift"] == 0.0


def test_k_iteration_budget_includes_rejected_siblings_after_selected_branch_act():
    from agentbench_frame.hl.report import derive_curve_rows

    events = [
        {
            "event_type": "version_created",
            "version_id": "v0",
            "act_id": "initial",
            "evaluation_status": "complete",
            "benchmark_score": 0.2,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000000",
            "version_id": "v0",
            "act_id": "initial",
        },
        {
            "event_type": "act_completed",
            "act_id": "act-b0",
            "iteration_id": "iter-1",
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
        },
        {
            "event_type": "version_created",
            "version_id": "v1",
            "act_id": "act-b0",
            "evaluation_status": "complete",
            "benchmark_score": 0.8,
        },
        {
            "event_type": "act_completed",
            "act_id": "act-b1",
            "iteration_id": "iter-1",
            "total_tokens": 70,
            "prompt_tokens": 55,
            "completion_tokens": 15,
        },
        {
            "event_type": "version_created",
            "version_id": "v2",
            "act_id": "act-b1",
            "evaluation_status": "complete",
            "benchmark_score": 0.3,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-1",
            "version_id": "v1",
            "act_id": "act-b0",
        },
    ]

    rows = derive_curve_rows(events)

    assert rows[1]["version_id"] == "v1"
    assert rows[1]["coding_agent_act"] == 2
    assert rows[1]["cumulative_total_tokens"] == 170


def test_report_writes_csv_and_three_panel_raster_and_vector_plots(tmp_path):
    from agentbench_frame.hl.report import write_hl_report

    outputs = write_hl_report(_events(), tmp_path)

    assert outputs["curves_csv"].is_file()
    assert outputs["curves_png"].is_file()
    assert outputs["curves_svg"].is_file()
    assert outputs["curves_png"].stat().st_size > 10_000
    svg = outputs["curves_svg"].read_text(encoding="utf-8")
    assert "Information Gain vs HL Iteration" in svg
    assert "Elo vs HL Iteration" in svg
    assert "Full-pool Win Rate vs HL Iteration" in svg
    assert "Occupancy shift" not in svg
    assert "Model budget" not in svg
    with outputs["curves_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["mean_local_policy_kl"] == "0.2"
    assert rows[2]["benchmark_score"] == ""


def test_curve_rows_use_final_evaluation_and_fixed_pool_elo():
    from agentbench_frame.hl.report import derive_curve_rows

    events = [
        {
            "event_type": "version_created",
            "version_id": "v0",
            "act_id": "initial",
            "evaluation_status": "incomplete",
            "benchmark_score": None,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000000",
            "version_id": "v0",
            "act_id": "initial",
        },
        {
            "event_type": "evaluation_completed",
            "version_id": "v0",
            "status": "complete",
            "benchmark_score": 0.0,
            "wins": 0,
            "losses": 3,
            "draws": 0,
        },
        {
            "event_type": "certification_completed",
            "version_id": "v0",
            "status": "complete",
            "score": 0.5,
            "passing_human_opponents": 8,
            "matches": [
                {"status": "complete", "opponent": "rank01", "seed": 1,
                 "result": "win"},
                {"status": "complete", "opponent": "rank02", "seed": 1,
                 "result": "loss"},
            ],
        },
        {
            "event_type": "elo_updated",
            "version_id": "v0",
            "phase": "learning",
            "rating": 9999.0,
        },
    ]

    rows = derive_curve_rows(events)

    assert rows[0]["evaluation_status"] == "complete"
    assert rows[0]["benchmark_score"] == 0.0
    assert rows[0]["full_pool_win_rate"] == 0.5
    assert rows[0]["rollman_elo"] != 9999.0
    assert rows[0]["mean_local_policy_kl"] == 0.0


def test_fixed_pool_elo_is_order_independent_and_comparable_across_panel_sizes():
    from agentbench_frame.hl.report import _fixed_pool_elo

    eighty_percent = [
        {"status": "complete", "result": "win"}
        for _ in range(8)
    ] + [
        {"status": "complete", "result": "loss"}
        for _ in range(2)
    ]
    ninety_percent_large = [
        {"status": "complete", "result": "win"}
        for _ in range(72)
    ] + [
        {"status": "complete", "result": "loss"}
        for _ in range(8)
    ]

    assert _fixed_pool_elo(eighty_percent) == _fixed_pool_elo(
        list(reversed(eighty_percent))
    )
    assert _fixed_pool_elo(ninety_percent_large) > _fixed_pool_elo(
        eighty_percent
    )


def test_origin_score_margin_comes_from_certification_when_no_reporting_panel():
    from agentbench_frame.hl.report import derive_curve_rows

    events = [
        {
            "event_type": "version_created",
            "version_id": "v0",
            "act_id": "bootstrap",
            "evaluation_status": "complete",
            "benchmark_score": 0.0,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000000",
            "version_id": "v0",
            "act_id": "bootstrap",
        },
        {
            "event_type": "certification_completed",
            "version_id": "v0",
            "status": "complete",
            "score": 0.5,
            "matches": [
                {
                    "status": "complete",
                    "result": "win",
                    "rollman_score": 20,
                    "ghosts_score": 10,
                },
                {
                    "status": "complete",
                    "result": "loss",
                    "rollman_score": -10,
                    "ghosts_score": 10,
                },
            ],
        },
    ]

    assert derive_curve_rows(events)[0]["mean_score_margin"] == -5.0


def test_single_iteration_axis_uses_only_integer_iteration_ticks():
    import matplotlib.pyplot as plt

    from agentbench_frame.hl.report import _set_iteration_axis

    figure, axis = plt.subplots()
    _set_iteration_axis(axis, [0])

    assert list(axis.get_xticks()) == [0]
    assert axis.get_xlim() == (-0.5, 0.5)
    plt.close(figure)


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


def test_curriculum_rows_capture_target_progress_and_stage_events():
    from agentbench_frame.hl.report import derive_curriculum_rows

    events = _events() + [
        {
            "event_type": "curriculum_started",
            "version_id": "v0",
            "active_target": "rank15",
            "locked_opponents": [f"rank{rank:02d}" for rank in range(1, 15)],
        },
        {
            "event_type": "curriculum_gate_completed",
            "version_id": "v0",
            "active_target": "rank15",
            "score": 0.0,
            "baseline": True,
            "stagnation_count": 0,
        },
        {
            "event_type": "curriculum_stage_promoted",
            "version_id": "v1",
            "completed_target": "rank15",
            "next_target": "rank14",
            "locked_opponents": [
                f"rank{rank:02d}" for rank in range(1, 16)
            ],
            "passing_human_opponents": 15,
        },
        {
            "event_type": "curriculum_target_selected",
            "version_id": "v1",
            "active_target": "rank14",
            "locked_opponents": [
                f"rank{rank:02d}" for rank in range(1, 16)
            ],
        },
    ]

    rows = derive_curriculum_rows(events)

    assert rows[0]["active_target"] == "rank15"
    assert rows[0]["locked_opponents"] == 14
    assert rows[1]["event"] == "gate"
    assert rows[1]["target_gate_score"] == 0.0
    assert rows[2]["event"] == "stage_promoted"
    assert rows[2]["passing_human_opponents"] == 15
    assert rows[3]["active_target"] == "rank14"


def test_report_writes_curriculum_csv_and_curve_fields(tmp_path):
    from agentbench_frame.hl.report import write_hl_report

    events = _events() + [
        {
            "event_type": "curriculum_gate_completed",
            "version_id": "v1",
            "active_target": "rank15",
            "score": 0.5,
            "baseline": False,
            "stagnation_count": 0,
        },
        {
            "event_type": "certification_completed",
            "version_id": "v1",
            "score": 0.9,
            "passing_human_opponents": 15,
        },
    ]

    outputs = write_hl_report(events, tmp_path)

    assert outputs["curriculum_csv"].is_file()
    with outputs["curves_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["active_target"] == "rank15"
    assert rows[1]["target_gate_score"] == "0.5"
    assert rows[1]["passing_human_opponents"] == "15"
    assert rows[1]["full_pool_win_rate"] == "0.9"


def test_k4_report_uses_proposal_cycle_as_integer_x_and_keeps_four_branches(
    tmp_path,
):
    from agentbench_frame.hl.report import (
        derive_branch_rows,
        derive_curve_rows,
        write_hl_report,
    )

    events = [
        {
            "event_type": "version_created",
            "version_id": "v0",
            "act_id": "bootstrap",
            "evaluation_status": "complete",
            "benchmark_score": 0.0,
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000000",
            "version_id": "v0",
            "act_id": "bootstrap",
        },
        *[
            {
                "event_type": "version_created",
                "version_id": f"v{index}",
                "act_id": f"act-b{index - 1}",
                "evaluation_status": "complete",
                "benchmark_score": score,
            }
            for index, score in enumerate((0.1, 0.0, 0.2, 0.0), start=1)
        ],
        *[
            {
                "event_type": "evaluation_completed",
                "version_id": f"v{index}",
                "status": "complete",
                "benchmark_score": score,
                "wins": int(score > 0),
                "draws": 0,
                "losses": int(score == 0),
                "matches": [
                    {
                        "status": "complete",
                        "result": "win" if score > 0 else "loss",
                        "rollman_score": 10 * index,
                        "ghosts_score": 50,
                    }
                ],
            }
            for index, score in enumerate((0.1, 0.0, 0.2, 0.0), start=1)
        ],
        *[
            {
                "event_type": "policy_kl_measured",
                "version_id": f"v{index}",
                "local_policy_kl_trace": [0.1 * index],
            }
            for index in range(1, 5)
        ],
        {
            "event_type": "search_parent_selected",
            "iteration_id": "iter-000001",
            "version_id": "v3",
            "act_id": "act-b2",
        },
        {
            "event_type": "reporting_panel_completed",
            "iteration_id": "iter-000001",
            "proposal_cycle": 1,
            "version_id": "v3",
            "status": "complete",
            "score": 0.625,
            "mean_score_margin": 14.0,
            "matches": [
                {"status": "complete", "result": "win"},
                {"status": "complete", "result": "loss"},
            ],
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000001",
            "parent_version_id": "v0",
            "selected_version_id": "v3",
            "candidate_version_ids": ["v1", "v2", "v3", "v4"],
        },
    ]

    rows = derive_curve_rows(events)
    branches = derive_branch_rows(events)

    assert [row["iteration"] for row in rows] == [0, 1]
    assert rows[1]["version_id"] == "v3"
    assert rows[1]["full_pool_win_rate"] == 0.625
    assert rows[1]["mean_score_margin"] == 14.0
    assert len(branches) == 4
    assert {row["iteration"] for row in branches} == {1}
    assert {row["branch_index"] for row in branches} == {0, 1, 2, 3}

    outputs = write_hl_report(events, tmp_path)
    assert outputs["branches_csv"].is_file()
    svg = outputs["curves_svg"].read_text(encoding="utf-8")
    assert "Score Margin vs HL Iteration" in svg
    assert "four rollout candidates" not in svg

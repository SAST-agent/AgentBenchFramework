import json
import hashlib
import tempfile
import unittest
from pathlib import Path


class LocalResearchReportTests(unittest.TestCase):
    def test_local_report_reads_first_hand_events_and_renders_research_fields(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "game" / "agent" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "run-1", "game": "game", "agent": "agent",
                "run_type": "eval", "benchmark_score": 0.75,
                "raw_score": 0.5, "evo_score": 0.75,
                "budget": {"learning_coding_agent_acts": 2},
                "benchmark_results": [{"case_id": "case-1", "outcome": "win", "valid": True}],
            }))
            (run_dir / "events.jsonl").write_text("\n".join([
                json.dumps({"event_type": "coding_agent_act", "act_id": "a1"}),
                json.dumps({"event_type": "act_evaluation", "act_id": "a1", "score": 0.5}),
                json.dumps({"event_type": "policy_kl_trace", "episode": 1, "trace": [0.1, 0.3]}),
            ]) + "\n")

            output = root / "site"
            builder = ReportBuilder(data_dir=str(root), output_dir=str(output))
            builder.build()
            html = (output / "index.html").read_text()

        self.assertIn("Information gain", html)
        self.assertIn("AUC / act", html)
        self.assertIn("case-1", html)
        self.assertIn("raw event records", html)

    def test_generals_zero_evo_score_and_raw_act_axis_are_preserved(self):
        from agentbench_frame.report.builder import ReportBuilder

        research = ReportBuilder._derive_research(
            {
                "benchmark_score": None,
                "raw_score": 0.0,
                "evo_score": 0.0,
                "gain": 0.0,
                "budget": {"learning_coding_agent_acts": 1},
            },
            [
                {"event_type": "evaluation", "phase": "raw", "score": 0.0},
                {"event_type": "evaluation", "phase": "evolved", "score": 0.0},
            ],
            "/missing/events.jsonl",
        )
        self.assertEqual(research["benchmark_score"], 0.0)
        self.assertEqual([point["x"] for point in research["score_history"]], [0, 1])

    def test_round2_formal_calibration_and_dense_data_remain_separate(self):
        from agentbench_frame.report.builder import ReportBuilder

        research = ReportBuilder._derive_research(
            {
                "benchmark_score": 0.3,
                "raw_score": 0.1,
                "evo_score_1": 0.2,
                "evo_score_2": 0.3,
                "gain_2": 0.2,
                "calibration_benchmark_id": "generals-hl-calibration-v1",
                "calibration_status": "complete",
                "calibration_score": 0.4,
                "calibration_in_target_range": True,
                "calibration_target_range": [0.2, 0.6],
                "budget": {"learning_coding_agent_acts": 1},
            },
            [
                {
                    "event_type": "evaluation",
                    "phase": "evolved_1_reproduction",
                    "score": 0.2,
                    "global_coding_agent_act": 1,
                },
                {
                    "event_type": "evaluation",
                    "phase": "evolved_2",
                    "score": 0.3,
                    "global_coding_agent_act": 2,
                },
                {
                    "event_type": "act_evaluation",
                    "score": 0.3,
                    "coding_agent_act": 1,
                    "global_coding_agent_act": 2,
                },
                {
                    "event_type": "calibration_result",
                    "split": "heldout",
                    "score": 0.4,
                },
                {
                    "event_type": "dense_episode_summary",
                    "case_id": "case-1",
                    "version": "v2",
                    "phase": "evaluation",
                    "completed_rounds_survived": 9,
                    "territory_share": {"terminal": 0.6, "auc": 4.2},
                    "target_army": {"terminal": 19, "auc": 120},
                    "army_margin": {"terminal": -31, "auc": -240},
                    "army_share": {"terminal": 0.55, "auc": 3.8},
                    "coin_share": {"terminal": None, "auc": None},
                    "net_main_pressure": {"terminal": 2.0, "auc": 8.0},
                },
                {
                    "event_type": "behavior_change_episode",
                    "version_before": "v1",
                    "version_after": "v2",
                    "action_disagreement": 1.0,
                    "action_disagreement_trace": [1.0],
                },
                {
                    "event_type": "behavior_change_episode",
                    "version_before": "v1",
                    "version_after": "v2",
                    "action_disagreement": 0.0,
                    "action_disagreement_trace": [0.0, 0.0, 0.0],
                },
                {
                    "event_type": "behavior_change",
                    "version_before": "v1",
                    "version_after": "v2",
                    "action_disagreement": 0.25,
                },
            ],
            "/missing/events.jsonl",
        )

        self.assertEqual(
            [point["x"] for point in research["score_history"]], [0, 1, 2]
        )
        self.assertEqual(
            [point["score"] for point in research["score_history"]],
            [0.1, 0.2, 0.3],
        )
        self.assertEqual(research["calibration"]["score"], 0.4)
        self.assertEqual(len(research["dense_history"]), 1)
        self.assertIsNone(
            research["dense_history"][0]["coin_share"]["terminal"]
        )
        self.assertEqual(
            research["dense_history"][0]["target_army"]["terminal"], 19
        )
        self.assertEqual(
            research["dense_history"][0]["army_margin"]["terminal"], -31
        )
        self.assertNotIn(
            0.4, [point["score"] for point in research["score_history"]]
        )
        behavior = research["action_disagreement_history"][0]
        self.assertEqual(
            behavior["episode_balanced_action_disagreement"], 0.5
        )
        self.assertEqual(
            behavior["decision_balanced_action_disagreement"], 0.25
        )

    def test_round2_panels_render_missing_dense_values_without_interpolation(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "28_generals" / "generals-hl" / "run-2"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "run-2",
                "game": "28_generals",
                "agent": "generals-hl",
                "run_type": "rule_iter",
                "status": "complete",
                "benchmark_score": 0.3,
                "raw_score": 0.1,
                "evo_score_1": 0.2,
                "evo_score_2": 0.3,
                "gain_2": 0.2,
                "calibration_benchmark_id": "generals-hl-calibration-v1",
                "calibration_status": "complete",
                "calibration_score": 0.4,
                "calibration_in_target_range": True,
                "calibration_target_range": [0.2, 0.6],
                "budget": {"learning_coding_agent_acts": 1},
            }))
            (run_dir / "events.jsonl").write_text(json.dumps({
                "event_type": "dense_episode_summary",
                "case_id": "case-1",
                "version": "v2",
                "phase": "evaluation",
                "completed_rounds_survived": 9,
                "territory_share": {"terminal": 0.6, "auc": 4.2},
                "army_share": {"terminal": 0.55, "auc": 3.8},
                "coin_share": {"terminal": None, "auc": None},
                "net_main_pressure": {"terminal": 2.0, "auc": 8.0},
            }) + "\n")

            output = root / "site"
            ReportBuilder(data_dir=str(root), output_dir=str(output)).build()
            html = (output / "index.html").read_text()

        self.assertIn("Calibration suite", html)
        self.assertIn("Dense trajectory diagnostics", html)
        self.assertIn("generals-hl-calibration-v1", html)
        self.assertIn("case-1", html)
        self.assertIn("missing", html)

    def test_recovery_report_links_cumulative_learning_budget_and_auc(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_run(run_id, summary, events=()):
                run_dir = root / "runs" / "game" / "agent" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "summary.json").write_text(json.dumps({
                    "run_id": run_id,
                    "game": "game",
                    "agent": "agent",
                    "run_type": "rule_iter",
                    **summary,
                }))
                (run_dir / "events.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in events)
                )

            parent_budget = {
                "learning_coding_agent_acts": 1,
                "learning_episodes": 12,
                "learning_env_steps": 100,
                "learning_total_tokens": 1000,
                "learning_time_s": 10.0,
            }
            source_budget = {
                "learning_coding_agent_acts": 1,
                "learning_episodes": 22,
                "learning_env_steps": 200,
                "learning_total_tokens": None,
                "learning_time_s": 20.0,
            }
            recovery_budget = {
                "learning_coding_agent_acts": 1,
                "learning_episodes": 0,
                "learning_env_steps": 0,
                "learning_total_tokens": 500,
                "learning_time_s": 5.0,
            }
            write_run("parent", {"status": "complete", "budget": parent_budget})
            write_run(
                "failed",
                {
                    "status": "provider_failed",
                    "budget": source_budget,
                    "parent_learning_budget": parent_budget,
                },
                [{
                    "event_type": "dense_episode_summary",
                    "case_id": "case-1",
                    "version": "v1",
                    "phase": "evaluation",
                    "completed_rounds_survived": 8,
                    "territory_share": {"terminal": 0.2},
                    "target_army": {"terminal": 10},
                    "army_margin": {"terminal": -5},
                    "coin_share": {"terminal": 0.3},
                    "net_main_pressure": {"terminal": -2},
                }],
            )
            write_run(
                "recovered",
                {
                    "status": "complete",
                    "evaluation_status": None,
                    "raw_score": 0.1,
                    "evo_score_1": 0.2,
                    "evo_score_2": 0.3,
                    "benchmark_score": 0.3,
                    "act_count": 3,
                    "round_act_count": 2,
                    "recovery_from_run_id": "failed",
                    "reused_heldout_calibration": True,
                    "source_learning_budget": source_budget,
                    "budget": recovery_budget,
                    "started_at": 3,
                },
                [{
                    "event_type": "dense_episode_summary",
                    "case_id": "case-1",
                    "version": "v2",
                    "phase": "evaluation",
                    "completed_rounds_survived": 9,
                    "territory_share": {"terminal": 0.25},
                    "target_army": {"terminal": 12},
                    "army_margin": {"terminal": -3},
                    "coin_share": {"terminal": 0.35},
                    "net_main_pressure": {"terminal": -1},
                }],
            )

            builder = ReportBuilder(
                data_dir=str(root), output_dir=str(root / "site")
            )
            builder.build()
            recovered = next(
                run for run in builder.runs if run["run_id"] == "recovered"
            )

        cumulative = recovered["research"]["cumulative_learning_budget"]
        self.assertEqual(cumulative["learning_coding_agent_acts"], 3)
        self.assertEqual(cumulative["learning_episodes"], 34)
        self.assertEqual(cumulative["learning_env_steps"], 300)
        self.assertIsNone(cumulative["learning_total_tokens"])
        self.assertEqual(cumulative["learning_time_s"], 35.0)
        self.assertIsNone(recovered["research"]["auc"]["auc_episode"])
        self.assertIsNone(
            recovered["research"]["auc"]["auc_coding_agent_act"]
        )
        self.assertEqual(
            [point["x"] for point in recovered["research"]["score_history"]],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [point["score"] for point in recovered["research"]["score_history"]],
            [0.1, 0.2, None, 0.3],
        )
        self.assertEqual(
            {
                item["version"]
                for item in recovered["research"]["dense_history"]
            },
            {"v1", "v2"},
        )
        pair = recovered["research"]["dense_pairs"][0]
        self.assertEqual(pair["case_id"], "case-1")
        self.assertEqual(pair["status"], "complete")
        self.assertEqual(pair["completed_rounds_survived_delta"], 1)
        self.assertAlmostEqual(
            pair["deltas"]["territory_share"]["terminal"], 0.05
        )
        self.assertEqual(
            pair["deltas"]["target_army"]["terminal"], 2
        )
        self.assertEqual(
            recovered["research"]["evaluation_status"], "complete"
        )

    def test_dense_pairing_keeps_mismatched_case_sets_missing(self):
        from agentbench_frame.report.builder import ReportBuilder

        pairs = ReportBuilder._pair_dense_history([
            {
                "case_id": "only-v1",
                "version": "v1",
                "phase": "evaluation",
                "completed_rounds_survived": 8,
            },
            {
                "case_id": "only-v2",
                "version": "v2",
                "phase": "evaluation",
                "completed_rounds_survived": 9,
            },
        ])

        self.assertEqual(
            {pair["status"] for pair in pairs},
            {"missing_v1", "missing_v2"},
        )
        self.assertTrue(all(
            pair["completed_rounds_survived_delta"] is None
            for pair in pairs
        ))

    def test_round3_preserves_failed_act_gap_gate_and_decision_classes(self):
        from agentbench_frame.report.builder import ReportBuilder

        research = ReportBuilder._derive_research(
            {
                "benchmark_score": 0.35,
                "raw_score": 0.0,
                "evo_score_1": 0.1,
                "evo_score_2": 0.2,
                "evo_score_3": 0.35,
                "gain_3": 0.35,
                "score_history": [0.0, 0.1, None, 0.2, 0.35],
                "AUC_coding_agent_act": None,
                "AUC_episode": None,
                "AUC_env_step": None,
                "AUC_token": None,
                "AUC_time": None,
                "auc_status": "unavailable_missing_score_point",
                "behavior_gate": {
                    "passed": True,
                    "conditions": {"behavior_changed": True},
                },
                "budget": {"learning_coding_agent_acts": 1},
            },
            [
                {
                    "event_type": "decision_class_summary",
                    "version": "v2",
                    "total": 100,
                    "counts": {"main_army_move": 97},
                    "rates": {"main_army_move": 0.97},
                },
                {
                    "event_type": "decision_class_summary",
                    "version": "v3",
                    "total": 100,
                    "counts": {"non_main_army_move": 30},
                    "rates": {"non_main_army_move": 0.3},
                },
                {
                    "event_type": "behavior_gate",
                    "passed": True,
                    "conditions": {"behavior_changed": True},
                    "dense_deltas": {
                        "terminal_territory_share": 0.05,
                    },
                },
            ],
            "/missing/events.jsonl",
        )

        self.assertEqual(research["evo_score"], 0.35)
        self.assertEqual(research["gain"], 0.35)
        self.assertIsNone(
            research["auc"]["auc_coding_agent_act"]
        )
        self.assertIsNone(research["auc"]["auc_episode"])
        self.assertEqual(
            [point["x"] for point in research["score_history"]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [point["score"] for point in research["score_history"]],
            [0.0, 0.1, None, 0.2, 0.35],
        )
        self.assertEqual(
            research["score_history"][2]["status"],
            "missing",
        )
        self.assertTrue(research["behavior_gate"]["passed"])
        self.assertEqual(
            [
                item["version"]
                for item in research["decision_class_history"]
            ],
            ["v2", "v3"],
        )
        self.assertEqual(
            research["decision_class_history"][1]["counts"][
                "non_main_army_move"
            ],
            30,
        )

    def test_round3_gate_and_decision_classes_render_in_dashboard(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (
                root / "runs" / "28_generals" / "generals-hl" / "v3-run"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "v3-run",
                "game": "28_generals",
                "agent": "generals-hl",
                "run_type": "rule_iter",
                "status": "complete",
                "evaluation_status": "complete",
                "benchmark_score": 0.3888888888888889,
                "raw_score": 0.0,
                "evo_score_3": 0.3888888888888889,
                "gain_3": 0.3888888888888889,
                "score_history": [0.0, 0.0, None, 0.0, 0.3888888888888889],
                "AUC_coding_agent_act": None,
                "budget": {"learning_coding_agent_acts": 1},
            }))
            (run_dir / "events.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "event_type": "decision_class_summary",
                        "version": "v2",
                        "total": 1072,
                        "counts": {
                            "main_army_move": 1000,
                            "non_main_army_move": 2,
                            "general_upgrade": 0,
                            "end_only": 70,
                        },
                        "rates": {
                            "main_army_move": 1000 / 1072,
                            "non_main_army_move": 2 / 1072,
                            "general_upgrade": 0.0,
                            "end_only": 70 / 1072,
                        },
                    }),
                    json.dumps({
                        "event_type": "decision_class_summary",
                        "version": "v3",
                        "total": 1072,
                        "counts": {
                            "main_army_move": 186,
                            "non_main_army_move": 691,
                            "general_upgrade": 125,
                            "end_only": 70,
                        },
                        "rates": {
                            "main_army_move": 186 / 1072,
                            "non_main_army_move": 691 / 1072,
                            "general_upgrade": 125 / 1072,
                            "end_only": 70 / 1072,
                        },
                    }),
                    json.dumps({
                        "event_type": "behavior_gate",
                        "passed": True,
                        "conditions": {
                            "behavior_changed": True,
                            "validation_complete": True,
                        },
                        "improved_dense_metrics": [
                            "terminal_territory_share"
                        ],
                        "dense_deltas": {
                            "terminal_territory_share": 0.255,
                        },
                    }),
                ]) + "\n"
            )
            output = root / "site"
            ReportBuilder(
                data_dir=str(root),
                output_dir=str(output),
            ).build()
            html = (output / "index.html").read_text()

        self.assertIn("Behavior gate", html)
        self.assertIn("Decision classes", html)
        self.assertIn("terminal territory share", html)
        self.assertIn("64.5%", html)
        self.assertIn("passed", html)

    def test_round4_reports_feedback_validation_and_nonblocking_diagnostics(self):
        from agentbench_frame.report.builder import ReportBuilder

        summary = {
            "benchmark_score": 0.5,
            "raw_score": 0.0,
            "evo_score_1": 0.0,
            "evo_score_2": 0.0,
            "evo_score_3": 7 / 18,
            "evo_score_4": 0.5,
            "gain_4": 0.5,
            "score_history": [0.0, 0.0, None, 0.0, 7 / 18, 0.5],
            "AUC_coding_agent_act": None,
            "feedback_read": {
                "episodes": 6,
                "decision_records": 42,
                "serialized_bytes": 12000,
            },
            "behavior_diagnostics": {
                "action_disagreement": 0.25,
                "validation_complete": True,
                "formal_evaluation_blocking": False,
                "dense_deltas": {
                    "terminal_army_margin": 10.0,
                },
            },
            "budget": {
                "learning_episodes": 6,
                "validation_episodes": 6,
                "evaluation_episodes": 18,
            },
        }
        events = [
            {
                "event_type": "feedback_read",
                "episodes": 6,
                "decision_records": 42,
                "serialized_bytes": 12000,
            },
            {
                "event_type": "behavior_diagnostics",
                "action_disagreement": 0.25,
                "validation_complete": True,
                "formal_evaluation_blocking": False,
                "dense_deltas": {
                    "terminal_army_margin": 10.0,
                },
            },
        ]

        research = ReportBuilder._derive_research(
            summary,
            events,
            "/missing/events.jsonl",
        )

        self.assertEqual(research["evo_score"], 0.5)
        self.assertEqual(research["gain"], 0.5)
        self.assertEqual(research["feedback_read"]["episodes"], 6)
        self.assertTrue(
            research["behavior_diagnostics"]["validation_complete"]
        )
        self.assertFalse(
            research["behavior_diagnostics"][
                "formal_evaluation_blocking"
            ]
        )
        self.assertIsNone(
            research["auc"]["auc_coding_agent_act"]
        )

    def test_round4_nonblocking_diagnostics_render_in_dashboard(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (
                root / "runs" / "28_generals"
                / "generals-hl" / "v4-run"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "v4-run",
                "game": "28_generals",
                "agent": "generals-hl",
                "run_type": "rule_iter",
                "status": "complete",
                "evaluation_status": "complete",
                "benchmark_score": 0.5,
                "raw_score": 0.0,
                "evo_score_4": 0.5,
                "gain_4": 0.5,
                "score_history": [
                    0.0, 0.0, None, 0.0, 7 / 18, 0.5,
                ],
                "AUC_coding_agent_act": None,
                "budget": {
                    "learning_episodes": 6,
                    "validation_episodes": 6,
                    "evaluation_episodes": 18,
                },
            }))
            (run_dir / "events.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "event_type": "feedback_read",
                        "episodes": 6,
                        "decision_records": 42,
                        "serialized_bytes": 12000,
                    }),
                    json.dumps({
                        "event_type": "behavior_diagnostics",
                        "action_disagreement": 0.25,
                        "validation_complete": True,
                        "valid_validation_case_count": 6,
                        "validation_case_count": 6,
                        "formal_evaluation_blocking": False,
                        "dense_deltas": {
                            "terminal_army_margin": 10.0,
                        },
                    }),
                ]) + "\n"
            )
            receipt = (
                root / "derived" / "28_generals"
                / "generals-hl" / "campaign-budget.json"
            )
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({
                "schema_version": "1.0",
                "status": "derived_prior_attempt_added",
                "success_run_id": "v4-run",
                "prior_attempt_run_id": "v4-prompt-failed",
                "mutation_policy": (
                    "inputs_immutable_separate_derived_receipt"
                ),
                "after": {
                    "learning_coding_agent_acts": 5,
                    "learning_episodes": 58,
                    "learning_env_steps": 30882,
                    "learning_game_agent_decision_steps": 12091,
                    "learning_primitive_commands": 50465,
                    "learning_prompt_tokens": None,
                    "learning_completion_tokens": None,
                    "learning_total_tokens": None,
                    "learning_time_s": 779.9,
                },
            }))

            output = root / "site"
            ReportBuilder(
                data_dir=str(root),
                output_dir=str(output),
            ).build()
            html = (output / "index.html").read_text()

        self.assertIn("Behavior diagnostics", html)
        self.assertIn("does not block formal evaluation", html)
        self.assertIn("Feedback episodes read", html)
        self.assertIn(">42<", html)
        self.assertIn("terminal army margin", html)
        self.assertIn("Validation episodes", html)
        self.assertIn(">58<", html)
        self.assertIn("derived campaign budget", html)

    def test_round5_rollback_and_dual_comparisons_are_preserved(self):
        from agentbench_frame.report.builder import ReportBuilder

        research = ReportBuilder._derive_research(
            {
                "benchmark_score": 0.5,
                "raw_score": 0.0,
                "evo_score_5": 0.5,
                "gain_5": 0.5,
                "parent_run_id": "v4-run",
                "parent_version": "v4",
                "parent_content_hash": "4" * 64,
                "starting_version": "v3",
                "rollback_source_version": "v3",
                "rollback_content_hash": "3" * 64,
                "budget": {"learning_coding_agent_acts": 1},
            },
            [
                {
                    "event_type": "version_rollback",
                    "parent_run_id": "v4-run",
                    "parent_version": "v4",
                    "parent_content_hash": "4" * 64,
                    "starting_version": "v3",
                    "rollback_source_version": "v3",
                    "rollback_content_hash": "3" * 64,
                },
                {
                    "event_type": "behavior_diagnostics",
                    "action_disagreement_vs_v3": 0.25,
                    "action_disagreement_vs_v4": 0.5,
                    "v3_probe_decision_count": 42,
                    "v4_probe_decision_count": 43,
                    "validation_complete": True,
                    "valid_validation_case_count": 6,
                    "validation_case_count": 6,
                    "dense_deltas_vs_v3": {
                        "terminal_army_margin": 10.0,
                    },
                    "dense_deltas_vs_v4": {
                        "terminal_army_margin": 20.0,
                    },
                    "formal_evaluation_blocking": False,
                },
            ],
            "/missing/events.jsonl",
        )

        self.assertEqual(research["evo_score"], 0.5)
        self.assertEqual(research["gain"], 0.5)
        self.assertEqual(
            research["version_rollback"]["starting_version"],
            "v3",
        )
        diagnostics = research["behavior_diagnostics"]
        self.assertEqual(
            diagnostics["action_disagreement_vs_v3"],
            0.25,
        )
        self.assertEqual(
            diagnostics["action_disagreement_vs_v4"],
            0.5,
        )
        self.assertEqual(
            diagnostics["dense_deltas_vs_v4"][
                "terminal_army_margin"
            ],
            20.0,
        )

    def test_round5_rollback_and_dual_comparisons_render(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (
                root / "runs" / "28_generals"
                / "generals-hl" / "v5-run"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "v5-run",
                "game": "28_generals",
                "agent": "generals-hl",
                "status": "complete",
                "benchmark_score": 0.5,
                "raw_score": 0.0,
                "evo_score_5": 0.5,
                "gain_5": 0.5,
                "parent_run_id": "v4-run",
                "parent_version": "v4",
                "parent_content_hash": "4" * 64,
                "starting_version": "v3",
                "rollback_source_version": "v3",
                "rollback_content_hash": "3" * 64,
                "budget": {"learning_coding_agent_acts": 1},
            }))
            (run_dir / "events.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "event_type": "version_rollback",
                        "parent_run_id": "v4-run",
                        "parent_version": "v4",
                        "parent_content_hash": "4" * 64,
                        "starting_version": "v3",
                        "rollback_source_version": "v3",
                        "rollback_content_hash": "3" * 64,
                    }),
                    json.dumps({
                        "event_type": "behavior_diagnostics",
                        "action_disagreement_vs_v3": None,
                        "action_disagreement_vs_v4": None,
                        "validation_complete": True,
                        "valid_validation_case_count": 6,
                        "validation_case_count": 6,
                        "dense_deltas_vs_v3": {
                            "terminal_army_margin": 10.0,
                        },
                        "dense_deltas_vs_v4": {
                            "terminal_army_margin": 20.0,
                        },
                        "formal_evaluation_blocking": False,
                    }),
                ]) + "\n"
            )
            summary_path = run_dir / "summary.json"
            events_path = run_dir / "events.jsonl"
            receipt = (
                root / "derived" / "28_generals"
                / "generals-hl" / "v5-behavior.json"
            )
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({
                "schema_version": "1.0",
                "status": "derived_behavior_measurement_recovered",
                "mutation_policy": (
                    "inputs_immutable_separate_derived_receipt"
                ),
                "success_run_id": "v5-run",
                "success_summary_ref": str(summary_path),
                "success_summary_sha256": hashlib.sha256(
                    summary_path.read_bytes()
                ).hexdigest(),
                "events_ref": str(events_path),
                "events_sha256": hashlib.sha256(
                    events_path.read_bytes()
                ).hexdigest(),
                "measurement": {
                    "v5_vs_v3": {
                        "decision_count": 42,
                        "changed_count": 10,
                        "action_disagreement": 0.25,
                    },
                    "v5_vs_v4": {
                        "decision_count": 43,
                        "changed_count": 21,
                        "action_disagreement": 0.5,
                    },
                    "v5_decision_classes": {
                        "total": 85,
                        "counts": {"main_army_move": 20},
                        "rates": {"main_army_move": 20 / 85},
                    },
                },
            }))

            output = root / "site"
            ReportBuilder(
                data_dir=str(root),
                output_dir=str(output),
            ).build()
            html = (output / "index.html").read_text()

        self.assertIn("Rollback lineage", html)
        self.assertIn("v4 → v3", html)
        self.assertIn("Action disagreement vs v3", html)
        self.assertIn("Action disagreement vs v4", html)
        self.assertIn("v5 − v3", html)
        self.assertIn("v5 − v4", html)
        self.assertIn("derived behavior receipt", html)
        self.assertIn(">0.25<", html)
        self.assertIn(">0.50<", html)

    def test_controlled_reference_policy_kl_keeps_missing_curve_gap_out_of_ig(self):
        from agentbench_frame.report.builder import ReportBuilder

        def transition(before, after, mean, complete):
            sensitivity = {
                epsilon: {
                    "mean_kl_nats": (
                        None if mean is None else mean * multiplier
                    ),
                    "coverage": {"complete": complete, "total": 12},
                }
                for epsilon, multiplier in (
                    ("0.001", 1.2),
                    ("0.01", 1.0),
                    ("0.05", 0.8),
                    ("0.1", 0.6),
                )
            }
            return {
                "version_before": before,
                "version_after": after,
                "mean_kl_nats": mean,
                "coverage": {"complete": complete, "total": 12},
                "sensitivity": sensitivity,
            }

        research = ReportBuilder._derive_research(
            {
                "controlled_reference_policy_kl": {
                    "primary_epsilon": "0.01",
                    "reference_state_count": 12,
                    "transitions": [
                        transition("v0", "v1", 1.0, 12),
                        transition("v1", "v2", None, 11),
                        transition("v2", "v3", 3.0, 12),
                        transition("v3", "v4", 4.0, 12),
                    ],
                },
            },
            [
                {
                    "event_type": "action_space_count",
                    "measurement_state_id": "seed289101-seat0-decision2",
                    "support_size": "123456789012345678901234567890",
                    "status": "complete",
                    "elapsed_time_s": 1.5,
                }
            ],
            "/missing/events.jsonl",
        )

        self.assertEqual(research["ig_history"], [])
        self.assertEqual(
            [
                point["mean_kl_nats"]
                for point in research[
                    "controlled_reference_policy_kl_history"
                ]
            ],
            [1.0, None, 3.0, 4.0],
        )
        self.assertEqual(
            [
                [point["label"] for point in segment]
                for segment in research[
                    "controlled_reference_policy_kl_svg_segments"
                ]
            ],
            [["v0→v1"], ["v2→v3", "v3→v4"]],
        )
        self.assertEqual(
            len(
                research[
                    "controlled_reference_policy_kl_sensitivity"
                ]
            ),
            16,
        )
        support = research["controlled_reference_support_states"][0]
        self.assertEqual(
            support["support_size"],
            "123456789012345678901234567890",
        )
        self.assertAlmostEqual(support["log10_support_size"], 29.0915, 4)

    def test_controlled_reference_policy_kl_panel_states_scope_and_coverage(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (
                root
                / "runs"
                / "28_generals"
                / "generals-policy-kl"
                / "measurement-1"
            )
            run_dir.mkdir(parents=True)
            sensitivity = {
                epsilon: {
                    "mean_kl_nats": value,
                    "coverage": {"complete": 11, "total": 12},
                }
                for epsilon, value in (
                    ("0.001", 5.0),
                    ("0.01", 4.0),
                    ("0.05", 3.0),
                    ("0.1", 2.0),
                )
            }
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "measurement-1",
                "game": "28_generals",
                "agent": "generals-policy-kl",
                "run_type": "measurement",
                "status": "incomplete_policy_measurement",
                "controlled_reference_policy_kl": {
                    "primary_epsilon": "0.01",
                    "reference_state_count": 12,
                    "transitions": [{
                        "version_before": "v0",
                        "version_after": "v1",
                        "mean_kl_nats": None,
                        "coverage": {"complete": 11, "total": 12},
                        "sensitivity": sensitivity,
                    }],
                },
            }))
            (run_dir / "events.jsonl").write_text(json.dumps({
                "event_type": "action_space_count",
                "measurement_state_id": "seed289101-seat0-decision2",
                "support_size": "1000000000000000000000000000000",
                "status": "complete",
                "elapsed_time_s": 2.0,
            }) + "\n")

            output = root / "site"
            ReportBuilder(
                data_dir=str(root),
                output_dir=str(output),
            ).build()
            html = (output / "index.html").read_text()

        self.assertIn("Controlled-reference policy KL", html)
        self.assertIn("epsilon = 0.01", html)
        self.assertIn("early controlled real states", html)
        self.assertIn("not epistemic information gain", html)
        self.assertIn("11 / 12", html)
        self.assertIn("seed289101-seat0-decision2", html)

    def test_comparison_report_accepts_measurement_run_without_win_rate(self):
        from agentbench_frame.report.builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "game" / "measurement" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id": "run-1",
                "game": "game",
                "agent": "measurement",
                "run_type": "measurement",
                "status": "complete",
                "win_rate": None,
                "total_episodes": None,
            }))
            (run_dir / "events.jsonl").write_text("")

            output = root / "site"
            ReportBuilder(
                data_dir=str(root),
                output_dir=str(output),
            ).build()

            compare_html = (output / "compare.html").read_text()

        self.assertIn("measurement", compare_html)
        self.assertIn("0.0%", compare_html)


if __name__ == "__main__":
    unittest.main()

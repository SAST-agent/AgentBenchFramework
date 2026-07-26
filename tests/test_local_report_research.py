import json
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
                    "army_share": {"terminal": 0.55, "auc": 3.8},
                    "coin_share": {"terminal": None, "auc": None},
                    "net_main_pressure": {"terminal": 2.0, "auc": 8.0},
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
        self.assertNotIn(
            0.4, [point["score"] for point in research["score_history"]]
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


if __name__ == "__main__":
    unittest.main()

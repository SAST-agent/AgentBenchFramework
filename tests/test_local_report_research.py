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


if __name__ == "__main__":
    unittest.main()

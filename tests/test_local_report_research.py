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
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 2,
                    "measurement_status": "incomplete",
                    "trace": [0.2, None],
                    "trajectory_kl_episode": 999.0,
                    "mean_local_policy_kl": 999.0,
                    "decision_steps": 2,
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 3,
                    "measurement_status": "complete",
                    "trace": [0.2, 0.4],
                    "trajectory_kl_episode": 999.0,
                    "mean_local_policy_kl": 999.0,
                    "decision_steps": 2,
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 4,
                    "measurement_status": "quarantined",
                    "trace": [0.7],
                    "decision_steps": 1,
                    "decisions": [{"local_policy_kl": 0.7}],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 5,
                    "measurement_status": "complete",
                    "trace": [0.8],
                    "decision_steps": 2,
                    "decisions": [
                        {"local_policy_kl": 0.8},
                        {"local_policy_kl": 0.1},
                    ],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 6,
                    "trace": ["0.1", 0.3],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 7,
                    "trace": [True, 0.3],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 8,
                    "trace": [0.1, 0.3],
                    "decision_steps": 2,
                    "decisions": [
                        {"local_policy_kl": 0.1},
                        {"local_policy_kl": 0.3},
                    ],
                }),
            ]) + "\n")

            output = root / "site"
            builder = ReportBuilder(data_dir=str(root), output_dir=str(output))
            builder.build()
            ig_history = builder.runs[0]["research"]["ig_history"]
            ig_chart = builder.runs[0]["research"]["ig_chart"]
            html = (output / "index.html").read_text()

        self.assertAlmostEqual(ig_history[0]["trajectory_kl_episode"], 0.4)
        self.assertAlmostEqual(ig_history[0]["mean_local_policy_kl"], 0.2)
        self.assertEqual(ig_history[0]["status"], "complete")
        self.assertIsNone(ig_history[1]["trajectory_kl_episode"])
        self.assertIsNone(ig_history[1]["mean_local_policy_kl"])
        self.assertEqual(ig_history[1]["status"], "incomplete")
        self.assertAlmostEqual(ig_history[2]["trajectory_kl_episode"], 0.6)
        self.assertAlmostEqual(ig_history[2]["mean_local_policy_kl"], 0.3)
        self.assertIsNone(ig_history[3]["trajectory_kl_episode"])
        self.assertEqual(ig_history[3]["status"], "quarantined")
        self.assertIsNone(ig_history[4]["trajectory_kl_episode"])
        self.assertEqual(ig_history[4]["status"], "incomplete")
        for point in ig_history[5:]:
            self.assertIsNone(point["trajectory_kl_episode"])
            self.assertEqual(point["status"], "incomplete")
        self.assertEqual(len(ig_chart["segments"]), 2)
        self.assertIn("Information gain", html)
        self.assertIn("Trajectory KL", html)
        self.assertIn("Mean local policy KL", html)
        self.assertIn("nats / episode", html)
        self.assertIn("nats / decision", html)
        self.assertIn('aria-label="Trajectory KL by episode"', html)
        self.assertIn('data-segment-count="2"', html)
        self.assertIn("0.40", html)
        self.assertIn("0.20", html)
        self.assertIn("0.60", html)
        self.assertIn("missing", html)
        self.assertNotIn("999.00", html)
        self.assertIn("AUC / act", html)
        self.assertIn("case-1", html)
        self.assertIn("raw event records", html)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path


class LocalResearchReportTests(unittest.TestCase):
    def test_report_does_not_promote_legacy_or_forged_rich_events_to_formal_ig(self):
        from agentbench_frame.eval.information_gain import policy_kl
        from agentbench_frame.report.builder import ReportBuilder

        local = policy_kl([0.75, 0.25], [0.5, 0.5], epsilon=0.01)
        forged = {
            "event_type": "policy_kl_trace",
            "episode": 2,
            "version_before": "v1",
            "version_after": "v2",
            "measurement_profile": "24_miracle_policy_information_gain_v2",
            "measurement_status": "complete",
            "epsilon": 0.01,
            "direction": "new||old",
            "log_base": "e",
            "rollout_source": "new_policy",
            "estimand": "epsilon_regularized_local_kl_sum_under_new_policy_occupancy",
            "information_gain_estimand": "epsilon_regularized_mean_local_policy_kl_under_new_policy_occupancy",
            "aggregation": "arithmetic_mean",
            "information_gain_unit": "nats / decision",
            "local_policy_kl_sum_unit": "nats / episode",
            "decision_steps": 1,
            "trace": [local + 0.25],
            "trajectory_kl_episode": local + 0.25,
            "mean_local_policy_kl": local + 0.25,
            "information_gain": local + 0.25,
            "local_policy_kl_sum": local + 0.25,
            "errors": [],
            "metadata": {},
            "decisions": [{
                "decision_step": 1,
                "context_ref": "context",
                "action_schema_version": "actions-v1",
                "support_id": "support",
                "legal_action_ids": ["a", "b"],
                "selected_action_id": "a",
                "new_distribution": {"a": 0.75, "b": 0.25},
                "old_distribution": {"a": 0.5, "b": 0.5},
                "new_probabilities": [0.75, 0.25],
                "old_probabilities": [0.5, 0.5],
                "local_policy_kl": local + 0.25,
                "errors": [],
            }],
        }
        valid = json.loads(json.dumps(forged))
        valid["episode"] = 3
        valid["trace"] = [local]
        valid["trajectory_kl_episode"] = local
        valid["mean_local_policy_kl"] = local
        valid["information_gain"] = local
        valid["local_policy_kl_sum"] = local
        valid["decisions"][0]["local_policy_kl"] = local
        history = ReportBuilder._derive_research(
            {},
            [
                {"event_type": "policy_kl_trace", "episode": 1, "trace": [0.2]},
                forged,
                valid,
            ],
            "missing-events.jsonl",
        )["ig_history"]

        self.assertEqual(history[0]["status"], "legacy_unverified")
        self.assertIsNone(history[0]["information_gain"])
        self.assertEqual(history[1]["status"], "incomplete")
        self.assertIsNone(history[1]["information_gain"])
        self.assertEqual(history[2]["status"], "complete")
        self.assertAlmostEqual(history[2]["information_gain"], local)

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
            oversized_integer = "1" * 5001
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
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 9,
                    "measurement_status": "complete",
                    "trace": [0.1, 0.3],
                    "decisions": [
                        {"local_policy_kl": 0.1},
                        {"local_policy_kl": 0.3},
                    ],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 10,
                    "trace": [1e308, 1e308],
                }),
                json.dumps({
                    "event_type": "policy_kl_trace",
                    "episode": 11,
                    "trace": [10 ** 3999],
                }),
            ]) + "\n" + (
                '{"event_type":"policy_kl_trace","episode":12,"trace":['
                + oversized_integer
                + "]}\n"
            ))

            output = root / "site"
            builder = ReportBuilder(data_dir=str(root), output_dir=str(output))
            builder.build()
            ig_history = builder.runs[0]["research"]["ig_history"]
            ig_chart = builder.runs[0]["research"]["ig_chart"]
            quality = builder.runs[0]["research"]["quality"]
            html = (output / "index.html").read_text(encoding="utf-8")

        self.assertIsNone(ig_history[0]["trajectory_kl_episode"])
        self.assertIsNone(ig_history[0]["mean_local_policy_kl"])
        self.assertIsNone(ig_history[0]["information_gain"])
        self.assertIsNone(ig_history[0]["local_policy_kl_sum"])
        self.assertEqual(ig_history[0]["status"], "legacy_unverified")
        self.assertEqual(ig_history[0]["estimand"], "legacy_unspecified")
        self.assertEqual(len(ig_history), 11)
        self.assertEqual(quality["malformed_lines"], 1)
        self.assertIsNone(ig_history[1]["trajectory_kl_episode"])
        self.assertIsNone(ig_history[1]["mean_local_policy_kl"])
        self.assertEqual(ig_history[1]["status"], "legacy_unverified")
        self.assertIsNone(ig_history[2]["trajectory_kl_episode"])
        self.assertIsNone(ig_history[2]["mean_local_policy_kl"])
        self.assertIsNone(ig_history[2]["information_gain"])
        self.assertIsNone(ig_history[3]["trajectory_kl_episode"])
        self.assertEqual(ig_history[3]["status"], "legacy_unverified")
        self.assertIsNone(ig_history[4]["trajectory_kl_episode"])
        self.assertEqual(ig_history[4]["status"], "legacy_unverified")
        for point in ig_history[5:]:
            self.assertIsNone(point["trajectory_kl_episode"])
            self.assertEqual(point["status"], "legacy_unverified")
        self.assertEqual(len(ig_chart["segments"]), 0)
        self.assertIn("Information gain", html)
        self.assertIn("Policy IG", html)
        self.assertIn("Local KL sum", html)
        self.assertIn("nats / episode", html)
        self.assertIn("nats / decision", html)
        self.assertIn('aria-label="Policy information gain by episode"', html)
        self.assertIn('data-segment-count="0"', html)
        self.assertIn("missing", html)
        self.assertNotIn("999.00", html)
        self.assertIn("AUC / act", html)
        self.assertIn("case-1", html)
        self.assertIn("raw event records", html)


if __name__ == "__main__":
    unittest.main()

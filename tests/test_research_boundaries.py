import json
import tempfile
import unittest
from pathlib import Path


class ResearchBoundaryTests(unittest.TestCase):
    def test_multi_axis_auc_uses_same_gap_semantics_for_all_budgets(self):
        from agentbench_frame.eval.curves import multi_axis_auc

        points = [
            {"coding_agent_act": 0, "episode": 0, "env_step": 0, "token": 0, "time_s": 0, "score": 0.0},
            {"coding_agent_act": 1, "episode": 2, "env_step": 4, "token": 10, "time_s": 1, "score": 1.0},
            {"coding_agent_act": 2, "episode": 4, "env_step": 8, "token": 20, "time_s": 2, "score": None},
            {"coding_agent_act": 3, "episode": 6, "env_step": 12, "token": 30, "time_s": 3, "score": 0.5},
        ]

        auc = multi_axis_auc(points)

        self.assertEqual(auc["auc_coding_agent_act"], 0.5)
        self.assertEqual(auc["auc_episode"], 1.0)
        self.assertEqual(auc["auc_env_step"], 2.0)
        self.assertEqual(auc["auc_token"], 5.0)
        self.assertEqual(auc["auc_time_s"], 0.5)

    def test_occupancy_log_keeps_reference_samples_and_separate_shift(self):
        from agentbench_frame.tracking.run import Run

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", data_dir=tmp)
            run.log_occupancy(
                episode=1,
                version_before="v0",
                version_after="v1",
                state_ids=["a", "a"],
                reference_state_ids=["b", "b"],
            )
            run.finish()
            records = [json.loads(line) for line in Path(run.run_dir, "events.jsonl").read_text().splitlines()]

        occupancy = records[0]
        self.assertEqual(occupancy["state_ids"], ["a", "a"])
        self.assertEqual(occupancy["reference_state_ids"], ["b", "b"])
        self.assertGreater(occupancy["occupancy_shift"], 0.0)
        self.assertNotIn("information_gain", occupancy)

    def test_event_quality_warns_on_bad_lines_and_unknown_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        report = inspect_event_lines([
            json.dumps({"event_id": "e1", "event_type": "episode", "run_id": "r"}),
            "not-json",
            json.dumps({"event_id": "e1", "event_type": "future_event", "run_id": "r"}),
            json.dumps({"event_type": "episode", "run_id": "r"}),
        ])

        self.assertEqual(report.malformed_lines, 1)
        self.assertEqual(report.duplicate_event_ids, 1)
        self.assertEqual(report.unknown_event_types, 1)
        self.assertEqual(report.missing_event_ids, 1)
        self.assertTrue(report.warnings)

    def test_round2_event_types_are_known_without_hiding_future_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        known = [
            "calibration_spec",
            "calibration_result",
            "dense_trajectory",
            "dense_episode_summary",
            "dense_metric_error",
            "lineage_import",
            "behavior_change_episode",
        ]
        lines = [
            json.dumps(
                {
                    "event_id": f"e{index}",
                    "event_type": event_type,
                    "run_id": "r",
                }
            )
            for index, event_type in enumerate(known)
        ]
        lines.append(
            json.dumps(
                {
                    "event_id": "future",
                    "event_type": "future_round3_event",
                    "run_id": "r",
                }
            )
        )

        report = inspect_event_lines(lines)

        self.assertEqual(report.valid_events, 8)
        self.assertEqual(report.unknown_event_types, 1)

    def test_round3_diagnostics_are_known_event_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        lines = [
            json.dumps({
                "event_type": event_type,
                "event_id": f"event-{index}",
                "run_id": "run-v3",
            })
            for index, event_type in enumerate(
                (
                    "learning_validation",
                    "decision_class_summary",
                    "behavior_gate",
                    "behavior_measurement_error",
                )
            )
        ]

        report = inspect_event_lines(lines)

        self.assertEqual(report.unknown_event_types, 0)

    def test_round4_feedback_and_diagnostics_are_known_event_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        lines = [
            json.dumps({
                "event_type": event_type,
                "event_id": f"event-{index}",
                "run_id": "run-v4",
            })
            for index, event_type in enumerate(
                (
                    "feedback_read",
                    "critical_window_selection",
                    "behavior_diagnostics",
                )
            )
        ]

        report = inspect_event_lines(lines)

        self.assertEqual(report.unknown_event_types, 0)

    def test_round5_rollback_is_a_known_event_type(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        report = inspect_event_lines([
            json.dumps({
                "event_type": "version_rollback",
                "event_id": "event-v5-rollback",
                "run_id": "run-v5",
            })
        ])

        self.assertEqual(report.unknown_event_types, 0)

    def test_round7_champion_audit_events_are_known_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        lines = [
            json.dumps({
                "event_type": event_type,
                "event_id": f"event-{index}",
                "run_id": "run-v7",
            })
            for index, event_type in enumerate(
                (
                    "champion_validation_gate",
                    "champion_sealed_claim",
                    "recovery_import",
                )
            )
        ]

        report = inspect_event_lines(lines)

        self.assertEqual(report.unknown_event_types, 0)

    def test_scientific_attribution_events_are_known_types(self):
        from agentbench_frame.tracking.quality import inspect_event_lines

        event_types = (
            "attribution_policy_materialized",
            "attribution_game_result",
            "attribution_factorial_effect",
            "diagnostic_state_selected",
            "diagnostic_policy_probe",
            "trajectory_divergence",
            "attribution_report_frozen",
        )
        report = inspect_event_lines([
            json.dumps({
                "event_type": event_type,
                "event_id": f"attribution-{index}",
                "run_id": "run-attribution",
            })
            for index, event_type in enumerate(event_types)
        ])

        self.assertEqual(report.unknown_event_types, 0)


if __name__ == "__main__":
    unittest.main()

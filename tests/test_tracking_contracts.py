import json
import tempfile
import unittest
from pathlib import Path


class TrackingContractTests(unittest.TestCase):
    def test_tracked_env_records_terminated_and_truncated_without_changing_api(self):
        from agentbench_frame.tracking.wrappers import TrackedEnv

        class FiveTupleEnv:
            def reset(self):
                return "initial"

            def step(self, _action):
                return "next", 1.0, False, True, {"termination_reason": "budget"}

        records = []
        env = TrackedEnv(FiveTupleEnv(), on_step=records.append)
        result = env.step("action")

        self.assertEqual(result, ("next", 1.0, False, True, {"termination_reason": "budget"}))
        self.assertTrue(records[0].truncated)
        self.assertFalse(records[0].terminated)
        self.assertEqual(records[0].termination_reason, "budget")

    def test_jsonl_writer_appends_without_erasing_previous_events(self):
        from agentbench_frame.tracking.writer import JSONLWriter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            first = JSONLWriter(str(path))
            first.write({"event": "first"})
            first.close()
            second = JSONLWriter(str(path))
            second.write({"event": "second"})
            second.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["first", "second"])

    def test_run_events_have_forward_compatible_common_fields(self):
        from agentbench_frame.tracking.run import Run

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", data_dir=tmp)
            run.write("custom", value=3)
            run.finish()
            records = [json.loads(line) for line in Path(run.run_dir, "events.jsonl").read_text().splitlines()]

            self.assertEqual(len(records), 1)
            record = records[0]
            for field in ("schema_version", "event_id", "event_type", "event", "run_id", "created_at"):
                self.assertIn(field, record)
            self.assertEqual(record["event"], "custom")
            self.assertEqual(record["event_type"], "custom")

    def test_run_accepts_dataclass_step_records_from_tracked_env(self):
        from agentbench_frame.tracking.run import Run

        class OneStepEnv:
            def reset(self):
                return "initial"

            def step(self, _action):
                return "next", 1.0, True, {"termination_reason": "rule"}

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", data_dir=tmp)
            env = run.wrap_env(OneStepEnv())
            env.step("action")
            run.finish()
            records = [json.loads(line) for line in Path(run.run_dir, "events.jsonl").read_text().splitlines()]

            step = next(record for record in records if record["event_type"] == "step")
            self.assertEqual(step["action"], "action")
            self.assertEqual(step["termination_reason"], "rule")

    def test_budget_ledger_separates_phases_and_preserves_unknown_usage(self):
        from agentbench_frame.tracking.budget import BudgetLedger

        ledger = BudgetLedger()
        ledger.add("learning", episodes=2, env_steps=5, prompt_tokens=10, completion_tokens=4, time_s=1.5)
        ledger.add("evaluation", episodes=1, env_steps=3, prompt_tokens=None, completion_tokens=None, time_s=2.0)
        snapshot = ledger.snapshot()

        self.assertEqual(snapshot["learning_episodes"], 2)
        self.assertEqual(snapshot["learning_coding_agent_acts"], 0)
        self.assertEqual(snapshot["evaluation_env_steps"], 3)
        self.assertEqual(snapshot["total_episodes"], 3)
        self.assertIsNone(snapshot["evaluation_total_tokens"])
        self.assertIsNone(snapshot["total_tokens"])
        self.assertAlmostEqual(snapshot["total_time_s"], 3.5)

    def test_budget_ledger_separates_calibration_without_polluting_learning(self):
        from agentbench_frame.tracking.budget import BudgetLedger

        ledger = BudgetLedger()
        ledger.add(
            "calibration",
            episodes=3,
            env_steps=30,
            game_agent_decision_steps=15,
            primitive_commands=45,
            time_s=2.0,
        )
        ledger.add(
            "learning",
            episodes=2,
            env_steps=20,
            game_agent_decision_steps=10,
            primitive_commands=30,
            time_s=1.0,
        )
        ledger.add(
            "evaluation",
            episodes=1,
            env_steps=10,
            game_agent_decision_steps=5,
            primitive_commands=15,
            time_s=0.5,
        )

        snapshot = ledger.snapshot()

        self.assertEqual(snapshot["calibration_episodes"], 3)
        self.assertEqual(snapshot["learning_episodes"], 2)
        self.assertEqual(snapshot["evaluation_episodes"], 1)
        self.assertEqual(snapshot["total_episodes"], 6)
        self.assertEqual(snapshot["total_env_steps"], 60)
        self.assertEqual(snapshot["total_game_agent_decision_steps"], 30)
        self.assertEqual(snapshot["total_primitive_commands"], 90)
        self.assertAlmostEqual(snapshot["total_time_s"], 3.5)

    def test_act_recorder_keeps_failed_act_and_creates_logical_version_for_unchanged_content(self):
        from agentbench_frame.tracking.iteration import VersionedActRecorder

        events = []
        recorder = VersionedActRecorder(emit=events.append)
        act = recorder.begin_act(provider="test", version_before="v0")
        finished = recorder.finish_act(
            act.act_id,
            status="completed",
            snapshot_content_hash="same-content",
            changed_files=[],
        )
        failed = recorder.begin_act(provider="test", version_before=finished.version_after)
        failed = recorder.finish_act(failed.act_id, status="failed", error="crash")

        self.assertNotEqual(finished.version_after, "v0")
        self.assertEqual(finished.changed_files, [])
        self.assertIsNone(failed.version_after)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(len(events), 2)

    def test_run_summary_scores_named_agent_and_persists_budget(self):
        from agentbench_frame.tracking.run import Run

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", data_dir=tmp)
            run.log_budget("learning", episodes=4, env_steps=8, time_s=1.0)
            run.log_budget("evaluation", episodes=0, env_steps=0, time_s=0.5)
            run.log_episode(1.0, 2, winner=1, agent_player_id=1)
            run.log_episode(0.0, 3, winner=-1, agent_player_id=1)
            summary = run.finish()

            self.assertAlmostEqual(summary["benchmark_score"], 0.75)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["draws"], 1)
            self.assertEqual(summary["budget"]["learning_episodes"], 4)
            self.assertEqual(summary["budget"]["total_env_steps"], 13)
            self.assertAlmostEqual(summary["budget"]["total_time_s"], 1.5)

    def test_run_persists_raw_policy_trace_and_occupancy_observations(self):
        from agentbench_frame.tracking.run import Run

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", data_dir=tmp)
            run.log_policy_kl_trace(
                episode=2,
                version_before="v1",
                version_after="v2",
                trace=[0.1, 0.2],
                context_refs=["s0", "s1"],
                epsilon=0.01,
            )
            run.log_occupancy(
                episode=2,
                version_before="v1",
                version_after="v2",
                state_ids=["s0", "s1", "s1"],
            )
            run.finish()

            records = [json.loads(line) for line in Path(run.run_dir, "events.jsonl").read_text().splitlines()]
            trace = next(record for record in records if record["event_type"] == "policy_kl_trace")
            occupancy = next(record for record in records if record["event_type"] == "occupancy")
            self.assertEqual(trace["trace"], [0.1, 0.2])
            self.assertEqual(trace["context_refs"], ["s0", "s1"])
            self.assertEqual(occupancy["state_ids"], ["s0", "s1", "s1"])

    def test_run_attaches_budget_coordinates_to_act_evaluation(self):
        from agentbench_frame.tracking.run import Run

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.start("game", "agent", run_type="rule_iter", data_dir=tmp)
            run.log_budget("learning", episodes=2, env_steps=6, prompt_tokens=10,
                           completion_tokens=4, time_s=1.5)
            act = run.begin_act("synthetic", version_before="v0")
            run.finish_act(act.act_id, "completed", snapshot_content_hash="hash")
            event = run.record_act_evaluation(act.act_id, {"score": 0.5})
            run.finish()

        self.assertEqual(event["coding_agent_act"], 0)
        self.assertEqual(event["episode"], 2)
        self.assertEqual(event["env_step"], 6)
        self.assertEqual(event["token"], 14)
        self.assertAlmostEqual(event["time_s"], 1.5)


if __name__ == "__main__":
    unittest.main()

import json
import math
import unittest


class HLEventTests(unittest.TestCase):
    def test_finalized_events_append_once_with_common_fields(self):
        from agentbench_frame.hl.events import HLEventWriter, read_events

        with self.subTest("append-only"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                path = f"{directory}/events.jsonl"
                writer = HLEventWriter(path, run_id="run-1")
                first = writer.write(
                    "run_started",
                    game="29_rollman",
                    iteration_config={},
                )
                second = writer.write(
                    "act_completed",
                    act_id="act-0001",
                    iteration_id="iter-0001",
                    status="completed",
                    prompt_tokens=123,
                    completion_tokens=None,
                )

                records = read_events(path)
                self.assertEqual([record["event_id"] for record in records], [first, second])
                self.assertEqual([record["event_type"] for record in records], ["run_started", "act_completed"])
                self.assertEqual(records[0]["schema_version"], "1.0")
                self.assertEqual(records[0]["run_id"], "run-1")
                self.assertTrue(records[0]["created_at"].endswith("Z"))
                self.assertIsNone(records[1]["completion_tokens"])

    def test_non_finite_numbers_and_secrets_never_enter_fact_log(self):
        from agentbench_frame.hl.events import HLEventWriter
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            writer = HLEventWriter(f"{directory}/events.jsonl", run_id="run-1")
            for payload in (
                {"score": math.nan},
                {"score": math.inf},
                {"api_key": "not-even-a-real-key"},
                {"message": "Authorization: Bearer sk-secret-material"},
            ):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    writer.write("act_completed", **payload)

    def test_duplicate_event_id_is_rejected_without_rewriting_prior_line(self):
        from agentbench_frame.hl.events import HLEventWriter, read_events
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/events.jsonl"
            writer = HLEventWriter(path, run_id="run-1")
            writer.write(
                "run_started",
                event_id="fixed-id",
                iteration_config={},
            )
            with self.assertRaises(ValueError):
                writer.write("act_completed", event_id="fixed-id")
            self.assertEqual(len(read_events(path)), 1)

    def test_unknown_event_types_are_read_and_reported(self):
        from agentbench_frame.hl.events import read_events
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/events.jsonl"
            record = {
                "schema_version": "1.0",
                "event_id": "future-1",
                "event_type": "future_event",
                "run_id": "run-1",
                "created_at": "2026-07-31T00:00:00Z",
            }
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            warnings = []
            records = read_events(path, warn=warnings.append)

            self.assertEqual(records, [record])
            self.assertEqual(warnings, ["unknown event type: future_event"])

    def test_reader_rejects_wrong_schema_and_duplicate_ids(self):
        from agentbench_frame.hl.events import read_events
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/events.jsonl"
            record = {
                "schema_version": "0.9",
                "event_id": "same",
                "event_type": "future_event",
                "run_id": "run-1",
                "created_at": "2026-07-31T00:00:00Z",
            }
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                read_events(path)

            record["schema_version"] = "1.0"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate event_id"):
                read_events(path)

    def test_curriculum_events_reject_unknown_fields(self):
        from agentbench_frame.hl.events import HLEventWriter
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            writer = HLEventWriter(
                f"{directory}/events.jsonl",
                run_id="run-curriculum",
            )
            writer.write(
                "curriculum_started",
                version_id="v000000",
                active_target="rank15",
                active_target_rank=15,
                locked_opponents=["rank01"],
                required_human_opponents=16,
                stage_origin_version_id="v000000",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                writer.write(
                    "curriculum_stagnated",
                    version_id="v000004",
                    active_target="rank15",
                    stage_best_version_id="v000000",
                    stage_best_score=0.0,
                    stagnation_count=4,
                    invented=True,
                )

    def test_measurement_failure_is_a_strict_finalized_event(self):
        from agentbench_frame.hl.events import HLEventWriter, read_events
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/events.jsonl"
            writer = HLEventWriter(path, run_id="run-curriculum")
            writer.write(
                "measurement_failed",
                version_id="v000001",
                parent_version_id="v000000",
                error_type="TypeError",
                error_message="candidate policy failed on a reference state",
            )

            event = read_events(path)[0]
            self.assertEqual(event["event_type"], "measurement_failed")
            self.assertEqual(event["version_id"], "v000001")
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                writer.write(
                    "measurement_failed",
                    version_id="v000002",
                    parent_version_id="v000000",
                    error_type="TypeError",
                    error_message="invalid policy",
                    traceback="must not enter the fact log",
                )

    def test_experience_rebuild_records_the_rejected_candidate(self):
        from agentbench_frame.hl.events import HLEventWriter, read_events
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/events.jsonl"
            writer = HLEventWriter(path, run_id="run-curriculum")
            writer.write(
                "experience_rebuilt",
                rejected_version_id="v000001",
                accepted_updates=0,
                experience_path=f"{directory}/experience/SKILL.md",
            )

            event = read_events(path)[0]
            self.assertEqual(event["event_type"], "experience_rebuilt")
            self.assertEqual(event["accepted_updates"], 0)


if __name__ == "__main__":
    unittest.main()

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
                first = writer.write("run_started", game="29_rollman")
                second = writer.write(
                    "act_completed",
                    act_id="act-0001",
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
            writer.write("run_started", event_id="fixed-id")
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


if __name__ == "__main__":
    unittest.main()

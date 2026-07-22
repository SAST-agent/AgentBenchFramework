import tempfile
import unittest
from pathlib import Path


class SnapshotContractTests(unittest.TestCase):
    def test_workspace_manifest_hashes_files_and_reports_changed_paths(self):
        from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one")
            snapshotter = LocalWorkspaceSnapshotter()
            before = snapshotter.capture(root)
            (root / "a.txt").write_text("two")
            (root / "b.txt").write_text("new")
            after = snapshotter.capture(root, previous=before)
            unchanged = snapshotter.capture(root, previous=after)

            self.assertNotEqual(before.content_hash, after.content_hash)
            self.assertEqual(after.changed_files, ["a.txt", "b.txt"])
            self.assertEqual(unchanged.changed_files, [])
            self.assertEqual(after.files["a.txt"], snapshotter.file_hash(root / "a.txt"))


class ControllerContractTests(unittest.TestCase):
    def test_controller_connects_provider_usage_and_workspace_version(self):
        from agentbench_frame.tracking.controller import CodingAgentController
        from agentbench_frame.tracking.budget import BudgetLedger
        from agentbench_frame.tracking.iteration import VersionedActRecorder
        from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
        from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

        class FakeProvider:
            provider_name = "fake"

            def invoke(self, context):
                Path(context["workspace_root"], "generated.txt").write_text("result")
                return ProviderInvocation(
                    status="completed",
                    usage=ProviderUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
                    tool_call_count=2,
                    elapsed_time_s=0.25,
                    raw_output_ref="artifact://output",
                )

        with tempfile.TemporaryDirectory() as tmp:
            events = []
            ledger = BudgetLedger()
            controller = CodingAgentController(
                FakeProvider(),
                recorder=VersionedActRecorder(emit=events.append),
                snapshotter=LocalWorkspaceSnapshotter(),
                budget=ledger,
            )
            record = controller.run_act(
                {"task": "write", "workspace_root": tmp},
                workspace_root=tmp,
                version_before="v7",
            )

            self.assertEqual(record.status, "completed")
            self.assertEqual(record.provider, "fake")
            self.assertEqual(record.total_tokens, 14)
            self.assertEqual(record.changed_files, ["generated.txt"])
            self.assertIsNotNone(record.version_after)
            self.assertEqual(record.version_after, "v8")
            self.assertEqual(len(events), 1)
            self.assertEqual(ledger.snapshot()["learning_coding_agent_acts"], 1)
            self.assertEqual(ledger.snapshot()["learning_total_tokens"], 14)

    def test_controller_retains_failed_act_and_snapshots_readable_workspace(self):
        from agentbench_frame.tracking.controller import CodingAgentController
        from agentbench_frame.tracking.iteration import VersionedActRecorder
        from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

        class FailingProvider:
            provider_name = "fake"

            def invoke(self, _context):
                raise RuntimeError("provider crashed")

        with tempfile.TemporaryDirectory() as tmp:
            events = []
            controller = CodingAgentController(
                FailingProvider(),
                recorder=VersionedActRecorder(emit=events.append),
                snapshotter=LocalWorkspaceSnapshotter(),
            )
            record = controller.run_act({}, workspace_root=tmp, version_before="v0")

            self.assertEqual(record.status, "failed")
            self.assertIn("provider crashed", record.error)
            self.assertIsNotNone(record.version_after)
            self.assertEqual(len(events), 1)


class MeasurementBoundaryTests(unittest.TestCase):
    def test_canonical_state_id_is_stable_under_mapping_order(self):
        from agentbench_frame.eval.measurement import canonical_state_id

        first = canonical_state_id({"b": [2, 1], "a": {"x": True}})
        second = canonical_state_id({"a": {"x": True}, "b": [2, 1]})
        changed = canonical_state_id({"a": {"x": False}, "b": [2, 1]})

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_agent_exposes_explicit_distribution_hook_without_forcing_provider_details(self):
        from agentbench_frame.agent.base import BaseAgent

        class MinimalAgent(BaseAgent):
            def act(self, observation):
                return observation

        with self.assertRaises(NotImplementedError):
            MinimalAgent().get_action_distribution({}, ["a"])


if __name__ == "__main__":
    unittest.main()

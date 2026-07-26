import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


class ProviderParsingContractTests(unittest.TestCase):
    def test_codex_jsonl_parser_keeps_usage_thread_and_tool_events(self):
        from agentbench_frame.tracking.providers import parse_codex_jsonl

        payload = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "thr-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
            json.dumps({"type": "item.completed", "item": {"type": "file_change"}}),
            json.dumps({
                "type": "turn.completed",
                "usage": {"input_tokens": 12, "cached_input_tokens": 3,
                           "output_tokens": 5, "reasoning_output_tokens": 2},
            }),
        ])

        result = parse_codex_jsonl(payload)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.prompt_tokens, 12)
        self.assertEqual(result.usage.completion_tokens, 5)
        self.assertEqual(result.usage.total_tokens, 17)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.metadata["thread_id"], "thr-1")
        self.assertEqual(result.metadata["raw_event_count"], 5)

    def test_claude_stream_parser_counts_tool_use_and_reads_result_usage(self):
        from agentbench_frame.tracking.providers import parse_claude_stream_json

        payload = "\n".join([
            json.dumps({"type": "system", "subtype": "init", "session_id": "sess-1"}),
            json.dumps({"type": "assistant", "message": {
                "content": [{"type": "text", "text": "working"},
                             {"type": "tool_use", "name": "Edit", "id": "tool-1"}],
                "usage": {"input_tokens": 20, "output_tokens": 7},
            }}),
            json.dumps({"type": "result", "subtype": "success", "is_error": False,
                        "num_turns": 2, "duration_ms": 1200,
                        "usage": {"input_tokens": 20, "output_tokens": 7},
                        "result": "done"}),
        ])

        result = parse_claude_stream_json(payload)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(result.usage.completion_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 27)
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.metadata["session_id"], "sess-1")
        self.assertEqual(result.metadata["num_turns"], 2)

    def test_provider_builders_keep_workspace_and_safe_defaults(self):
        from agentbench_frame.tracking.providers import ClaudeCodeProvider, CodexProvider

        codex = CodexProvider(executable="codex")
        self.assertEqual(
            codex.build_command({"prompt": "fix", "workspace_root": "/tmp/work"}),
            ["codex", "exec", "--json", "--sandbox", "workspace-write", "fix"],
        )
        claude = ClaudeCodeProvider(executable="claude")
        command = claude.build_command({"prompt": "fix", "workspace_root": "/tmp/work"})
        self.assertEqual(command[:5], ["claude", "-p", "fix", "--output-format", "stream-json"])
        self.assertIn("--permission-mode", command)
        self.assertNotIn("--dangerously-skip-permissions", command)

    def test_controller_emits_provider_metadata_and_persists_raw_output(self):
        from agentbench_frame.tracking.controller import CodingAgentController
        from agentbench_frame.tracking.provider import ProviderInvocation
        from agentbench_frame.tracking.iteration import VersionedActRecorder

        class Provider:
            provider_name = "synthetic"

            def invoke(self, context):
                Path(context["workspace_root"], "out.txt").write_text("ok")
                return ProviderInvocation(
                    status="completed",
                    raw_output_ref=context["raw_output_path"],
                    metadata={"raw_event_count": 2, "provider_status": "success"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            events = []
            controller = CodingAgentController(
                Provider(), recorder=VersionedActRecorder(emit=events.append),
                raw_output_dir=tmp,
            )
            record = controller.run_act({"workspace_root": tmp, "prompt": "write"}, workspace_root=tmp)

            self.assertEqual(record.provider_metadata["raw_event_count"], 2)
            self.assertTrue(record.raw_output_ref.endswith(".jsonl"))
            self.assertEqual(events[0]["provider_metadata"]["provider_status"], "success")

    def test_codex_provider_captures_process_stream_and_return_status(self):
        from agentbench_frame.tracking.providers import CodexProvider

        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "fake-codex"
            executable.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"fake\"}'\n"
                "printf '%s\\n' '{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":3}}'\n"
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            raw = Path(tmp) / "raw.jsonl"

            result = CodexProvider(executable=str(executable)).invoke({
                "prompt": "do it", "workspace_root": tmp, "raw_output_path": str(raw),
            })

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.usage.total_tokens, 5)
            self.assertEqual(result.raw_output_ref, str(raw))
            self.assertIn("turn.completed", raw.read_text())

    def test_codex_provider_routes_large_prompt_through_stdin(self):
        from agentbench_frame.tracking.providers import CodexProvider

        provider = CodexProvider(executable="codex")
        command = provider.build_command({"prompt": "x" * 200_000})
        self.assertEqual(command[-1], "-")


if __name__ == "__main__":
    unittest.main()

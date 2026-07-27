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
        """Codex provider invokes a real subprocess and parses its JSONL stream.

        Cross-platform real subprocess: instead of a ``#!/bin/sh`` script (which
        the Windows loader rejects with ``WinError 193``), this test runs an
        argv form ``[sys.executable, path_to_fake_codex.py]``. The fake codex
        is a real Python script that emits the same two JSONL events. The
        provider's ``executable`` parameter accepts either a single string
        (legacy) or a pre-split argv list — this uses argv to exercise the
        cross-platform path.
        """
        from agentbench_frame.tracking.providers import CodexProvider

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_codex.py"
            fake.write_text(
                # Emit the two codex-JSONL events every real Codex CLI prints:
                #   1) thread.started
                #   2) turn.completed with usage (token count)
                "import json, sys; "
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fake'})); "
                "print(json.dumps({'type': 'turn.completed', "
                "'usage': {'input_tokens': 2, 'output_tokens': 3}}))",
                encoding="utf-8",
            )
            raw = Path(tmp) / "raw.jsonl"

            # argv pre-split guaranteed cross-platform;[sys.executable,
            # script.py] runs a real process on Windows AND POSIX.
            argv = [sys.executable, str(fake)]
            result = CodexProvider(executable=argv, sandbox="workspace-write").invoke({
                "prompt": "do it", "workspace_root": tmp, "raw_output_path": str(raw),
            })

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.usage.total_tokens, 5)
            self.assertEqual(result.usage.prompt_tokens, 2)
            self.assertEqual(result.usage.completion_tokens, 3)
            self.assertEqual(result.raw_output_ref, str(raw))
            self.assertIn("turn.completed", raw.read_text(encoding="utf-8"))
            # provider should record return_code 0 (the fake script exits cleanly)
            self.assertEqual(result.metadata.get("return_code"), 0)

    def test_codex_provider_accepts_legacy_string_executable(self):
        """Backward compatibility: a plain string ``executable`` continues to
        be split by shlex and used as the argv[0:] prefix. We exercise this
        without invoking a shell: route through ``sys.executable`` so the path
        round-trips even when it contains spaces (Windows Python install path).
        """
        from agentbench_frame.tracking.providers import CodexProvider

        p = CodexProvider(executable="codex")
        # ``executable`` may be stored as either a string or a list; both
        # backward-compatible forms are accepted. The contract: build_command
        # prepends whatever we passed.
        cmd = p.build_command({"prompt": "hi", "workspace_root": "/nowhere"})
        # first element is the literal "codex" string, then the subcommand
        # tokens the Codex CLI accepts; the prompt is appended last.
        self.assertEqual(cmd[0], "codex")
        self.assertEqual(cmd[1:5], ["exec", "--json", "--sandbox", "workspace-write"])
        self.assertEqual(cmd[-1], "hi")


if __name__ == "__main__":
    unittest.main()

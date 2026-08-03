import json
import os
import stat
import subprocess
from pathlib import Path

import pytest


def _provider_config(**overrides):
    from agentbench_frame.hl.config import ProviderConfig

    values = {
        "model": "gpt-5.5",
        "review_model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "env_key": "AGENTBENCH_API_KEY",
        "base_url": "https://lab.example/ai-platform/sub2api",
        "context_mode": "resumable",
    }
    values.update(overrides)
    return ProviderConfig(**values)


def test_first_and_resumed_commands_use_official_exec_shapes(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    provider = CodexSessionProvider(_provider_config(), run_root=tmp_path)
    first = provider.build_command("first task", tmp_path / "candidate", session_id=None)
    resumed = provider.build_command("next task", tmp_path / "candidate", session_id="thread-123")

    assert first[:4] == ["codex", "exec", "--json", "--sandbox"]
    assert "workspace-write" in first
    assert ["-C", str(tmp_path / "candidate")] == first[
        first.index("-C") : first.index("-C") + 2
    ]
    assert first[-1] == "first task"
    assert resumed[:4] == ["codex", "exec", "resume", "--json"]
    assert resumed[-2:] == ["thread-123", "next task"]
    assert "--ephemeral" not in first
    assert "--ephemeral" not in resumed


def test_structured_planner_command_uses_schema_output_and_phase_reasoning(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    provider = CodexSessionProvider(_provider_config(), run_root=tmp_path)
    schema = tmp_path / "planner.schema.json"
    output = tmp_path / "planner.json"

    command = provider.build_command(
        "produce four briefs",
        tmp_path / "candidate",
        session_id=None,
        output_schema_path=schema,
        output_last_message_path=output,
        reasoning_effort="high",
        sandbox_mode="read-only",
    )

    assert command[:4] == ["codex", "exec", "--json", "--sandbox"]
    assert command[4] == "read-only"
    assert command[command.index("--output-schema") + 1] == str(schema)
    assert command[command.index("--output-last-message") + 1] == str(output)
    assert "model_reasoning_effort=\"high\"" in command


def test_provider_config_is_secret_free_and_excludes_key_from_agent_shells(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    provider = CodexSessionProvider(_provider_config(), run_root=tmp_path)
    config_text = provider.config_path.read_text(encoding="utf-8")

    assert "agentbench_proxy" in config_text
    assert 'model = "gpt-5.5"' in config_text
    assert 'model_reasoning_effort = "xhigh"' in config_text
    assert "https://lab.example/ai-platform/sub2api" in config_text
    assert "AGENTBENCH_API_KEY" in config_text
    assert "CODEX_API_KEY" in config_text
    assert "sk-" not in config_text


def test_provider_config_renders_native_rollout_budget_exactly(tmp_path):
    from agentbench_frame.hl.config import HLRunConfig
    from agentbench_frame.hl.provider import CodexSessionProvider

    run = HLRunConfig.from_mapping(
        {
            "game": "29_rollman",
            "provider": {
                "kind": "codex",
                "model": "gpt-5.5",
                "rollout_budget": {
                    "enabled": True,
                    "limit_tokens": 70000,
                    "reminder_at_remaining_tokens": [20000, 10000, 5000],
                    "sampling_token_weight": 1.0,
                    "prefill_token_weight": 1.0,
                },
            },
        }
    )

    provider = CodexSessionProvider(run.provider, run_root=tmp_path)
    config_text = provider.config_path.read_text(encoding="utf-8")

    assert "[features.rollout_budget]" in config_text
    assert "enabled = true" in config_text
    assert "limit_tokens = 70000" in config_text
    assert "reminder_at_remaining_tokens = [20000, 10000, 5000]" in config_text
    assert "sampling_token_weight = 1.0" in config_text
    assert "prefill_token_weight = 1.0" in config_text


def test_preflight_verifies_exact_cli_and_generated_feature_config(tmp_path):
    from agentbench_frame.hl.config import HLRunConfig
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "preflight-codex"
    executable.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' 'codex-cli 0.146.0-alpha.9.2'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"features\" ] && [ \"$2\" = \"list\" ]; then\n"
        "  if grep -q '^enabled = true$' \"$CODEX_HOME/config.toml\"; then\n"
        "    printf '%s\\n' 'rollout_budget under development true'\n"
        "  else\n"
        "    printf '%s\\n' 'rollout_budget under development false'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exit 9\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    run = HLRunConfig.from_mapping(
        {
            "game": "29_rollman",
            "provider": {
                "kind": "codex",
                "executable": str(executable),
                "expected_cli_version": "codex-cli 0.146.0-alpha.9.2",
                "rollout_budget": {
                    "enabled": True,
                    "limit_tokens": 70000,
                },
            },
        }
    )
    provider = CodexSessionProvider(
        run.provider,
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
    )

    facts = provider.preflight()

    assert facts["cli_version"] == "codex-cli 0.146.0-alpha.9.2"
    assert facts["rollout_budget_enabled"] is True
    persisted = json.loads(
        (tmp_path / "run" / "provider-preflight.json").read_text(encoding="utf-8")
    )
    assert persisted == facts
    assert "sk-runtime-only" not in json.dumps(persisted)


def test_preflight_rejects_cli_version_drift(tmp_path):
    from agentbench_frame.hl.config import HLRunConfig
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "wrong-version-codex"
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'codex-cli 0.999.0'\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    run = HLRunConfig.from_mapping(
        {
            "game": "29_rollman",
            "provider": {
                "kind": "codex",
                "executable": str(executable),
                "expected_cli_version": "codex-cli 0.146.0-alpha.9.2",
            },
        }
    )
    provider = CodexSessionProvider(
        run.provider,
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
    )

    with pytest.raises(RuntimeError, match="version mismatch"):
        provider.preflight()


def test_provider_classifies_native_budget_exhaustion_and_weighted_usage(tmp_path):
    from agentbench_frame.hl.config import HLRunConfig
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "budget-codex"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-budget\"}'\n"
        "printf '%s\\n' '{\"type\":\"item.completed\",\"item\":{\"type\":\"file_change\"}}'\n"
        "printf '%s\\n' '{\"type\":\"turn.failed\",\"error\":{\"message\":\"SessionBudgetExceeded\"},\"usage\":{\"input_tokens\":20,\"cached_input_tokens\":15,\"output_tokens\":7}}'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    run = HLRunConfig.from_mapping(
        {
            "game": "29_rollman",
            "provider": {
                "kind": "codex",
                "executable": str(executable),
                "transport_retry_attempts": 0,
                "rollout_budget": {
                    "enabled": True,
                    "limit_tokens": 70000,
                    "sampling_token_weight": 1.0,
                    "prefill_token_weight": 1.0,
                },
            },
        }
    )
    provider = CodexSessionProvider(
        run.provider,
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()

    result = provider.invoke(
        prompt="bounded edit",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "budget.jsonl",
    )

    assert result.status == "failed"
    assert result.metadata["rollout_budget_exhausted"] is True
    assert result.metadata["termination_reason"] == "rollout_budget_exhausted"
    assert result.metadata["weighted_tokens"] == 12.0
    assert result.metadata["rollout_budget_limit_tokens"] == 70000
    assert result.usage.prompt_tokens == 20
    assert result.usage.cached_input_tokens == 15
    assert result.usage.completion_tokens == 7


def test_provider_classifies_real_codex_shared_rollout_budget_error(tmp_path):
    from agentbench_frame.hl.config import HLRunConfig
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "shared-budget-codex"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-shared-budget\"}'\n"
        "printf '%s\\n' '{\"type\":\"item.completed\",\"item\":{\"type\":\"file_change\"}}'\n"
        "printf '%s\\n' '{\"type\":\"error\",\"message\":\"shared rollout token budget exhausted\"}'\n"
        "printf '%s\\n' '{\"type\":\"turn.failed\",\"error\":{\"message\":\"shared rollout token budget exhausted\"}}'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    run = HLRunConfig.from_mapping(
        {
            "game": "29_rollman",
            "provider": {
                "kind": "codex",
                "executable": str(executable),
                "transport_retry_attempts": 0,
                "rollout_budget": {
                    "enabled": True,
                    "limit_tokens": 70000,
                },
            },
        }
    )
    provider = CodexSessionProvider(
        run.provider,
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()

    result = provider.invoke(
        prompt="bounded edit",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "shared-budget.jsonl",
    )

    assert result.status == "failed"
    assert result.metadata["rollout_budget_exhausted"] is True
    assert result.metadata["termination_reason"] == "rollout_budget_exhausted"
    assert result.metadata["weighted_tokens"] is None


def test_runtime_key_is_scoped_to_codex_process_and_never_logged(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    provider = CodexSessionProvider(
        _provider_config(),
        run_root=tmp_path,
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "UNRELATED_SECRET": "must-not-pass",
        },
    )
    child_env = provider.build_environment()

    assert child_env["CODEX_API_KEY"] == "sk-runtime-only"
    assert "AGENTBENCH_API_KEY" not in child_env
    assert "UNRELATED_SECRET" not in child_env
    assert provider.redacted_environment()["CODEX_API_KEY"] == "<redacted>"


def test_fake_codex_invocation_retains_jsonl_session_and_exact_usage(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "fake-codex"
    executable.write_text(
        "#!/bin/sh\n"
        "test \"$CODEX_API_KEY\" = \"sk-runtime-only\" || exit 9\n"
        "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-fake\"}'\n"
        "printf '%s\\n' '{\"type\":\"item.completed\",\"item\":{\"type\":\"file_change\"}}'\n"
        "printf '%s\\n' '{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":20,\"cached_input_tokens\":15,\"output_tokens\":7,\"reasoning_output_tokens\":3}}'\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    config = _provider_config(executable=str(executable))
    provider = CodexSessionProvider(
        config,
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
        idle_timeout_s=1,
    )
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-1.jsonl"

    result = provider.invoke(
        prompt="make one change",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert result.status == "completed"
    assert result.metadata["thread_id"] == "thread-fake"
    assert result.usage.prompt_tokens == 20
    assert result.usage.cached_input_tokens == 15
    assert result.usage.completion_tokens == 7
    assert result.usage.reasoning_output_tokens == 3
    assert result.tool_call_count == 1
    assert raw.read_text(encoding="utf-8").count("\n") == 3
    assert "sk-runtime-only" not in json.dumps(result.metadata)


def test_provider_timeout_persists_partial_jsonl(tmp_path, monkeypatch):
    from agentbench_frame.hl.provider import CodexSessionProvider

    partial = (
        '{"type":"thread.started","thread_id":"thread-partial"}\n'
        '{"type":"item.completed","item":{"type":"file_change"}}\n'
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=3,
            output=partial,
            stderr="hard phase deadline",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-timeout.jsonl"
    provider = CodexSessionProvider(
        _provider_config(),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
        timeout_s=3,
    )

    result = provider.invoke(
        prompt="bounded task",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert result.status == "timeout"
    assert result.raw_output_ref == str(raw)
    assert raw.read_text(encoding="utf-8") == partial
    assert result.metadata["thread_id"] == "thread-partial"


def test_provider_idle_timeout_persists_stream_and_identifies_deadline(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "idle-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "print('{\"type\":\"thread.started\",\"thread_id\":\"thread-idle\"}', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-idle.jsonl"
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
        timeout_s=3,
        idle_timeout_s=1.0,
    )

    result = provider.invoke(
        prompt="bounded idle task",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert result.status == "timeout"
    assert result.metadata["timeout_kind"] == "idle"
    assert result.metadata["thread_id"] == "thread-idle"
    assert "thread-idle" in raw.read_text(encoding="utf-8")
    assert result.elapsed_time_s < 2.0


@pytest.mark.parametrize("idle_timeout", [2, None])
def test_coding_provider_stops_before_edit_after_hard_tool_grace_limit(
    tmp_path, idle_timeout
):
    """A coding act that only browses cannot consume its entire wall-clock act."""

    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "browse-forever-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, time\n"
        "print(json.dumps({'type':'thread.started','thread_id':'thread-tools'}), flush=True)\n"
        "for index in range(30):\n"
        " print(json.dumps({'type':'item.started','item':{'id':str(index),'type':'command_execution','command':'true'}}), flush=True)\n"
        " print(json.dumps({'type':'item.completed','item':{'id':str(index),'type':'command_execution','command':'true','status':'completed'}}), flush=True)\n"
        " time.sleep(0.05)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-tools.jsonl"
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
        timeout_s=5,
        idle_timeout_s=idle_timeout,
    )

    result = provider.invoke(
        prompt="# HL iteration act-b00 — 候选 1/4\nwrite a candidate",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert result.status == "failed"
    assert "pre-edit tool call limit 14" in str(result.error)
    assert result.elapsed_time_s < 2.0
    records = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
    ]
    started = sum(record.get("type") == "item.started" for record in records)
    assert started <= 16 if idle_timeout is not None else started == 30
    assert any(
        "provider_tool_limit_exceeded" in str(record.get("message"))
        for record in records
    )


def test_rollman_v2_candidate_stops_after_six_pre_edit_tool_calls(tmp_path):
    """Catch packet acts that spend the saved context budget browsing again."""
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "browse-rollman-v2-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'type':'thread.started','thread_id':'thread-v2'}))\n"
        "for index in range(8):\n"
        " print(json.dumps({'type':'item.started','item':{'id':str(index),'type':'command_execution','command':'true'}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
        timeout_s=5,
        idle_timeout_s=None,
    )

    result = provider.invoke(
        prompt=(
            "# HL iteration act-b00 — 候选 1/4\n"
            "candidate-context-contract: rollman-v2\n"
        ),
        workspace=workspace,
        raw_output_path=tmp_path / "run/provider/act-v2.jsonl",
    )

    assert result.status == "failed"
    assert "pre-edit tool call limit 6" in str(result.error)


def test_provider_retries_zero_usage_transport_failure_and_preserves_attempt(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl.provider import CodexSessionProvider

    failed = (
        '{"type":"thread.started","thread_id":"thread-disconnected"}\n'
        '{"type":"error","message":"stream disconnected before completion: error sending request"}\n'
        '{"type":"turn.failed","error":{"message":"stream disconnected before completion"}}\n'
    )
    completed = (
        '{"type":"thread.started","thread_id":"thread-retried"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":7}}\n'
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 1, failed, "")
        return subprocess.CompletedProcess(args[0], 0, completed, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("agentbench_frame.hl.provider.time.sleep", lambda _: None)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-retry.jsonl"
    provider = CodexSessionProvider(
        _provider_config(
            transport_retry_attempts=2,
            transport_retry_backoff_seconds=1.0,
        ),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
    )

    result = provider.invoke(
        prompt="retry transport only",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert len(calls) == 2
    assert result.status == "completed"
    assert result.metadata["transport_retry_count"] == 1
    attempt = raw.with_name("act-retry.attempt-1.jsonl")
    assert attempt.read_text(encoding="utf-8") == failed
    assert raw.read_text(encoding="utf-8") == completed


def test_provider_does_not_retry_failed_act_after_token_usage(tmp_path, monkeypatch):
    from agentbench_frame.hl.provider import CodexSessionProvider

    output = (
        '{"type":"thread.started","thread_id":"thread-billed"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":7}}\n'
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args[0], 1, output, "provider exited")

    monkeypatch.setattr(subprocess, "run", run)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    provider = CodexSessionProvider(
        _provider_config(transport_retry_attempts=2),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
    )

    result = provider.invoke(
        prompt="do not duplicate billed work",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "act-billed.jsonl",
    )

    assert len(calls) == 1
    assert result.status == "failed"
    assert result.usage.total_tokens == 27


def test_provider_cools_down_next_act_after_rate_limit(tmp_path, monkeypatch):
    """A 429 must delay the next sibling instead of burning it immediately."""

    from agentbench_frame.hl.provider import CodexSessionProvider

    limited = (
        '{"type":"thread.started","thread_id":"thread-limited"}\n'
        '{"type":"error","message":"exceeded retry limit, last status: 429 Too Many Requests"}\n'
        '{"type":"turn.failed","error":{"message":"429 Too Many Requests"}}\n'
    )
    completed = (
        '{"type":"thread.started","thread_id":"thread-after-cooldown"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":7}}\n'
    )
    outputs = iter((limited, completed))

    def run(command, **_kwargs):
        output = next(outputs)
        return subprocess.CompletedProcess(
            command,
            1 if output == limited else 0,
            output,
            "",
        )

    class Clock:
        def __init__(self):
            self.now = 100.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("agentbench_frame.hl.provider.time.monotonic", clock.monotonic)
    monkeypatch.setattr("agentbench_frame.hl.provider.time.sleep", clock.sleep)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    provider = CodexSessionProvider(
        _provider_config(
            transport_retry_attempts=0,
            rate_limit_cooldown_seconds=30.0,
        ),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
    )

    first = provider.invoke(
        prompt="first sibling",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "act-one.jsonl",
    )
    second = provider.invoke(
        prompt="next sibling",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "act-two.jsonl",
    )

    assert first.status == "failed"
    assert second.status == "completed"
    assert clock.sleeps == [30.0]


def test_provider_retries_zero_usage_rate_limit_after_full_cooldown(
    tmp_path, monkeypatch
):
    """A zero-work 429 should recover inside the same act after one cooldown."""

    from agentbench_frame.hl.provider import CodexSessionProvider

    limited = (
        '{"type":"thread.started","thread_id":"thread-limited"}\n'
        '{"type":"error","message":"exceeded retry limit, last status: 429 Too Many Requests"}\n'
        '{"type":"turn.failed","error":{"message":"429 Too Many Requests"}}\n'
    )
    completed = (
        '{"type":"thread.started","thread_id":"thread-recovered"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":7}}\n'
    )
    outputs = iter((limited, completed))

    def run(command, **_kwargs):
        output = next(outputs)
        return subprocess.CompletedProcess(
            command,
            1 if output == limited else 0,
            output,
            "",
        )

    class Clock:
        def __init__(self):
            self.now = 100.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("agentbench_frame.hl.provider.time.monotonic", clock.monotonic)
    monkeypatch.setattr("agentbench_frame.hl.provider.time.sleep", clock.sleep)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    raw = tmp_path / "run" / "provider" / "act-rate-retry.jsonl"
    provider = CodexSessionProvider(
        _provider_config(
            transport_retry_attempts=1,
            transport_retry_backoff_seconds=2.0,
            rate_limit_cooldown_seconds=30.0,
        ),
        run_root=tmp_path / "run",
        environ={"AGENTBENCH_API_KEY": "sk-runtime-only"},
    )

    result = provider.invoke(
        prompt="recover this planner act",
        workspace=workspace,
        raw_output_path=raw,
    )

    assert result.status == "completed"
    assert result.metadata["transport_retry_count"] == 1
    assert clock.sleeps == [30.0]
    assert raw.with_name("act-rate-retry.attempt-1.jsonl").is_file()


def test_provider_rejects_tool_reads_from_another_run(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "fake-codex"
    other_run = (
        tmp_path
        / ".agentbench"
        / "29_rollman"
        / "runs"
        / "old-run"
        / "events.jsonl"
    )
    command = f"sed -n '1,20p' {other_run}"
    records = [
        {"type": "thread.started", "thread_id": "thread-tainted"},
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": command,
                "status": "completed",
                "exit_code": 0,
            },
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 20, "output_tokens": 7},
        },
    ]
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' {json.dumps(json.dumps(records[0]))}\n"
        f"printf '%s\\n' {json.dumps(json.dumps(records[1]))}\n"
        f"printf '%s\\n' {json.dumps(json.dumps(records[2]))}\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    run_root = (
        tmp_path
        / ".agentbench"
        / "29_rollman"
        / "runs"
        / "active-run"
    )
    workspace = (
        tmp_path / ".agentbench" / "29_rollman" / "candidate-curriculum"
    )
    workspace.mkdir(parents=True)
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=run_root,
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
    )

    result = provider.invoke(
        prompt="make one isolated change",
        workspace=workspace,
        raw_output_path=run_root / "provider" / "act-1.jsonl",
    )

    assert result.status == "failed"
    assert "access policy violation" in str(result.error)
    assert result.metadata["access_policy_violations"] == [
        str(other_run)
    ]
    assert (
        result.metadata["termination_reason"]
        == "provider_access_policy_violation"
    )
    assert result.usage.total_tokens == 27


def test_provider_stops_stream_immediately_after_access_policy_violation(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    run_root = tmp_path / ".agentbench" / "29_rollman" / "runs" / "active"
    workspace = tmp_path / ".agentbench" / "29_rollman" / "candidate"
    workspace.mkdir(parents=True)
    forbidden = tmp_path / ".agentbench" / "29_rollman" / "runs" / "old" / "events.jsonl"
    executable = tmp_path / "tainted-stream-codex"
    record = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": f"cat {forbidden}",
            "status": "completed",
        },
    }
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, time\n"
        f"print(json.dumps({record!r}), flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=run_root,
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
        timeout_s=5,
        idle_timeout_s=2,
    )

    result = provider.invoke(
        prompt="# Rollman scoped repair act-repair\nrepair",
        workspace=workspace,
        raw_output_path=run_root / "provider" / "act-tainted.jsonl",
    )

    assert result.status == "failed"
    assert result.elapsed_time_s < 2.0
    assert result.metadata["termination_reason"] == "provider_access_policy_violation"
    assert str(forbidden) in result.metadata["access_policy_violations"]


def test_provider_stops_coding_stream_at_raw_output_limit(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "giant-output-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, time\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'command_execution','command':'true','status':'completed','aggregated_output':'x'*600000}}), flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=tmp_path / "run",
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
        },
        timeout_s=5,
        idle_timeout_s=2,
    )

    result = provider.invoke(
        prompt="# HL iteration act-b00 — 候选 1/4\nwrite",
        workspace=workspace,
        raw_output_path=tmp_path / "run" / "provider" / "act-large.jsonl",
    )

    assert result.status == "failed"
    assert result.elapsed_time_s < 2.0
    assert result.metadata["termination_reason"] == "provider_output_limit_exceeded"


def test_provider_allows_declared_run_artifacts_and_candidate_workspace(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    executable = tmp_path / "fake-codex"
    run_root = (
        tmp_path
        / ".agentbench"
        / "29_rollman"
        / "runs"
        / "active-run"
    )
    workspace = (
        tmp_path / ".agentbench" / "29_rollman" / "candidate-curriculum"
    )
    workspace.mkdir(parents=True)
    allowed_paths = [
        workspace / "ai.py",
        run_root / "context" / "context-manifest.json",
        run_root / "experience" / "SKILL.md",
        run_root / "matches" / "v000000" / "learning" / "replay.jsonl",
        run_root / "research_state.json",
        run_root / "proposals" / "iter-000001" / "reducer_input.json",
        run_root / "distillation" / "rank15" / "ghost-hash.json",
    ]
    records = [
        {"type": "thread.started", "thread_id": "thread-clean"},
        *[
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": f"sed -n '1,20p' {path}",
                    "status": "completed",
                    "exit_code": 0,
                },
            }
            for path in allowed_paths
        ],
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 20, "output_tokens": 7},
        },
    ]
    executable.write_text(
        "#!/bin/sh\n"
        + "".join(
            f"printf '%s\\n' {json.dumps(json.dumps(record))}\n"
            for record in records
        ),
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    provider = CodexSessionProvider(
        _provider_config(executable=str(executable)),
        run_root=run_root,
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
    )

    result = provider.invoke(
        prompt="make one isolated change",
        workspace=workspace,
        raw_output_path=run_root / "provider" / "act-1.jsonl",
    )

    assert result.status == "completed"
    assert result.metadata["access_policy_violations"] == []


def test_provider_recovers_completed_persisted_output_without_new_process(tmp_path):
    from agentbench_frame.hl.provider import CodexSessionProvider

    run_root = tmp_path / ".agentbench" / "runs" / "active-run"
    workspace = tmp_path / ".agentbench" / "candidate"
    workspace.mkdir(parents=True)
    raw = run_root / "provider" / "act-000001-planner.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "thread.started", "thread_id": "thread-recovered"}
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": f"sed -n '1,20p' {run_root / 'research_state.json'}",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 20, "output_tokens": 7},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    provider = CodexSessionProvider(
        _provider_config(),
        run_root=run_root,
        environ={
            "AGENTBENCH_API_KEY": "sk-runtime-only",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
    )

    recovered = provider.recover_completed_output(
        raw_output_path=raw,
        workspace=workspace,
    )

    assert recovered.status == "completed"
    assert recovered.metadata["thread_id"] == "thread-recovered"
    assert recovered.metadata["recovered_from_persisted_output"] is True
    assert recovered.usage.total_tokens == 27

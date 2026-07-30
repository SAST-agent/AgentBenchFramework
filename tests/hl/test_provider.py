import json
import os
import stat
from pathlib import Path


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


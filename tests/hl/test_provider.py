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
    assert result.usage.total_tokens == 27


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

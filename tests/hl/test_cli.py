import json
import dataclasses
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/hl/29_rollman.yaml"


def _last_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_hl_validate_reports_open_ended_k1_rollback_defaults(capsys):
    from agentbench_frame.hl.cli import main

    assert main(["validate", "--config", str(CONFIG)]) == 0
    result = _last_json(capsys)

    assert result["valid"] is True
    assert result["game"] == "29_rollman"
    assert result["max_acts"] is None
    assert result["candidates_per_act"] == 1
    assert result["rollback_enabled"] is True
    assert result["human_opponents"] == 16


def test_hl_audit_verifies_frozen_backend_and_seed_adapter(capsys):
    from agentbench_frame.hl.cli import main

    assert main(["audit", "--config", str(CONFIG)]) == 0
    result = _last_json(capsys)

    assert result["all_core_files_match"] is True
    assert result["adapter_seed_injection_required"] is True


def test_hl_dry_run_needs_no_api_key_and_creates_no_provider_call(
    tmp_path, capsys, monkeypatch
):
    from agentbench_frame.hl.cli import main

    monkeypatch.delenv("AGENTBENCH_API_KEY", raising=False)
    assert (
        main(
            [
                "run",
                "--dry-run",
                "--config",
                str(CONFIG),
                "--run-dir",
                str(tmp_path / "dry-run"),
                "--workspace",
                str(tmp_path / "candidate"),
            ]
        )
        == 0
    )
    result = _last_json(capsys)

    assert result["dry_run"] is True
    assert result["would_call_model"] is False
    assert result["context_mode"] == "resumable"
    assert (tmp_path / "candidate" / "ai.py").is_file()
    assert not (tmp_path / "dry-run" / "provider").exists()


def test_real_run_fails_before_state_change_when_api_key_is_missing(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl import cli

    monkeypatch.delenv("AGENTBENCH_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_provider_environment", lambda _config: None)
    code = cli.main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--run-dir",
            str(tmp_path / "run"),
            "--workspace",
            str(tmp_path / "candidate"),
            "--acts",
            "1",
        ]
    )

    assert code == 2
    assert not (tmp_path / "run").exists()


def test_dotenv_credential_is_scoped_to_provider_environment(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl.cli import _provider_environment
    from agentbench_frame.hl.local_config import LocalHLConfig

    monkeypatch.delenv("AGENTBENCH_API_KEY", raising=False)
    source = tmp_path / "configs" / "hl" / "29_rollman.yaml"
    source.parent.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "AGENTBENCH_API_KEY=test-runtime-value\n",
        encoding="utf-8",
    )
    config = dataclasses.replace(
        LocalHLConfig.load(CONFIG),
        source_path=source,
    )

    environment = _provider_environment(config)

    assert environment is not None
    assert environment["AGENTBENCH_API_KEY"] == "test-runtime-value"
    assert "AGENTBENCH_API_KEY" not in __import__("os").environ

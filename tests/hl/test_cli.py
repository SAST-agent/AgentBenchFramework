import json
import dataclasses
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/hl/29_rollman.yaml"
CURRICULUM_CONFIG = ROOT / "configs/hl/29_rollman-curriculum.yaml"


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


def test_rollman_curriculum_config_matches_approved_experiment():
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CURRICULUM_CONFIG)

    assert config.run.origin.mode == "imported_version"
    assert config.run.origin.source_version == "v000001"
    assert config.run.curriculum.mode == "weakest_failed"
    assert config.run.curriculum.required_human_opponents == 16
    assert config.run.curriculum.stagnation_patience == 4
    assert config.run.iteration.candidates_per_act == 1
    assert config.run.iteration.max_acts is None
    assert config.run.evaluation.full_pool_every_iteration is True


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


def test_frozen_run_config_is_json_native_and_round_trips():
    from agentbench_frame.hl.cli import (
        _frozen_run_config,
        _normalize_frozen_run_config,
    )
    from agentbench_frame.hl.local_config import LocalHLConfig

    snapshot = _frozen_run_config(LocalHLConfig.load(CONFIG))

    assert isinstance(snapshot["run"]["evaluation"]["fixed_gate_seeds"], list)
    assert json.loads(json.dumps(snapshot)) == snapshot
    legacy_snapshot = json.loads(json.dumps(snapshot))
    del legacy_snapshot["run"]["origin"]
    del legacy_snapshot["run"]["curriculum"]
    assert _normalize_frozen_run_config(legacy_snapshot) == snapshot


def test_gate_saturated_selected_head_is_still_eligible_for_certification():
    from agentbench_frame.hl.cli import _eligible_certification_version_id
    from agentbench_frame.hl.evaluator import CandidateEvaluation

    eligible = _eligible_certification_version_id(
        lineage_head_version_id="v000001",
        evaluations_by_version={
            "v000000": CandidateEvaluation(status="complete", score=1.0),
            "v000001": CandidateEvaluation(status="complete", score=1.0),
        },
        completed_certifications={"v000000"},
        required_score=0.5,
    )

    assert eligible == "v000001"


def test_resume_accepts_zero_new_model_acts(monkeypatch):
    from agentbench_frame.hl import cli

    monkeypatch.setattr(cli, "_cmd_resume", lambda args: args.acts)

    assert (
        cli.main(
            [
                "resume",
                "--config",
                str(CONFIG),
                "--run-dir",
                "unused",
                "--acts",
                "0",
            ]
        )
        == 0
    )


def test_unbounded_loop_stops_on_provider_or_evaluation_failure():
    from types import SimpleNamespace
    from agentbench_frame.hl.cli import _iteration_stop_reason

    failed_provider = SimpleNamespace(
        selected=SimpleNamespace(
            provider=SimpleNamespace(status="failed"),
            evaluation=SimpleNamespace(status="failed"),
        )
    )
    incomplete_evaluation = SimpleNamespace(
        selected=SimpleNamespace(
            provider=SimpleNamespace(status="completed"),
            evaluation=SimpleNamespace(status="incomplete"),
        )
    )

    assert _iteration_stop_reason(failed_provider) == "provider_failed"
    assert _iteration_stop_reason(incomplete_evaluation) == "evaluation_incomplete"


def test_imported_run_accepts_zero_acts_without_provider_credential(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl import cli
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CONFIG)
    config = dataclasses.replace(
        config,
        run=dataclasses.replace(
            config.run,
            origin=cli.HLRunConfig.from_mapping(
                {
                    **config.run.to_dict(),
                    "origin": {
                        "mode": "imported_version",
                        "source_run": str(tmp_path / "source-run"),
                        "source_version": "v000001",
                    },
                    "curriculum": {
                        "mode": "weakest_failed",
                        "required_human_opponents": 16,
                    },
                }
            ).origin,
            curriculum=cli.HLRunConfig.from_mapping(
                {
                    **config.run.to_dict(),
                    "origin": {
                        "mode": "imported_version",
                        "source_run": str(tmp_path / "source-run"),
                        "source_version": "v000001",
                    },
                    "curriculum": {
                        "mode": "weakest_failed",
                        "required_human_opponents": 16,
                    },
                }
            ).curriculum,
        ),
    )
    calls = []
    monkeypatch.setattr(cli, "_load", lambda _path: config)
    monkeypatch.setattr(cli, "_provider_environment", lambda _config: None)
    monkeypatch.setattr(
        cli,
        "_run_real",
        lambda *args, **kwargs: calls.append(kwargs) or 0,
    )

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
            "0",
        ]
    )

    assert code == 0
    assert calls[0]["acts"] == 0
    assert calls[0]["provider_environment"] is None

import json
import dataclasses
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/hl/29_rollman.yaml"
CURRICULUM_CONFIG = ROOT / "configs/hl/29_rollman-curriculum.yaml"
REPAIR_V5_CONFIG = ROOT / "configs/hl/29_rollman-k4-repair-v5.yaml"


def _last_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_trace_fault_summary_exposes_bounded_redacted_candidate_error(tmp_path):
    from agentbench_frame.hl.cli import _trace_fault_summary

    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"type": "watch", "content": {"round": 0}})
        + "\n"
        + json.dumps(
            {
                "type": "ai_fault",
                "player": 0,
                "error": "RE",
                "detail": "player 0 emitted no frame",
                "stderr_tail": "Traceback: numpy failure sk-secretvalue123",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _trace_fault_summary(trace) == {
        "error": "RE",
        "detail": "player 0 emitted no frame",
        "stderr_tail": "Traceback: numpy failure [REDACTED]",
    }


def test_opponent_distillation_is_cached_by_trace_content(tmp_path, monkeypatch):
    from agentbench_frame.hl.cli import _ensure_opponent_distillation

    trace_a = tmp_path / "a.trace.jsonl"
    trace_b = tmp_path / "b.trace.jsonl"
    trace_a.write_text('{"type":"a"}\n', encoding="utf-8")
    trace_b.write_text('{"type":"b"}\n', encoding="utf-8")
    tool = tmp_path / "distill.py"
    tool.write_text("# fake\n", encoding="utf-8")
    calls = []
    output = {
        "schema_version": "1.0",
        "trace_count": 2,
        "ghost_decision_samples": 12,
        "coarse_backoff_patterns": [],
        "fine_patterns": [],
    }

    def run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args[0],
            0,
            json.dumps(output),
            "",
        )

    monkeypatch.setattr("agentbench_frame.hl.cli.subprocess.run", run)

    first = _ensure_opponent_distillation(
        traces=[trace_a, trace_b],
        distillation_tool=tool,
        output_root=tmp_path / "distillation",
    )
    second = _ensure_opponent_distillation(
        traces=[trace_a, trace_b],
        distillation_tool=tool,
        output_root=tmp_path / "distillation",
    )

    assert first == second
    assert len(calls) == 1
    distilled = json.loads(first.read_text(encoding="utf-8"))
    assert distilled["representation"] == "modal_patterns_v1"
    assert distilled["trace_count"] == 2
    assert distilled["ghost_decision_samples"] == 12
    assert distilled["coarse_backoff_patterns"] == []
    assert distilled["fine_patterns"] == []
    assert first.stat().st_size < 24 * 1024


def test_opponent_distillation_uses_inherited_research_debt(tmp_path):
    from agentbench_frame.hl.cli import _opponent_distillation_required
    from agentbench_frame.hl.research_state import ResearchState

    state_path = ResearchState.empty(max_bytes=16384).advance(
        exploration_debt=3,
    ).write(tmp_path / "research_state.json")

    assert _opponent_distillation_required(
        stagnation_count=1,
        research_state_path=state_path,
        research_state_max_bytes=16384,
    ) is True


def test_opponent_distillation_triggers_after_two_stagnant_cycles(tmp_path):
    from agentbench_frame.hl.cli import _opponent_distillation_required
    from agentbench_frame.hl.research_state import ResearchState

    state_path = ResearchState.empty(max_bytes=16384).advance(
        exploration_debt=2,
    ).write(tmp_path / "research_state.json")

    assert _opponent_distillation_required(
        stagnation_count=2,
        research_state_path=state_path,
        research_state_max_bytes=16384,
    ) is True


def test_main_curve_measurement_only_uses_linear_selected_successor():
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _measurement_candidates

    siblings = tuple(
        SimpleNamespace(
            name=f"branch-{index}",
            version=SimpleNamespace(version_id=f"v{index}"),
        )
        for index in range(4)
    )
    iteration = SimpleNamespace(
        candidates=siblings,
        selected=siblings[2],
        search_parent_version_id="v2",
    )

    assert _measurement_candidates(iteration) == (siblings[2],)


def test_main_curve_measurement_skips_rejected_sibling_when_parent_is_retained():
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _measurement_candidates

    selected_sibling = SimpleNamespace(
        version=SimpleNamespace(version_id="v2")
    )
    iteration = SimpleNamespace(
        selected=selected_sibling,
        search_parent_version_id="v1",
    )

    assert _measurement_candidates(iteration) == ()


def test_resume_does_not_remeasure_parent_retained_by_completed_proposal():
    from agentbench_frame.hl.cli import _pending_measurement_candidate

    events = [
        {
            "event_type": "version_created",
            "version_id": "v1",
            "parent_version_id": "v0",
            "evaluation_status": "complete",
        },
        {
            "event_type": "search_parent_selected",
            "iteration_id": "iter-000002",
            "version_id": "v1",
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000002",
            "parent_version_id": "v1",
            "selected_version_id": "v1",
        },
        {
            "event_type": "curriculum_gate_completed",
            "version_id": "v1",
        },
    ]

    assert _pending_measurement_candidate(events) is None


def test_resume_finalizes_no_change_proposal_without_measurement_events():
    from agentbench_frame.hl.cli import _pending_proposal_finalization

    events = [
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000002",
            "parent_version_id": "v1",
            "selected_version_id": "v1",
        }
    ]

    assert _pending_proposal_finalization(events) == (
        "iter-000002",
        "v1",
        "v1",
    )


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


def test_hl_validate_reports_staged_feedback_seed_counts(capsys, monkeypatch):
    from agentbench_frame.hl.cli import main

    monkeypatch.setenv("AGENTBENCH_SAST_ROOT", "/Users/qingle/Code/SAST")
    assert main(["validate", "--config", str(REPAIR_V5_CONFIG)]) == 0
    result = _last_json(capsys)

    assert result["quick_screen_seeds"] == 1
    assert result["finalist_seeds"] == 3


def test_rollman_curriculum_config_matches_approved_experiment():
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CURRICULUM_CONFIG)

    assert config.run.origin.mode == "imported_version"
    assert config.run.origin.source_version == "v000001"
    assert config.run.curriculum.mode == "weakest_failed"
    assert config.run.curriculum.required_human_opponents == 16
    assert config.run.curriculum.stagnation_patience == 16
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
    manifest = json.loads(
        Path(result["context_manifest"]).read_text(encoding="utf-8")
    )
    fixture = Path(manifest["files"]["smoke_fixture"]["path"])
    assert fixture.name == "rollman_smoke_fixture.py"
    assert fixture.is_file()
    assert not (tmp_path / "dry-run" / "provider").exists()


def test_framework_smoke_callback_reexecutes_fixture_and_checks_hashes(tmp_path):
    """Catch CLI wiring that trusts a provider-authored result file."""
    from agentbench_frame.games.rollman import rollman_smoke_fixture
    from agentbench_frame.hl.cli import _verify_rollman_candidate_smoke

    workspace = tmp_path / "candidate"
    control = workspace / ".agentbench"
    control.mkdir(parents=True)
    (workspace / "ai.py").write_text(
        "calls = 0\n"
        "def ai_func(state):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    prefix = 'new:' if calls == 1 else 'parent:'\n"
        "    return {'action': 1, 'memory_id': prefix}\n",
        encoding="utf-8",
    )
    scenario = control / "smoke_scenario.json"
    state = {
        "level": 1,
        "round_id": 1,
        "pacman": [2, 2],
        "ghosts": [[3, 3], [4, 4], [5, 5]],
        "board_size": 8,
    }
    scenario.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "states": [state, state],
                "activation_state_index": 0,
                "preservation_state_index": 1,
                "activation_memory_prefix": "new:",
                "preservation_forbidden_prefix": "new:",
                "preservation_memory_prefix": "parent:",
            }
        ),
        encoding="utf-8",
    )
    result_path = control / "candidate_smoke_result.json"
    result_path.write_text('{"status":"forged"}\n', encoding="utf-8")

    result = _verify_rollman_candidate_smoke(
        workspace=workspace,
        fixture_path=Path(rollman_smoke_fixture.__file__),
    )

    assert result["status"] == "complete"
    assert result["policy_sha256"] == rollman_smoke_fixture.sha256_file(
        workspace / "ai.py"
    )
    assert result["scenario_sha256"] == rollman_smoke_fixture.sha256_file(
        scenario
    )
    assert result["scenario_path"] == str(scenario.resolve())
    assert result["result_path"] == str(result_path.resolve())


def test_cli_packet_builders_keep_smoke_contract_on_candidates_only(tmp_path):
    """Catch smoke-only arguments being forwarded to the planner packet."""
    from agentbench_frame.hl.cli import (
        _write_rollman_candidate_packet,
        _write_rollman_planner_packet,
    )

    digest = tmp_path / "digest.json"
    manifest = tmp_path / "manifest.json"
    research = tmp_path / "research.json"
    experience = tmp_path / "SKILL.md"
    source = tmp_path / "candidate" / "ai.py"
    fixture = tmp_path / "context" / "rollman_smoke_fixture.py"
    source.parent.mkdir()
    fixture.parent.mkdir()
    digest.write_text('{"actions":[0,1,2,3,4]}\n', encoding="utf-8")
    manifest.write_text('{"bundle_hash":"test"}\n', encoding="utf-8")
    research.write_text('{"open_questions":[]}\n', encoding="utf-8")
    experience.write_text("experience\n", encoding="utf-8")
    source.write_text(
        "def helper(state):\n    return 0\n\n"
        "def ai_func(state):\n    return helper(state)\n",
        encoding="utf-8",
    )
    fixture.write_text("# fixture\n", encoding="utf-8")

    planner = _write_rollman_planner_packet(
        output_path=tmp_path / "planner.json",
        iteration_id="iter-test",
        parent_version_id="v000016",
        game_digest_path=digest,
        context_manifest_path=manifest,
        research_state_path=research,
        replay_evidence=[],
        previous_measurements={},
        active_target="rank15",
        candidate_source_path=source,
    )
    candidate = _write_rollman_candidate_packet(
        output_path=tmp_path / "candidate.json",
        iteration_id="iter-test",
        branch_brief={
            "branch_index": 0,
            "mechanism": "test",
            "code_symbols": ["ai_func", "helper"],
        },
        game_digest_path=digest,
        research_state_path=research,
        experience_path=experience,
        replay_evidence=[],
        previous_measurements={},
        active_target="rank15",
        locked_opponents=(),
        candidate_source_path=source,
        smoke_fixture_path=fixture,
        candidate_workspace=source.parent,
    )

    planner_value = json.loads(planner.read_text(encoding="utf-8"))
    candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
    assert "smoke_contract" not in planner_value
    assert candidate_value["smoke_contract"]["fixture_path"] == str(
        fixture.resolve()
    )


def test_imported_run_can_inherit_bounded_research_state(tmp_path):
    from agentbench_frame.hl.cli import _prepare_research_state
    from agentbench_frame.hl.config import OriginConfig
    from agentbench_frame.hl.local_config import LocalHLConfig
    from agentbench_frame.hl.research_state import ResearchState

    source_run = tmp_path / "source-run"
    source_state = ResearchState.empty(max_bytes=16384).advance(
        stable_knowledge=("rank15 requires cross-seed protection",),
        proposal_cycle=2,
    )
    source_path = source_state.write(source_run / "research_state.json")
    base = LocalHLConfig.load(CONFIG)
    config = dataclasses.replace(
        base,
        run=dataclasses.replace(
            base.run,
            origin=OriginConfig(
                mode="imported_version",
                source_run=str(source_run),
                source_version="v000000",
                reset_research_state=False,
            ),
        ),
    )

    destination = _prepare_research_state(
        config, run_dir=tmp_path / "new-run", resume=False
    )

    assert destination.read_bytes() == source_path.read_bytes()
    loaded = ResearchState.load_or_create(destination, max_bytes=16384)
    assert loaded.proposal_cycle == 2
    assert loaded.stable_knowledge == (
        "rank15 requires cross-seed protection",
    )


def test_imported_research_state_fails_closed_when_missing(tmp_path):
    from agentbench_frame.hl.cli import _prepare_research_state
    from agentbench_frame.hl.config import OriginConfig
    from agentbench_frame.hl.local_config import LocalHLConfig

    base = LocalHLConfig.load(CONFIG)
    config = dataclasses.replace(
        base,
        run=dataclasses.replace(
            base.run,
            origin=OriginConfig(
                mode="imported_version",
                source_run=str(tmp_path / "missing-run"),
                source_version="v000000",
                reset_research_state=False,
            ),
        ),
    )

    try:
        _prepare_research_state(
            config, run_dir=tmp_path / "new-run", resume=False
        )
    except FileNotFoundError as exc:
        assert "research state is missing" in str(exc)
    else:
        raise AssertionError("missing imported research state was accepted")


def test_imported_research_state_rejects_oversized_source(tmp_path):
    from agentbench_frame.hl.cli import _prepare_research_state
    from agentbench_frame.hl.config import ContextConfig, OriginConfig
    from agentbench_frame.hl.local_config import LocalHLConfig

    source_run = tmp_path / "source-run"
    source_run.mkdir()
    (source_run / "research_state.json").write_bytes(b"x" * 1025)
    base = LocalHLConfig.load(CONFIG)
    config = dataclasses.replace(
        base,
        run=dataclasses.replace(
            base.run,
            origin=OriginConfig(
                mode="imported_version",
                source_run=str(source_run),
                source_version="v000000",
                reset_research_state=False,
            ),
            context=ContextConfig(research_state_max_bytes=1024),
        ),
    )

    try:
        _prepare_research_state(
            config, run_dir=tmp_path / "new-run", resume=False
        )
    except ValueError as exc:
        assert "exceeds max_bytes=1024" in str(exc)
    else:
        raise AssertionError("oversized imported research state was accepted")


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
    assert "values" not in snapshot["paths"]
    assert "workspace" in snapshot["paths"]
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


def test_replay_summary_readiness_rejects_corrupt_evidence(monkeypatch):
    from agentbench_frame.hl import cli

    calls = []

    def summarize(*, replay, summarizer):
        calls.append(str(replay))
        if str(replay).endswith("broken.jsonl"):
            raise ValueError("missing terminal frame")
        return replay

    monkeypatch.setattr(cli, "_ensure_replay_summary", summarize)

    assert not cli._replay_summaries_ready(
        (
            {"status": "complete", "replay": "valid.jsonl"},
            {"status": "complete", "replay": "broken.jsonl"},
        ),
        summarizer="summarize.py",
    )
    assert calls == ["valid.jsonl", "broken.jsonl"]


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


def test_resume_parses_explicit_provider_compatibility_change(monkeypatch):
    from agentbench_frame.hl import cli

    monkeypatch.setattr(
        cli,
        "_cmd_resume",
        lambda args: int(args.allow_provider_compatibility_change),
    )

    assert (
        cli.main(
            [
                "resume",
                "--config",
                str(CONFIG),
                "--run-dir",
                "unused",
                "--allow-provider-compatibility-change",
            ]
        )
        == 1
    )


def test_resume_can_replan_an_interrupted_proposal_cycle(tmp_path, monkeypatch):
    from agentbench_frame.hl import cli
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CONFIG)
    calls = []
    monkeypatch.setattr(cli, "_load", lambda _path: config)
    monkeypatch.setattr(cli, "_run_real", lambda *args, **kwargs: calls.append(kwargs) or 0)

    code = cli.main(
        [
            "resume",
            "--config",
            str(CONFIG),
            "--run-dir",
            str(tmp_path / "run"),
            "--acts",
            "0",
            "--replan-pending",
        ]
    )

    assert code == 0
    assert calls[0]["resume"] is True
    assert calls[0]["replan_pending"] is True


def test_aggregate_report_cli_passes_integer_origin_and_run_paths(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl import cli, report

    calls = []
    monkeypatch.setattr(
        report,
        "write_aggregate_report",
        lambda source, phase, **values: calls.append(
            (source, phase, values)
        )
        or {"aggregate_curves_csv": tmp_path / "aggregate-curves.csv"},
    )

    code = cli.main(
        [
            "aggregate-report",
            "--source-run",
            str(tmp_path / "source"),
            "--phase-run",
            str(tmp_path / "phase"),
            "--origin-iteration",
            "11",
        ]
    )

    assert code == 0
    assert calls[0][0] == (tmp_path / "source").resolve()
    assert calls[0][1] == (tmp_path / "phase").resolve()
    assert calls[0][2]["global_origin_iteration"] == 11


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


def test_pending_planner_is_recovered_from_valid_persisted_stream(tmp_path):
    from agentbench_frame.hl.cli import _pending_planner_recovery
    from agentbench_frame.tracking.provider import ProviderInvocation

    workspace = tmp_path / "candidate"
    (workspace / ".agentbench").mkdir(parents=True)
    (workspace / ".agentbench" / "branch_briefs.json").write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"diagnosis-{index}",
                    "mechanism": mechanism,
                    "activation_condition": f"condition-{index}",
                    "preservation_contract": f"preserve-{index}",
                    "expected_change": f"expected-{index}",
                    "falsifier": f"falsifier-{index}",
                    "code_symbols": ["ai_func", "helper"],
                }
                for index, mechanism in enumerate(
                    ("adapter", "capture filter", "portal controller", "respawn memory")
                )
            ]
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "run" / "provider" / "act-000001-planner.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"type":"turn.completed"}\n', encoding="utf-8")

    class Provider:
        def __init__(self):
            self.calls = []

        def recover_completed_output(self, *, raw_output_path, workspace):
            self.calls.append((Path(raw_output_path), Path(workspace)))
            return ProviderInvocation(status="completed")

    provider = Provider()
    historical = [
        {
            "event_type": "proposal_cycle_started",
            "iteration_id": "iter-000001",
            "parent_version_id": "v000000",
        },
        {
            "event_type": "act_completed",
            "iteration_id": "iter-000001",
            "act_id": "act-000001-planner",
            "status": "failed",
            "raw_output_ref": str(raw),
        },
    ]

    recovered = _pending_planner_recovery(
        historical,
        provider=provider,
        workspace=workspace,
    )

    assert recovered is not None
    assert recovered.metadata["act_id"] == "act-000001-planner"
    assert recovered.metadata["iteration_id"] == "iter-000001"
    assert provider.calls == [(raw, workspace)]


def test_completed_planner_artifact_resumes_without_provider_call(tmp_path):
    from agentbench_frame.hl.cli import _pending_planner_recovery

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    persisted = tmp_path / "run" / "proposals" / "iter-000001" / "branch_briefs.json"
    persisted.parent.mkdir(parents=True)
    persisted.write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"diagnosis-{index}",
                    "mechanism": mechanism,
                    "activation_condition": f"condition-{index}",
                    "preservation_contract": f"preserve-{index}",
                    "expected_change": f"expected-{index}",
                    "falsifier": f"falsifier-{index}",
                    "code_symbols": ["ai_func", "helper"],
                }
                for index, mechanism in enumerate(
                    ("route", "shield", "portal", "escape")
                )
            ]
        ),
        encoding="utf-8",
    )

    recovered = _pending_planner_recovery(
        [
            {
                "event_type": "proposal_cycle_started",
                "iteration_id": "iter-000001",
            },
            {
                "event_type": "planner_completed",
                "iteration_id": "iter-000001",
                "act_id": "act-000001-planner",
                "branch_briefs": str(persisted),
            },
        ],
        provider=object(),
        workspace=workspace,
    )

    assert recovered.status == "completed"
    assert recovered.metadata["recovered_from_persisted_output"] is True
    assert (workspace / ".agentbench" / "branch_briefs.json").read_bytes() == persisted.read_bytes()


def test_pending_repair_is_recovered_from_checkpoint_and_immutable_version(tmp_path):
    from agentbench_frame.hl.cli import _pending_repair_recoveries
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.evaluator import CandidateEvaluation

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    initial = store.snapshot(parent_version_id=None, act_id="act-initial")
    (workspace / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    repaired = store.snapshot(
        parent_version_id=initial.version_id,
        act_id="act-000006-repair-b01",
        edit_type="repair",
    )
    checkpoint = tmp_path / "checkpoints" / "act-000006-repair-b01.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps(
            {
                "act_id": repaired.act_id,
                "iteration_id": "iter-000012",
                "branch_index": 1,
                "provider_status": "completed",
                "raw_output_ref": str(tmp_path / "provider.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    evaluation = CandidateEvaluation(status="complete", score=1.0)
    historical = [
        {
            "event_type": "proposal_cycle_started",
            "iteration_id": "iter-000012",
        },
        {
            "event_type": "checkpoint_created",
            "iteration_id": "iter-000012",
            "act_id": repaired.act_id,
            "path": str(checkpoint),
        },
        {
            "event_type": "version_created",
            "version_id": repaired.version_id,
            "act_id": repaired.act_id,
            "content_hash": repaired.content_hash,
        },
        {
            "event_type": "repair_completed",
            "iteration_id": "iter-000012",
            "act_id": repaired.act_id,
            "branch_index": 1,
            "initial_version_id": initial.version_id,
            "repaired_version_id": repaired.version_id,
            "status": "completed",
        },
    ]

    recovered = _pending_repair_recoveries(
        historical,
        version_store=store,
        evaluations_by_version={repaired.version_id: evaluation},
    )

    assert recovered[1].version == repaired
    assert recovered[1].evaluation is evaluation
    assert recovered[1].provider.metadata["recovered_from_persisted_output"] is True


def test_interrupted_bootstrap_with_clean_file_change_is_recoverable(tmp_path):
    from agentbench_frame.hl.cli import _bootstrap_recovery_candidate
    from agentbench_frame.tracking.provider import ProviderInvocation

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("def ai_func(state): return 1\n", encoding="utf-8")
    raw = tmp_path / "run" / "provider" / "act-000001-b00.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "file_change"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Provider:
        def recover_completed_output(self, *, raw_output_path, workspace):
            return ProviderInvocation(
                status="failed",
                metadata={"access_policy_violations": []},
            )

    recovery = _bootstrap_recovery_candidate(
        [
            {
                "event_type": "act_completed",
                "act_id": "act-000001-b00",
                "status": "failed",
                "raw_output_ref": str(raw),
            }
        ],
        workspace=workspace,
        provider=Provider(),
    )

    assert recovery == {
        "failed_act_id": "act-000001-b00",
        "raw_output_ref": str(raw),
        "failure_reason": "provider_interrupted_after_workspace_edit",
    }


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


def test_model_bootstrap_run_accepts_zero_proposal_cycles_with_provider(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl import cli
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CONFIG)
    calls = []
    monkeypatch.setattr(cli, "_load", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "_provider_environment",
        lambda _config: {"AGENTBENCH_API_KEY": "runtime-value"},
    )
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
    assert calls[0]["provider_environment"] == {
        "AGENTBENCH_API_KEY": "runtime-value"
    }


def test_model_bootstrap_resume_uses_provider_for_zero_cycle_recovery(
    tmp_path, monkeypatch
):
    from agentbench_frame.hl import cli
    from agentbench_frame.hl.local_config import LocalHLConfig

    config = LocalHLConfig.load(CONFIG)
    calls = []
    monkeypatch.setattr(cli, "_load", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "_provider_environment",
        lambda _config: {"AGENTBENCH_API_KEY": "runtime-value"},
    )
    monkeypatch.setattr(
        cli,
        "_run_real",
        lambda *args, **kwargs: calls.append(kwargs) or 0,
    )

    code = cli.main(
        [
            "resume",
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
    assert calls[0]["resume"] is True
    assert calls[0]["provider_environment"] == {
        "AGENTBENCH_API_KEY": "runtime-value"
    }


def test_provider_preflight_runs_before_paid_execution():
    from agentbench_frame.hl.cli import _preflight_provider

    class Provider:
        def __init__(self):
            self.calls = 0

        def preflight(self):
            self.calls += 1
            return {"cli_version": "codex-cli test"}

    provider = Provider()

    facts = _preflight_provider(provider)

    assert facts == {"cli_version": "codex-cli test"}
    assert provider.calls == 1
    assert _preflight_provider(None) is None

from pathlib import Path


def _workspace(root: Path) -> Path:
    workspace = root / "candidate"
    workspace.mkdir()
    (workspace / "agent.py").write_text("VALUE = 0\n", encoding="utf-8")
    return workspace


class FakeProvider:
    def __init__(self, edits, experience_updates=None):
        self.edits = iter(edits)
        self.experience_updates = iter(experience_updates or [])
        self.calls = []

    def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
        from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage

        edit = next(self.edits)
        Path(workspace, "agent.py").write_text(edit, encoding="utf-8")
        try:
            update = next(self.experience_updates)
        except StopIteration:
            update = None
        if update is not None:
            import json

            update_path = Path(workspace, ".agentbench/experience_update.json")
            update_path.parent.mkdir(parents=True, exist_ok=True)
            update_path.write_text(json.dumps(update), encoding="utf-8")
        Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_output_path).write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        self.calls.append(
            {"prompt": prompt, "workspace": str(workspace), "session_id": session_id}
        )
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(
                prompt_tokens=10,
                cached_input_tokens=8,
                completion_tokens=2,
                total_tokens=12,
                token_accuracy="exact",
            ),
            raw_output_ref=str(raw_output_path),
            metadata={"thread_id": f"thread-{len(self.calls)}"},
        )


class FakeEvaluator:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def evaluate(self, version):
        from agentbench_frame.hl.evaluator import CandidateEvaluation

        value = next(self.outcomes)
        if value is None:
            return CandidateEvaluation(status="incomplete", score=None, error="fixture failure")
        return CandidateEvaluation(status="complete", score=value)


def _controller(tmp_path, provider, evaluator, *, k=1, patience=3, experience=None):
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter
    from agentbench_frame.hl.lineage import LineageManager

    workspace = _workspace(tmp_path)
    versions = VersionStore(workspace, tmp_path / "versions")
    writer = HLEventWriter(tmp_path / "events.jsonl", run_id="run-test")
    lineage = LineageManager(rollback_patience=patience, rollback_margin=0.05)
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=provider,
        evaluator=evaluator,
        version_store=versions,
        lineage=lineage,
        events=writer,
        iteration=IterationConfig(max_acts=None, candidates_per_act=k),
        rollback=RollbackConfig(patience=patience),
        prompt_factory=lambda **values: (
            f"act={values['act_id']} branch={values['branch_index']}/{values['branch_count']}"
        ),
        experience_manager=experience,
    )
    return controller


def test_k_candidates_are_siblings_and_gate_selects_best_complete_score(tmp_path):
    provider = FakeProvider(["VALUE = 1\n", "VALUE = 2\n", "VALUE = 3\n"])
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.30, 0.80, 0.50]),
        k=3,
    )
    initial = controller.initialize()
    result = controller.run_act()

    assert len(result.candidates) == 3
    assert {candidate.version.parent_version_id for candidate in result.candidates} == {
        initial.version_id
    }
    assert result.selected.version.version_id == result.candidates[1].version.version_id
    assert controller.lineage.lineage_head_version_id == result.selected.version.version_id
    assert len({candidate.version.content_hash for candidate in result.candidates}) == 3
    checkpoint = tmp_path / "checkpoints" / f"{result.candidates[0].act_id}.json"
    assert checkpoint.is_file()
    assert "branch=0/3" in checkpoint.read_text(encoding="utf-8")


def test_stream_failed_bootstrap_can_be_recovered_without_second_provider_call(
    tmp_path,
):
    from agentbench_frame.hl.events import read_events

    provider = FakeProvider([])
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.25]),
    )
    (controller.workspace / "agent.py").write_text(
        "VALUE = 99\n", encoding="utf-8"
    )

    version = controller.recover_bootstrap(
        failed_act_id="act-000001-b00",
        raw_output_ref=str(tmp_path / "provider" / "failed.jsonl"),
        failure_reason="provider_stream_disconnected_after_workspace_edit",
    )

    assert provider.calls == []
    assert controller.lineage.lineage_head_version_id == version.version_id
    assert controller.lineage.champion_version_id == version.version_id
    events = read_events(tmp_path / "events.jsonl")
    recovered = [
        event
        for event in events
        if event["event_type"] == "bootstrap_recovered"
    ][0]
    assert recovered["failed_act_id"] == "act-000001-b00"
    assert recovered["version_id"] == version.version_id


def test_equal_win_rates_select_better_score_margin_instead_of_first_branch(tmp_path):
    from agentbench_frame.hl.evaluator import CandidateEvaluation

    class MarginEvaluator:
        def __init__(self):
            self.margins = iter((-500, -100, -200, -300))

        def evaluate(self, version):
            margin = next(self.margins)
            return CandidateEvaluation(
                status="complete",
                score=0.0,
                matches=(
                    {
                        "status": "complete",
                        "result": "loss",
                        "opponent": "rank15",
                        "seed": 101,
                        "rollman_score": 0,
                        "ghosts_score": -margin,
                        "game_agent_decisions": 100,
                    },
                ),
            )

    controller = _controller(
        tmp_path,
        FakeProvider(
            ["VALUE = 1\n", "VALUE = 2\n", "VALUE = 3\n", "VALUE = 4\n"]
        ),
        MarginEvaluator(),
        k=4,
    )
    controller.initialize()

    result = controller.run_act(promote_champion=False)

    assert result.selected.branch_index == 1


def test_staged_k4_evaluation_gives_all_quick_feedback_and_only_two_finalists(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import HLEventWriter
    from agentbench_frame.hl.lineage import LineageManager

    class StagedEvaluator:
        def __init__(self, workspace):
            self.workspace = workspace
            self.quick_calls = []
            self.finalist_calls = []
            self.finalist_workspace_values = {}

        def quick_screen(self, version):
            self.quick_calls.append(version.version_id)
            branch = int(version.version_id[1:]) - 1
            margins = (-500, -100, -200, -300)
            return CandidateEvaluation(
                status="complete",
                score=0.0,
                matches=(
                    {
                        "phase": "quick_screen",
                        "status": "complete",
                        "result": "loss",
                        "opponent": "rank15",
                        "seed": 101,
                        "rollman_score": 0,
                        "ghosts_score": -margins[branch],
                    },
                ),
            )

        def evaluate_finalist(self, version):
            self.finalist_calls.append(version.version_id)
            value = int(
                (self.workspace / "agent.py")
                .read_text(encoding="utf-8")
                .split("=")[1]
            )
            self.finalist_workspace_values[version.version_id] = value
            win = version.version_id == "v000002"
            return CandidateEvaluation(
                status="complete",
                score=1.0 if win else 0.0,
                matches=(
                    {
                        "phase": "finalist",
                        "status": "complete",
                        "result": "win" if win else "loss",
                        "opponent": "rank15",
                        "seed": 102,
                        "rollman_score": 100 if win else 0,
                        "ghosts_score": 0 if win else 100,
                    },
                ),
            )

        @staticmethod
        def combine_stages(quick, finalist):
            matches = (*quick.matches, *finalist.matches)
            points = sum(
                1.0 if match["result"] == "win" else 0.5 if match["result"] == "draw" else 0.0
                for match in matches
            ) / len(matches)
            return CandidateEvaluation(status="complete", score=points, matches=matches)

    workspace = _workspace(tmp_path)
    evaluator = StagedEvaluator(workspace)
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=FakeProvider(
            ["VALUE = 1\n", "VALUE = 2\n", "VALUE = 3\n", "VALUE = 4\n"]
        ),
        evaluator=evaluator,
        version_store=VersionStore(workspace, tmp_path / "versions"),
        lineage=LineageManager(),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-staged"),
        iteration=IterationConfig(
            candidates_per_cycle=4,
            planner_enabled=True,
            reducer_enabled=True,
            finalist_count=2,
        ),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: f"branch={values['branch_index']}",
    )
    controller.initialize()

    result = controller.run_act(promote_champion=False)

    assert len(evaluator.quick_calls) == 4
    assert set(evaluator.finalist_calls) == {"v000002", "v000003"}
    assert evaluator.finalist_workspace_values == {
        "v000002": 2,
        "v000003": 3,
    }
    assert len(result.finalists) == 2
    assert result.selected.version.version_id == "v000002"
    assert len(result.selected.evaluation.matches) == 2


def test_bootstrap_registers_only_provider_written_algorithm_as_origin(tmp_path):
    from agentbench_frame.hl.events import read_events

    provider = FakeProvider(["VALUE = 41\n", "VALUE = 42\n"])
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.35, 0.50]),
    )

    origin = controller.bootstrap()
    next_iteration = controller.run_act()
    events = read_events(tmp_path / "events.jsonl")

    assert origin.version.parent_version_id is None
    assert origin.version.edit_type == "initial"
    assert origin.evaluation.score == 0.35
    assert (controller.run_root / "versions" / "manifests" / "v000000.json").is_file()
    assert len(
        [event for event in events if event["event_type"] == "version_created"]
    ) == 2
    assert [
        event["version_id"]
        for event in events
        if event["event_type"] == "candidate_selected"
    ][0] == origin.version.version_id
    assert provider.calls[0]["session_id"] is None
    assert provider.calls[1]["session_id"] == "thread-1"
    assert next_iteration.selected.act_id.startswith("act-000002")
    assert controller.summary()["coding_agent_acts"] == 2


def test_imported_origin_is_registered_without_a_provider_call(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.events import read_events

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_workspace = _workspace(source_root)
    source_run = source_root / "run-source"
    source_store = VersionStore(source_workspace, source_run / "versions")
    source = source_store.snapshot(
        parent_version_id=None,
        act_id="source-act",
    )
    target_root = tmp_path / "target"
    target_root.mkdir()
    provider = FakeProvider([])
    controller = _controller(
        target_root,
        provider,
        FakeEvaluator([]),
    )

    imported = controller.initialize_imported(
        source_run=source_run,
        source_version_id=source.version_id,
    )
    events = read_events(target_root / "events.jsonl")

    assert imported.version_id == "v000000"
    assert imported.content_hash == source.content_hash
    assert imported.edit_type == "imported_origin"
    assert provider.calls == []
    assert controller.summary()["coding_agent_acts"] == 0
    assert controller.lineage.lineage_head_version_id == imported.version_id
    origin_events = [
        event
        for event in events
        if event["event_type"] == "origin_imported"
    ]
    assert len(origin_events) == 1
    assert origin_events[0]["source_run_id"] == "run-source"
    assert origin_events[0]["source_version_id"] == source.version_id
    assert origin_events[0]["source_content_hash"] == source.content_hash
    assert origin_events[0]["version_id"] == imported.version_id


def test_bootstrap_rejects_unchanged_scaffold_without_creating_origin(tmp_path):
    import pytest

    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 0\n"]),
        FakeEvaluator([]),
    )

    with pytest.raises(RuntimeError, match="did not change"):
        controller.bootstrap()

    assert not list(
        (controller.run_root / "versions" / "manifests").glob("*.json")
    )


def test_incomplete_bootstrap_persists_case_errors_and_can_retry_without_model(tmp_path):
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import read_events

    class RetryEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, version):
            self.calls += 1
            if self.calls == 1:
                return CandidateEvaluation(
                    status="incomplete",
                    score=None,
                    error="fixed case incomplete",
                    matches=(
                        {
                            "phase": "learning",
                            "status": "incomplete",
                            "opponent": "rank01",
                            "seed": 101,
                            "error": "player 1 timed out",
                        },
                    ),
                )
            return CandidateEvaluation(
                status="complete",
                score=1.0,
                matches=(
                    {
                        "phase": "learning",
                        "status": "complete",
                        "opponent": "rank01",
                        "seed": 101,
                        "result": "win",
                    },
                ),
            )

    provider = FakeProvider(["VALUE = 9\n"])
    controller = _controller(tmp_path, provider, RetryEvaluator())

    origin = controller.bootstrap()
    retried = controller.retry_head_evaluation()
    events = read_events(tmp_path / "events.jsonl")
    evaluations = [
        event for event in events if event["event_type"] == "evaluation_completed"
    ]

    assert origin.evaluation.status == "incomplete"
    assert evaluations[0]["status"] == "incomplete"
    assert evaluations[0]["matches"][0]["error"] == "player 1 timed out"
    assert retried.status == "complete"
    assert len(provider.calls) == 1
    assert controller.lineage.champion_version_id == origin.version.version_id
    assert [event["status"] for event in evaluations] == ["incomplete", "complete"]


def test_each_invocation_gets_logical_version_and_incomplete_score_stays_missing(tmp_path):
    provider = FakeProvider(["VALUE = 0\n"])
    controller = _controller(tmp_path, provider, FakeEvaluator([None]))
    initial = controller.initialize()
    result = controller.run_act()

    assert result.selected.version.version_id != initial.version_id
    assert result.selected.version.content_hash == initial.content_hash
    assert result.selected.evaluation.status == "incomplete"
    assert result.selected.evaluation.score is None


def test_sustained_degradation_makes_next_act_start_from_historical_champion(tmp_path):
    provider = FakeProvider(
        ["VALUE = 10\n", "VALUE = 11\n", "VALUE = 12\n", "VALUE = 13\n"]
    )
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.80, 0.60, 0.61, 0.62, 0.70]),
        patience=3,
    )
    initial = controller.initialize(evaluate=True)
    controller.run_act()
    controller.run_act()
    third = controller.run_act()
    fourth = controller.run_act()

    assert controller.lineage.champion_version_id == initial.version_id
    assert fourth.parent_version_id == initial.version_id
    assert fourth.rollback is not None
    assert fourth.rollback.from_version_id == third.selected.version.version_id
    assert fourth.rollback.to_version_id == initial.version_id


def test_explicit_safe_parent_overrides_latest_curriculum_candidate(tmp_path):
    provider = FakeProvider(["VALUE = 1\n", "VALUE = 2\n"])
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.8, 0.6, 0.7]),
    )
    initial = controller.initialize(evaluate=True)
    first = controller.run_act()

    second = controller.run_act(parent_version_id=initial.version_id)

    assert first.selected.version.version_id != initial.version_id
    assert second.parent_version_id == initial.version_id
    assert second.rollback is not None
    assert second.rollback.from_version_id == first.selected.version.version_id
    assert second.rollback.to_version_id == initial.version_id
    assert second.rollback.reason == "curriculum_regression"


def test_open_ended_config_has_no_implicit_iteration_cap(tmp_path):
    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n"]),
        FakeEvaluator([0.5]),
    )

    assert controller.iteration.max_acts is None
    assert controller.reached_iteration_limit() is False


def test_controller_ingests_structured_experience_update_outside_version(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager

    experience = ExperienceManager(tmp_path / "experience")
    provider = FakeProvider(
        ["VALUE = 1\n"],
        experience_updates=[
            {
                "stable_knowledge": ["Replay-complete collision evidence is reusable."],
                "failed_hypotheses": [],
                "replay_evidence": ["replay-x level 1 round 3"],
                "active_questions": [],
            }
        ],
    )
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.5]),
        experience=experience,
    )
    controller.initialize()
    result = controller.run_act()

    assert "collision evidence" in experience.path.read_text(encoding="utf-8")
    assert ".agentbench/experience_update.json" not in result.selected.version.files


def test_k4_proposal_cycle_uses_one_parent_and_reducer_sees_all_feedback(tmp_path):
    import json

    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter
    from agentbench_frame.hl.lineage import LineageManager
    from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage

    class ProposalProvider:
        def __init__(self):
            self.calls = []
            self.candidate_index = 0

        def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
            self.calls.append(prompt)
            control = Path(workspace, ".agentbench")
            control.mkdir(parents=True, exist_ok=True)
            if "phase=planner" in prompt:
                (control / "branch_briefs.json").write_text(
                    json.dumps(
                        {
                            "branches": [
                                {
                                    "branch_index": index,
                                    "diagnosis": f"diagnosis-{index}",
                                    "mechanism": mechanism,
                                    "activation_condition": f"condition-{index}",
                                    "preservation_contract": f"preserve-{index}",
                                    "expected_change": f"expected-{index}",
                                    "falsifier": f"falsifier-{index}",
                                }
                                for index, mechanism in enumerate(
                                    ("planner", "predictor", "shield", "portal")
                                )
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            elif "phase=candidate" in prompt:
                self.candidate_index += 1
                Path(workspace, "agent.py").write_text(
                    f"VALUE = {self.candidate_index}\n",
                    encoding="utf-8",
                )
            elif "phase=reducer" in prompt:
                (control / "research_state_update.json").write_text(
                    json.dumps(
                        {
                            "stable_knowledge": ["branch 1 improved the gate"],
                            "failed_hypotheses": ["branch 0 regressed"],
                            "open_questions": ["portal timing"],
                            "recent_comparisons": [
                                {"selected_branch": 1, "branches": [0, 1, 2, 3]}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                raise AssertionError(prompt)
            Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_output_path).write_text("{}\n", encoding="utf-8")
            return ProviderInvocation(
                status="completed",
                usage=ProviderUsage(prompt_tokens=1, completion_tokens=1),
                raw_output_ref=str(raw_output_path),
                metadata={"thread_id": f"thread-{len(self.calls)}"},
            )

    workspace = _workspace(tmp_path)
    provider = ProposalProvider()
    research_state_path = tmp_path / "research_state.json"
    from agentbench_frame.hl.research_state import ResearchState

    ResearchState.empty(max_bytes=4096).advance(
        official_champion_version_id="v000000",
        active_target="rank15",
    ).write(research_state_path)
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=provider,
        evaluator=FakeEvaluator([0.8, 0.1, 0.7, 0.4, 0.2]),
        version_store=VersionStore(workspace, tmp_path / "versions"),
        lineage=LineageManager(),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-k4"),
        iteration=IterationConfig(
            candidates_per_cycle=4,
            planner_enabled=True,
            reducer_enabled=True,
            finalist_count=2,
        ),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: (
            f"phase={values['phase']} branch={values.get('branch_index')} "
            f"brief={values.get('branch_brief')} input={values.get('reducer_input')}"
        ),
        research_state_path=research_state_path,
        research_state_max_bytes=4096,
    )
    origin = controller.initialize(evaluate=True)

    result = controller.run_proposal_cycle(parent_version_id=origin.version_id)

    assert len(result.candidates) == 4
    assert {candidate.version.parent_version_id for candidate in result.candidates} == {
        origin.version_id
    }
    assert len(result.finalists) == 2
    assert result.selected in result.finalists
    assert controller.lineage.lineage_head_version_id == result.selected.version.version_id
    assert controller.lineage.champion_version_id == origin.version_id
    reducer_payload = json.loads(result.reducer_input_path.read_text(encoding="utf-8"))
    assert {row["branch_index"] for row in reducer_payload["candidates"]} == {
        0,
        1,
        2,
        3,
    }
    assert len(provider.calls) == 6
    research = ResearchState.load_or_create(
        research_state_path,
        max_bytes=4096,
    )
    assert research.proposal_cycle == 1
    assert research.search_parent_version_id == result.selected.version.version_id
    assert research.official_champion_version_id == origin.version_id
    assert research.active_target == "rank15"
    assert research.recent_comparisons[0]["selected_branch"] == 1


def test_failed_reducer_output_cannot_mutate_research_state(tmp_path):
    import json

    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter, read_events
    from agentbench_frame.hl.lineage import LineageManager
    from agentbench_frame.hl.research_state import ResearchState
    from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage

    class Provider:
        def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
            control = Path(workspace, ".agentbench")
            control.mkdir(parents=True, exist_ok=True)
            status = "completed"
            if "phase=planner" in prompt:
                (control / "branch_briefs.json").write_text(
                    json.dumps(
                        {
                            "branches": [
                                {
                                    "branch_index": index,
                                    "diagnosis": f"diagnosis-{index}",
                                    "mechanism": ("planner", "predictor", "shield", "portal")[index],
                                    "activation_condition": f"condition-{index}",
                                    "preservation_contract": f"preserve-{index}",
                                    "expected_change": f"expected-{index}",
                                    "falsifier": f"falsifier-{index}",
                                }
                                for index in range(4)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            elif "phase=candidate" in prompt:
                Path(workspace, "agent.py").write_text(
                    f"VALUE = {len(list((control).iterdir()))}\n",
                    encoding="utf-8",
                )
            elif "phase=reducer" in prompt:
                (control / "research_state_update.json").write_text(
                    json.dumps(
                        {
                            "stable_knowledge": ["tainted update"],
                            "failed_hypotheses": [],
                            "open_questions": [],
                            "recent_comparisons": [],
                        }
                    ),
                    encoding="utf-8",
                )
                status = "failed"
            Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_output_path).write_text("{}\n", encoding="utf-8")
            return ProviderInvocation(
                status=status,
                usage=ProviderUsage(prompt_tokens=1, completion_tokens=1),
                raw_output_ref=str(raw_output_path),
                metadata={"thread_id": "thread"},
            )

    workspace = _workspace(tmp_path)
    research_path = tmp_path / "research_state.json"
    ResearchState.empty(max_bytes=4096).advance(
        official_champion_version_id="v000000",
        active_target="rank15",
    ).write(research_path)
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=Provider(),
        evaluator=FakeEvaluator([0.8, 0.1, 0.7, 0.4, 0.2]),
        version_store=VersionStore(workspace, tmp_path / "versions"),
        lineage=LineageManager(),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-failed-reducer"),
        iteration=IterationConfig(
            candidates_per_cycle=4,
            planner_enabled=True,
            reducer_enabled=True,
            finalist_count=2,
        ),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: f"phase={values['phase']}",
        research_state_path=research_path,
        research_state_max_bytes=4096,
    )
    origin = controller.initialize(evaluate=True)

    controller.run_proposal_cycle(parent_version_id=origin.version_id)

    research = ResearchState.load_or_create(research_path, max_bytes=4096)
    assert research.proposal_cycle == 0
    assert "tainted update" not in research.stable_knowledge
    reducer = [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if event["event_type"] == "reducer_completed"
    ][0]
    assert reducer["status"] == "failed"
    assert reducer["output_path"] is None


def test_k4_cycle_reuses_valid_persisted_planner_without_second_api_call(tmp_path):
    import json

    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter
    from agentbench_frame.hl.lineage import LineageManager
    from agentbench_frame.tracking.provider import ProviderInvocation

    class RecoveryProvider:
        def __init__(self):
            self.calls = []

        def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
            self.calls.append(prompt)
            assert "phase=planner" not in prompt
            control = Path(workspace, ".agentbench")
            control.mkdir(parents=True, exist_ok=True)
            if "phase=candidate" in prompt:
                Path(workspace, "agent.py").write_text(
                    f"VALUE = {len(self.calls)}\n", encoding="utf-8"
                )
            else:
                (control / "research_state_update.json").write_text(
                    json.dumps(
                        {
                            "stable_knowledge": ["recovered planner was valid"],
                            "failed_hypotheses": [],
                            "open_questions": [],
                            "recent_comparisons": [
                                {"branches": [0, 1, 2, 3]}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_output_path).write_text("{}\n", encoding="utf-8")
            return ProviderInvocation(status="completed")

    workspace = _workspace(tmp_path)
    control = workspace / ".agentbench"
    control.mkdir()
    (control / "branch_briefs.json").write_text(
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
                }
                for index, mechanism in enumerate(
                    ("adapter", "capture filter", "portal controller", "respawn memory")
                )
            ]
        ),
        encoding="utf-8",
    )
    provider = RecoveryProvider()
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=provider,
        evaluator=FakeEvaluator([0.5, 0.1, 0.2, 0.3, 0.4]),
        version_store=VersionStore(workspace, tmp_path / "versions"),
        lineage=LineageManager(),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-recovery"),
        iteration=IterationConfig(
            candidates_per_cycle=4,
            planner_enabled=True,
            reducer_enabled=True,
            finalist_count=2,
        ),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: f"phase={values['phase']}",
    )
    origin = controller.initialize(evaluate=True)
    controller._coding_agent_acts = 1
    recovered = ProviderInvocation(
        status="completed",
        raw_output_ref=str(tmp_path / "provider" / "act-000001-planner.jsonl"),
        metadata={
            "act_id": "act-000001-planner",
            "iteration_id": "iter-000001",
            "recovered_from_persisted_output": True,
        },
    )

    result = controller.run_proposal_cycle(
        parent_version_id=origin.version_id,
        planner_recovery=recovered,
    )

    assert result.planner is recovered
    assert len(result.candidates) == 4
    assert len(provider.calls) == 5
    assert all("phase=planner" not in prompt for prompt in provider.calls)


def test_k4_cycle_keeps_current_parent_when_every_candidate_regresses(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import HLEventWriter
    from agentbench_frame.hl.lineage import LineageManager

    class Provider(FakeProvider):
        def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
            import json

            control = Path(workspace, ".agentbench")
            control.mkdir(parents=True, exist_ok=True)
            if "planner" in prompt:
                (control / "branch_briefs.json").write_text(
                    json.dumps(
                        {
                            "branches": [
                                {
                                    "branch_index": index,
                                    "diagnosis": f"d-{index}",
                                    "mechanism": (
                                        "portal search",
                                        "ghost prediction",
                                        "shield timing",
                                        "junction escape",
                                    )[index],
                                    "activation_condition": f"condition-{index}",
                                    "preservation_contract": f"preserve-{index}",
                                    "expected_change": f"e-{index}",
                                    "falsifier": f"f-{index}",
                                }
                                for index in range(4)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            elif "candidate" in prompt:
                Path(workspace, "agent.py").write_text(
                    f"VALUE = {len(self.calls) + 1}\n", encoding="utf-8"
                )
            elif "reducer" in prompt:
                (control / "research_state_update.json").write_text(
                    json.dumps(
                        {
                            "stable_knowledge": [],
                            "failed_hypotheses": ["all branches regressed"],
                            "open_questions": [],
                            "recent_comparisons": [],
                        }
                    ),
                    encoding="utf-8",
                )
            return super().invoke(
                prompt=prompt,
                workspace=workspace,
                raw_output_path=raw_output_path,
                session_id=session_id,
            )

    parent_evaluation = CandidateEvaluation(
        status="complete",
        score=0.0,
        matches=(
                {
                    "status": "complete",
                    "result": "loss",
                    "opponent": "rank15",
                    "seed": 101,
                    "rollman_score": 90,
                "ghosts_score": 100,
            },
        ),
    )
    candidate_evaluations = [
        CandidateEvaluation(
            status="complete",
            score=0.0,
            matches=(
                {
                    "status": "complete",
                    "result": "loss",
                    "opponent": "rank15",
                    "seed": 101,
                    "rollman_score": margin,
                    "ghosts_score": 100,
                },
            ),
        )
        for margin in (10, 20, 30, 40)
    ]

    workspace = _workspace(tmp_path)
    provider = Provider(["", "", "", "", "", ""])
    evaluator = FakeEvaluator([])
    evaluator.evaluate = lambda version: candidate_evaluations.pop(0)
    controller = HLController(
        workspace=workspace,
        run_root=tmp_path,
        provider=provider,
        evaluator=evaluator,
        version_store=VersionStore(workspace, tmp_path / "versions"),
        lineage=LineageManager(),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-k4"),
        iteration=IterationConfig(
            candidates_per_cycle=4,
            planner_enabled=True,
            reducer_enabled=True,
            finalist_count=2,
        ),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: f"phase={values['phase']}",
    )
    origin = controller.initialize()

    result = controller.run_proposal_cycle(
        parent_version_id=origin.version_id,
        parent_evaluation=parent_evaluation,
    )

    assert result.search_parent_version_id == origin.version_id
    assert controller.lineage.lineage_head_version_id == origin.version_id
    assert result.selected.version.version_id != origin.version_id


def test_curriculum_can_defer_experience_until_candidate_validation(tmp_path):
    from agentbench_frame.hl.events import read_events
    from agentbench_frame.hl.experience import ExperienceManager

    experience = ExperienceManager(tmp_path / "experience")
    provider = FakeProvider(
        ["VALUE = 1\n"],
        experience_updates=[
            {
                "stable_knowledge": ["Only commit this after policy probes pass."],
                "failed_hypotheses": [],
                "replay_evidence": ["rank15 seed 101 level 2 round 7"],
                "active_questions": [],
            }
        ],
    )
    controller = _controller(
        tmp_path,
        provider,
        FakeEvaluator([0.0]),
        experience=experience,
    )
    controller.initialize()

    result = controller.run_act(defer_experience=True)

    assert "Only commit this" not in experience.path.read_text(encoding="utf-8")
    assert not [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if event["event_type"] == "experience_updated"
    ]
    controller.commit_experience(result.selected)
    assert "Only commit this" in experience.path.read_text(encoding="utf-8")
    assert len(
        [
            event
            for event in read_events(tmp_path / "events.jsonl")
            if event["event_type"] == "experience_updated"
        ]
    ) == 1


def test_completed_matches_emit_win_rate_inputs_and_role_elo(tmp_path):
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import read_events

    class MatchEvaluator:
        def evaluate(self, version):
            return CandidateEvaluation(
                status="complete",
                score=0.5,
                matches=(
                    {
                        "status": "complete",
                        "opponent": "rank01",
                        "seed": 7,
                        "result": "win",
                        "rollman_score": 4,
                        "ghosts_score": 2,
                    },
                    {
                        "status": "complete",
                        "opponent": "rank01",
                        "seed": 8,
                        "result": "loss",
                        "rollman_score": 1,
                        "ghosts_score": 3,
                    },
                ),
            )

    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n"]),
        MatchEvaluator(),
    )
    controller.initialize()
    controller.run_act()
    events = read_events(tmp_path / "events.jsonl")

    evaluation = [
        event for event in events if event["event_type"] == "evaluation_completed"
    ][0]
    assert (evaluation["wins"], evaluation["draws"], evaluation["losses"]) == (
        1,
        0,
        1,
    )
    assert len(
        [event for event in events if event["event_type"] == "elo_updated"]
    ) == 2


def test_same_version_records_distinct_targets_without_match_id_collision(
    tmp_path,
):
    from agentbench_frame.hl.events import read_events

    controller = _controller(
        tmp_path,
        FakeProvider([]),
        FakeEvaluator([]),
    )
    version = controller.initialize()

    first = controller.record_matches(
        version=version,
        act_id=version.act_id,
        phase="learning",
        matches=(
            {
                "status": "complete",
                "opponent": "rank15",
                "seed": 101,
                "result": "loss",
            },
        ),
    )
    second = controller.record_matches(
        version=version,
        act_id=version.act_id,
        phase="learning",
        matches=(
            {
                "status": "complete",
                "opponent": "rank14",
                "seed": 101,
                "result": "loss",
            },
        ),
    )

    match_events = [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if event["event_type"] == "match_completed"
    ]
    assert first == 1
    assert second == 1
    assert len(match_events) == 2
    assert len({event["match_id"] for event in match_events}) == 2


def test_resume_restores_act_counter_and_parent_session(tmp_path):
    from agentbench_frame.hl.events import read_events
    from agentbench_frame.hl.lineage import LineageManager

    first = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n"]),
        FakeEvaluator([0.5]),
    )
    first.initialize()
    first_result = first.run_act()
    history = read_events(tmp_path / "events.jsonl")

    resumed_lineage = LineageManager.from_events(history)
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter

    provider = FakeProvider(["VALUE = 2\n"])
    resumed = HLController(
        workspace=first.workspace,
        run_root=tmp_path,
        provider=provider,
        evaluator=FakeEvaluator([0.6]),
        version_store=VersionStore(first.workspace, tmp_path / "versions"),
        lineage=resumed_lineage,
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-test"),
        iteration=IterationConfig(),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: values["act_id"],
    )
    resumed.resume(history)
    second_result = resumed.run_act()

    assert second_result.parent_version_id == first_result.selected.version.version_id
    assert second_result.selected.act_id.startswith("act-000002")
    assert provider.calls[0]["session_id"] == "thread-1"


def test_resume_rebuilds_elo_from_finalized_match_events(tmp_path):
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import HLEventWriter, read_events
    from agentbench_frame.hl.lineage import LineageManager

    class WinningEvaluator:
        def evaluate(self, version):
            return CandidateEvaluation(
                status="complete",
                score=1.0,
                matches=(
                    {
                        "phase": "learning",
                        "status": "complete",
                        "opponent": "rank01",
                        "seed": 7,
                        "result": "win",
                    },
                ),
            )

    first = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n"]),
        WinningEvaluator(),
    )
    first.initialize()
    first_result = first.run_act()
    history = read_events(tmp_path / "events.jsonl")

    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.config import IterationConfig, RollbackConfig
    from agentbench_frame.hl.controller import HLController

    resumed = HLController(
        workspace=first.workspace,
        run_root=tmp_path,
        provider=FakeProvider(["VALUE = 2\n"]),
        evaluator=WinningEvaluator(),
        version_store=VersionStore(first.workspace, tmp_path / "versions"),
        lineage=LineageManager.from_events(history),
        events=HLEventWriter(tmp_path / "events.jsonl", run_id="run-test"),
        iteration=IterationConfig(),
        rollback=RollbackConfig(),
        prompt_factory=lambda **values: values["act_id"],
    )
    resumed.resume(history)
    resumed.record_matches(
        version=first_result.selected.version,
        act_id=first_result.selected.act_id,
        phase="certification",
        matches=(
            {
                "status": "complete",
                "opponent": "rank01",
                "seed": 8,
                "result": "win",
            },
        ),
    )

    elo_events = [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if event["event_type"] == "elo_updated"
    ]
    assert elo_events[-1]["rating_before"] == elo_events[-2]["rating"]
    assert elo_events[-1]["rating"] > elo_events[-2]["rating"]


def test_k_sibling_elo_is_version_local_and_branch_order_independent(tmp_path):
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import read_events

    class MatchEvaluator:
        def evaluate(self, version):
            return CandidateEvaluation(
                status="complete",
                score=1.0,
                matches=(
                    {
                        "status": "complete",
                        "opponent": "rank01",
                        "seed": 7,
                        "result": "win",
                    },
                ),
            )

    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n", "VALUE = 2\n"]),
        MatchEvaluator(),
        k=2,
    )
    controller.initialize()
    controller.run_act()

    elo_events = [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if event["event_type"] == "elo_updated"
    ]
    assert len(elo_events) == 2
    assert {event["rating_before"] for event in elo_events} == {1500.0}
    assert len({event["rating"] for event in elo_events}) == 1
    assert len({event["candidate"] for event in elo_events}) == 2


def test_external_certification_matches_enter_match_and_elo_event_stream(tmp_path):
    from agentbench_frame.hl.events import read_events

    controller = _controller(
        tmp_path,
        FakeProvider([]),
        FakeEvaluator([0.5]),
    )
    version = controller.initialize()
    controller.record_matches(
        version=version,
        act_id="initial",
        phase="certification",
        matches=(
            {
                "status": "complete",
                "opponent": "rank16",
                "seed": 11,
                "result": "win",
                "rollman_score": 9,
                "ghosts_score": 1,
            },
        ),
    )
    controller.record_matches(
        version=version,
        act_id="initial",
        phase="certification",
        matches=(
            {
                "status": "complete",
                "opponent": "rank16",
                "seed": 11,
                "result": "win",
            },
        ),
    )

    events = read_events(tmp_path / "events.jsonl")
    certification_matches = [
        event
        for event in events
        if event["event_type"] == "match_completed"
        and event.get("phase") == "certification"
    ]
    certification_elo = [
        event
        for event in events
        if event["event_type"] == "elo_updated"
        and event.get("phase") == "certification"
    ]
    assert len(certification_matches) == 1
    assert len(certification_elo) == 1
    assert certification_matches[0]["match_id"].endswith(
        "-certification-0000"
    )

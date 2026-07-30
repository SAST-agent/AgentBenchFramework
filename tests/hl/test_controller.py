from pathlib import Path


def _workspace(root: Path) -> Path:
    workspace = root / "candidate"
    workspace.mkdir()
    (workspace / "agent.py").write_text("VALUE = 0\n", encoding="utf-8")
    return workspace


class FakeProvider:
    def __init__(self, edits):
        self.edits = iter(edits)
        self.calls = []

    def invoke(self, *, prompt, workspace, raw_output_path, session_id=None):
        from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage

        edit = next(self.edits)
        Path(workspace, "agent.py").write_text(edit, encoding="utf-8")
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


def _controller(tmp_path, provider, evaluator, *, k=1, patience=3):
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


def test_open_ended_config_has_no_implicit_iteration_cap(tmp_path):
    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n"]),
        FakeEvaluator([0.5]),
    )

    assert controller.iteration.max_acts is None
    assert controller.reached_iteration_limit() is False


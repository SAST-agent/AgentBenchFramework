from pathlib import Path


def test_fake_closed_loop_records_versions_tokens_rollback_and_summary(tmp_path):
    from agentbench_frame.hl.events import read_events
    from tests.hl.test_controller import FakeEvaluator, FakeProvider, _controller

    controller = _controller(
        tmp_path,
        FakeProvider(["VALUE = 1\n", "VALUE = 2\n", "VALUE = 3\n", "VALUE = 4\n"]),
        FakeEvaluator([0.80, 0.60, 0.61, 0.62, 0.75]),
        patience=3,
    )
    controller.initialize(evaluate=True)
    for _ in range(4):
        controller.run_act()

    events = read_events(tmp_path / "events.jsonl")
    types = [event["event_type"] for event in events]
    act_events = [event for event in events if event["event_type"] == "act_completed"]

    assert types.count("version_created") == 5
    assert "rollback_selected" in types
    assert sum(event["total_tokens"] for event in act_events) == 48
    assert controller.summary()["best_score"] == 0.80
    assert controller.summary()["coding_agent_acts"] == 4


def test_frozen_rollman_repair_protocol_is_imported_k4_top2(monkeypatch):
    from agentbench_frame.hl.local_config import LocalHLConfig

    root = Path(__file__).parents[2]
    monkeypatch.setenv("AGENTBENCH_SAST_ROOT", str(root.parent))

    config = LocalHLConfig.load(
        root / "configs/hl/29_rollman-k4-repair.yaml"
    )

    assert config.run.origin.mode == "imported_version"
    assert config.run.origin.source_version == "v000037"
    assert config.run.iteration.candidates_per_cycle == 4
    assert config.run.iteration.scope_contract_required is True
    assert config.run.iteration.repair_enabled is True
    assert config.run.iteration.repair_top_k == 2
    assert config.run.iteration.repair_rounds == 1
    assert config.run.iteration.finalist_count == 2

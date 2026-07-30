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

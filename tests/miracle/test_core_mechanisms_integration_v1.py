import inspect

import pytest

import agentbench_frame.games.miracle as miracle
from agentbench_frame.games.miracle import decision_kl_v1 as decision_kl
from agentbench_frame.games.miracle import replay_reading_v1 as replay_reading


def test_both_public_api_groups_export_their_defining_objects():
    expected = {
        "DecisionKLEvidence": decision_kl.DecisionKLEvidence,
        "DecisionKLRecord": decision_kl.DecisionKLRecord,
        "compute_trajectory_kl": decision_kl.compute_trajectory_kl,
        "ReplayPacket": replay_reading.ReplayPacket,
        "ReplayReadingContext": replay_reading.ReplayReadingContext,
        "render_replay_timeline": replay_reading.render_replay_timeline,
    }
    for name, value in expected.items():
        assert getattr(miracle, name) is value
        assert name in miracle.__all__


def test_core_boundaries_coexist_without_scalar_or_packet_bypass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    forged_record = decision_kl.DecisionKLRecord(
        1, "schema", "support", ("a",), "complete", 0.0
    )
    with pytest.raises(TypeError, match="evidence"):
        miracle.compute_trajectory_kl([forged_record])

    forged_packet = replay_reading.ReplayPacket(
        "m", "r", "p", {"case_id": "x"}, "validation", {}, {}, {}, (), {}
    )
    with pytest.raises(TypeError, match="context"):
        miracle.render_replay_timeline(forged_packet)
    with pytest.raises(ValueError, match="trusted|issued"):
        miracle.render_replay_timeline(
            object.__new__(replay_reading.ReplayReadingContext)
        )
    assert miracle.compute_trajectory_kl([]).trajectory_kl is None
    assert not (tmp_path / "workspace").exists()


def test_public_core_signatures_have_no_external_execution_inputs():
    forbidden = {"provider", "judge", "policy", "factory", "session", "runner"}
    functions = (
        miracle.build_trusted_action_support,
        miracle.compute_trajectory_kl,
        miracle.preflight_replay_reading,
        miracle.open_replay_reading,
        miracle.render_replay_timeline,
    )
    for function in functions:
        assert not (set(inspect.signature(function).parameters) & forbidden)


def test_public_api_does_not_export_caller_support_local_kl_calculator():
    assert "compute_local_kl" not in decision_kl.__all__
    assert "compute_local_kl" not in miracle.__all__
    assert not hasattr(decision_kl, "compute_local_kl")
    assert not hasattr(miracle, "compute_local_kl")

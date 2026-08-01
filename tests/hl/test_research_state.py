import json

import pytest


def test_research_state_round_trip_is_canonical_and_bounded(tmp_path):
    from agentbench_frame.hl.research_state import ResearchState

    path = tmp_path / "research_state.json"
    state = ResearchState.empty(max_bytes=4096).advance(
        proposal_cycle=2,
        search_parent_version_id="v000008",
        official_champion_version_id="v000003",
        active_target="rank15",
        locked_opponents=("rank16",),
        stable_knowledge=("Path crossing causes captures.",),
        failed_hypotheses=("Pure Manhattan distance failed on seed 101.",),
        open_questions=("Does shield timing change the best route?",),
        recent_comparisons=({"branch_index": 1, "score_margin": -20},),
        exploration_debt=1,
    )

    state.write(path)
    loaded = ResearchState.load_or_create(path, max_bytes=4096)

    assert loaded == state
    assert json.loads(path.read_text(encoding="utf-8"))["proposal_cycle"] == 2
    assert len(path.read_bytes()) <= 4096


def test_research_state_rejects_credentials_and_byte_overflow():
    from agentbench_frame.hl.research_state import ResearchState

    state = ResearchState.empty(max_bytes=1024)
    with pytest.raises(ValueError, match="credential"):
        state.advance(stable_knowledge=("sk-secret-value",))
    with pytest.raises(ValueError, match="max_bytes"):
        state.advance(stable_knowledge=("x" * 4000,))

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


def test_reducer_update_only_changes_semantic_fields_and_framework_owns_pointers(
    tmp_path,
):
    from agentbench_frame.hl.research_state import (
        ResearchState,
        apply_reducer_update,
    )

    current = ResearchState.empty(max_bytes=4096).advance(
        official_champion_version_id="v000001",
        active_target="rank15",
        locked_opponents=("rank16",),
    )
    update = tmp_path / "update.json"
    update.write_text(
        json.dumps(
            {
                "stable_knowledge": ["junction escape improved margin"],
                "failed_hypotheses": ["nearest food alone"],
                "open_questions": ["portal timing"],
                "recent_comparisons": [{"selected_branch": 2}],
            }
        ),
        encoding="utf-8",
    )

    advanced = apply_reducer_update(
        current,
        update,
        proposal_cycle=3,
        search_parent_version_id="v000009",
        official_champion_version_id="v000001",
        exploration_debt=2,
    )

    assert advanced.proposal_cycle == 3
    assert advanced.search_parent_version_id == "v000009"
    assert advanced.official_champion_version_id == "v000001"
    assert advanced.active_target == "rank15"
    assert advanced.locked_opponents == ("rank16",)
    assert advanced.stable_knowledge == ("junction escape improved margin",)


def test_reducer_update_rejects_framework_owned_fields(tmp_path):
    from agentbench_frame.hl.research_state import (
        ResearchState,
        apply_reducer_update,
    )

    update = tmp_path / "update.json"
    update.write_text(
        json.dumps(
            {
                "official_champion_version_id": "v999999",
                "stable_knowledge": [],
                "failed_hypotheses": [],
                "open_questions": [],
                "recent_comparisons": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reducer update fields"):
        apply_reducer_update(
            ResearchState.empty(max_bytes=4096),
            update,
            proposal_cycle=1,
            search_parent_version_id="v000001",
            official_champion_version_id="v000000",
            exploration_debt=0,
        )


def test_framework_comparisons_prepend_exact_margin_evidence():
    from agentbench_frame.hl.research_state import (
        ResearchState,
        prepend_framework_comparisons,
    )

    current = ResearchState.empty(max_bytes=4096).advance(
        recent_comparisons=({"version": "v-old", "summary": "older"},),
    )
    advanced = prepend_framework_comparisons(
        current,
        (
            {
                "version_id": "v000073",
                "branch_index": 2,
                "opponent": "rank15",
                "seed": 102,
                "margin_delta": 380.0,
            },
        ),
    )

    assert advanced.recent_comparisons[0] == {
        "source": "framework_positive_margin_delta",
        "version_id": "v000073",
        "branch_index": 2,
        "opponent": "rank15",
        "seed": 102,
        "margin_delta": 380.0,
    }
    assert advanced.recent_comparisons[1]["version"] == "v-old"

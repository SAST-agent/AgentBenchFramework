import pytest


def test_role_elo_updates_once_per_valid_game_and_keeps_anchor_fixed():
    from agentbench_frame.arena.rating import RoleEloLedger

    ledger = RoleEloLedger(initial_rating=1500.0, k_factor=32.0)
    record = ledger.update_game(
        role="rollman",
        candidate="hl-run",
        opponent="human-rank01",
        result="win",
        act_id="act-1",
        version_id="v1",
        seed=7,
        anchor_opponent=True,
    )

    assert record.rating_before == pytest.approx(1500.0)
    assert record.rating_after == pytest.approx(1516.0)
    assert ledger.rating("rollman", "human-rank01") == pytest.approx(1500.0)
    assert ledger.rating("rollman", "hl-run") == pytest.approx(1516.0)
    assert len(ledger.history("rollman", "hl-run")) == 1


def test_loss_and_draw_move_candidate_in_correct_directions():
    from agentbench_frame.arena.rating import RoleEloLedger

    ledger = RoleEloLedger(initial_rating=1500.0, k_factor=32.0)
    loss = ledger.update_game(
        role="rollman",
        candidate="hl-run",
        opponent="human",
        result="loss",
        act_id="a1",
        version_id="v1",
        seed=1,
    )
    draw = ledger.update_game(
        role="rollman",
        candidate="hl-run",
        opponent="human",
        result="draw",
        act_id="a2",
        version_id="v2",
        seed=2,
    )

    assert loss.rating_after < loss.rating_before
    assert draw.rating_after > loss.rating_after


def test_invalid_game_does_not_update_elo():
    from agentbench_frame.arena.rating import RoleEloLedger

    ledger = RoleEloLedger()
    with pytest.raises(ValueError):
        ledger.update_game(
            role="rollman",
            candidate="hl-run",
            opponent="human",
            result="infrastructure_error",
            act_id="a1",
            version_id="v1",
            seed=1,
        )
    assert ledger.history("rollman", "hl-run") == ()


def test_series_is_applied_in_declared_order():
    from agentbench_frame.arena.rating import RoleEloLedger

    ledger = RoleEloLedger(initial_rating=1500.0, k_factor=32.0)
    records = ledger.update_series(
        role="rollman",
        candidate="hl-run",
        games=[
            {"opponent": "h1", "result": "win", "act_id": "a1", "version_id": "v1", "seed": 1},
            {"opponent": "h2", "result": "loss", "act_id": "a1", "version_id": "v1", "seed": 2},
            {"opponent": "h3", "result": "draw", "act_id": "a1", "version_id": "v1", "seed": 3},
        ],
        anchor_opponents=True,
    )

    assert [record.opponent for record in records] == ["h1", "h2", "h3"]
    assert [record.game_index for record in records] == [1, 2, 3]


def test_reconstruct_tower_snapshots_applies_delta_removals_between_sparse_samples():
    from agentbench_frame.games.antwar2.policy_probe import (
        _reconstruct_tower_snapshots,
    )

    tower0 = {
        "id": 10,
        "player": 0,
        "pos": {"x": 2, "y": 3},
        "type": 0,
        "cd": 1,
    }
    tower1 = {
        "id": 11,
        "player": 1,
        "pos": {"x": 7, "y": 8},
        "type": 0,
        "cd": 2,
    }
    replay = [
        {"round_state": {"towers": [tower0]}},
        {"round_state": {"towers": []}},
        {"round_state": {"towers": [tower1]}},
        {"round_state": {"towers": [{"id": 10, "type": -1}]}},
        {"round_state": {"towers": []}},
    ]

    snapshots = _reconstruct_tower_snapshots(replay)

    assert snapshots[0] == [tower0]
    assert snapshots[2] == [tower0, tower1]
    assert snapshots[3] == [tower1]
    assert snapshots[4] == [tower1]
    assert all(item["type"] >= 0 for snapshot in snapshots.values() for item in snapshot)

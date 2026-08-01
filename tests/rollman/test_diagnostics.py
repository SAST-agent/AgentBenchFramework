def test_primitive_counterfactuals_cover_exact_rollman_action_support():
    from agentbench_frame.games.rollman.diagnostics import primitive_counterfactuals

    class FakeTracker:
        def reset(self, state):
            self.value = dict(state)

        def step(self, rollman_action, ghosts_action):
            deltas = {
                0: (0, 0),
                1: (1, 0),
                2: (0, -1),
                3: (-1, 0),
                4: (0, 1),
            }
            dx, dy = deltas[rollman_action]
            x, y = self.value["pacman_coord"]
            self.value["pacman_coord"] = [x + dx, y + dy]
            self.value["score"] = [self.value["score"][0] + rollman_action, 0]

        def state(self):
            return self.value

    diagnostics = primitive_counterfactuals(
        state={
            "pacman_coord": [4, 4],
            "ghosts_coord": [[8, 8], [9, 9], [10, 10]],
            "score": [10, 0],
        },
        ghosts_action=(0, 0, 0),
        tracker_factory=FakeTracker,
    )

    assert [item.action for item in diagnostics] == [0, 1, 2, 3, 4]
    assert diagnostics[4].resulting_position == (4, 5)
    assert diagnostics[4].rollman_score_delta == 4


def test_empirical_opponent_order_uses_valid_ghost_results_not_filename_rank():
    from agentbench_frame.games.rollman.diagnostics import calibrate_opponent_difficulty

    order = calibrate_opponent_difficulty(
        (
            {
                "status": "complete",
                "opponent": "rank01",
                "result": "win",
                "rollman_score": 100,
                "ghosts_score": 0,
            },
            {
                "status": "complete",
                "opponent": "rank15",
                "result": "loss",
                "rollman_score": -100,
                "ghosts_score": 200,
            },
            {
                "status": "incomplete",
                "opponent": "rank16",
                "result": "loss",
                "rollman_score": -999,
                "ghosts_score": 999,
            },
        )
    )

    assert order == ("rank15", "rank01")

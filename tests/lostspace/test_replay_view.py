import unittest

from agentbench_frame.lostspace.replay_view import _unshift, render


# A miniature native replay: birthplaces + one round (4 turns) + score_dic.
REPLAY = [
    [[-3, -3, 1], [3, -3, 1], [3, 3, 1], [-3, 3, 1]],
    [
        [{"type": "move", "playerid": 0, "pos": [-3, -2, 1]}],
        [{"type": "getkey", "playerid": 1, "keyid": [2]}],
        [],
        [{"type": "escaped", "playerid": 3}],
    ],
    {"0": 4, "1": 3, "2": 2, "3": 1},
]


def _lines(**kwargs):
    return list(render(REPLAY, **kwargs))


class ReplayViewTest(unittest.TestCase):
    def test_unshift_centres_coordinates(self):
        self.assertEqual(_unshift([-3, -3, 1]), [0, 0, 1])
        self.assertEqual(_unshift([3, 3, 0]), [6, 6, 0])

    def test_header_shows_birthplaces_and_ranking(self):
        lines = _lines()
        joined = "\n".join(lines)
        self.assertIn("birthplaces (grid): [[0, 0, 1], [6, 0, 1], [6, 6, 1], [0, 6, 1]]", joined)
        # score 4=1st: player 0 ranks first
        self.assertIn("final ranking: 1.P0(4pts)", joined)

    def test_actions_unshifted_and_state_tracked(self):
        lines = _lines()
        joined = "\n".join(lines)
        self.assertIn("P0 move -> [0, 1, 1]", joined)
        self.assertIn("P1 getkey [2]", joined)
        self.assertIn("P3 *** ESCAPED ***", joined)
        # state line reflects tracked position / keys / status
        self.assertIn("P0[alive", joined)
        self.assertIn("keys[2]", joined)            # player 1 got key 2
        self.assertIn("P3[escaped", joined)

    def test_player_filter(self):
        lines = _lines(player_filter=0)
        joined = "\n".join(lines)
        self.assertIn("P0 move -> [0, 1, 1]", joined)
        self.assertNotIn("P1 getkey", joined)
        self.assertNotIn("P3 *** ESCAPED ***", joined)

    def test_round_filter_only_shows_one_round(self):
        lines = _lines(round_filter=1)
        self.assertEqual(sum(1 for ln in lines if ln.startswith("--- Round")), 1)


if __name__ == "__main__":
    unittest.main()

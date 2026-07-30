from agentbench_frame.research.rule_complexity import (
    EXPECTED_RULE_GAMES,
    measure_rule_corpus,
)


def test_rule_corpus_has_exact_expected_games():
    report = measure_rule_corpus()

    assert [game["game_id"] for game in report["games_by_id"]] == list(
        EXPECTED_RULE_GAMES
    )
    assert all(game["rule_atoms"] > 0 for game in report["games_by_id"])
    assert all(game["description_sha256"] for game in report["games_by_id"])


def test_deepclue_is_not_packaged_or_measured():
    report = measure_rule_corpus()

    assert "30_deepclue" not in {
        game["game_id"] for game in report["games_by_id"]
    }
    assert report["source"]["deepclue_included"] is False


def test_every_game_has_complete_semantic_boundary_and_provenance():
    report = measure_rule_corpus()

    for game in report["games_by_id"]:
        assert game["source_root"]
        assert game["reviewed"]
        assert game["includes"]
        assert game["excludes"]
        assert game["rule_atoms"] == sum(game["atom_breakdown"].values())
        assert set(game["atom_breakdown"]) == {
            "state",
            "action",
            "observation",
            "setup",
            "condition",
            "transition",
            "outcome",
        }
        assert all(game["atom_breakdown"][kind] > 0 for kind in (
            "state",
            "action",
            "observation",
            "setup",
            "condition",
            "transition",
            "outcome",
        ))

import json
from pathlib import Path


RULE_REPORT_PATH = Path(
    "docs/research/agentbench-rule-complexity-v1.json"
)
EXPECTED_RULE_ATOMS = {
    "23_doto": 206,
    "24_miracle": 203,
    "25_aquawar": 171,
    "25_lostspace": 196,
    "26_snakego": 145,
    "27_antwar": 211,
    "28_generals": 225,
    "29_rollman": 181,
    "30_antwar2": 405,
}
EXPECTED_AST_NODES = {
    "23_doto": 776,
    "24_miracle": 904,
    "25_aquawar": 632,
    "25_lostspace": 747,
    "26_snakego": 599,
    "27_antwar": 953,
    "28_generals": 1078,
    "29_rollman": 676,
    "30_antwar2": 1737,
}


def test_frozen_rule_report_has_exact_nine_game_measurements():
    report = json.loads(RULE_REPORT_PATH.read_text())

    assert report["schema_version"] == "agentbench.rule-complexity.v1"
    assert report["metric"]["language"] == "AB-Rule/1"
    assert report["metric"]["primary"] == "ast_nodes"
    assert report["metric"]["secondary"] == "rule_atoms"
    assert report["source"]["deepclue_included"] is False
    assert {
        game["game_id"]: game["ast_nodes"]
        for game in report["games"]
    } == EXPECTED_AST_NODES
    assert {
        game["game_id"]: game["rule_atoms"]
        for game in report["games"]
    } == EXPECTED_RULE_ATOMS
    assert [game["ast_nodes"] for game in report["games"]] == sorted(
        EXPECTED_AST_NODES.values()
    )
    for game in report["games"]:
        assert game["ast_nodes"] == (
            game["structural_ast_nodes"] + game["expression_ast_nodes"]
        )
        assert game["rule_atoms"] == sum(game["atom_breakdown"].values())
        assert len(game["description_sha256"]) == 64


def test_docs_keep_rule_and_implementation_complexity_separate():
    readme = Path("README.md").read_text()
    implementation_report = Path(
        "docs/research/agentbench-ludi-k-v1.md"
    ).read_text()

    assert "agentbench complexity rules" in readme
    assert "canonical AST nodes" in readme
    assert "rule atoms (RA)" in readme
    assert "agentbench-rule-complexity-v1.md" in implementation_report
    assert "implementation-description complexity" in implementation_report
    assert "canonical AST nodes" in implementation_report
    assert "rule atoms (RA) as a secondary" in implementation_report
    assert "must not be compared numerically" in implementation_report

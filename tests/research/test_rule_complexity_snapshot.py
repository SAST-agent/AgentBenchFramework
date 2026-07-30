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


def test_frozen_rule_report_has_exact_nine_game_measurements():
    report = json.loads(RULE_REPORT_PATH.read_text())

    assert report["schema_version"] == "agentbench.rule-complexity.v1"
    assert report["metric"]["language"] == "AB-Rule/1"
    assert report["metric"]["primary"] == "rule_atoms"
    assert report["source"]["deepclue_included"] is False
    assert {
        game["game_id"]: game["rule_atoms"]
        for game in report["games"]
    } == EXPECTED_RULE_ATOMS
    assert [game["rule_atoms"] for game in report["games"]] == sorted(
        EXPECTED_RULE_ATOMS.values()
    )
    for game in report["games"]:
        assert game["rule_atoms"] == sum(game["atom_breakdown"].values())
        assert len(game["description_sha256"]) == 64


def test_docs_keep_rule_and_implementation_complexity_separate():
    readme = Path("README.md").read_text()
    implementation_report = Path(
        "docs/research/agentbench-ludi-k-v1.md"
    ).read_text()

    assert "agentbench complexity rules" in readme
    assert "rule atoms (RA)" in readme
    assert "agentbench-rule-complexity-v1.md" in implementation_report
    assert "implementation-description complexity" in implementation_report
    assert "must not be compared numerically" in implementation_report

import json
import os

import pytest

from agentbench_frame.cli import main


def test_complexity_rules_cli_writes_nine_game_reports(tmp_path, capsys):
    json_output = tmp_path / "reports/rules.json"
    markdown_output = tmp_path / "reports/rules.md"

    main(
        [
            "complexity",
            "rules",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    report = json.loads(json_output.read_text())
    output = capsys.readouterr().out
    assert len(report["games"]) == 9
    assert report["metric"]["primary"] == "ast_nodes"
    assert report["metric"]["secondary"] == "rule_atoms"
    assert "AB-Rule/1 canonical AST complexity" in output
    assert output.count(" AST  ") == 9
    assert output.count(" RA") == 9
    assert "AST nodes" in markdown_output.read_text()
    assert "Rule atoms" in markdown_output.read_text()
    assert str(json_output) in output
    assert str(markdown_output) in output


def test_complexity_rules_cli_rejects_same_output_path(tmp_path):
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="different paths"):
        main(
            [
                "complexity",
                "rules",
                "--json-output",
                str(output),
                "--markdown-output",
                str(output),
            ]
        )


def test_complexity_rules_cli_rejects_output_aliases(tmp_path):
    output = tmp_path / "report"
    output.write_text("existing")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(output)
    hardlink = tmp_path / "hardlink"
    os.link(output, hardlink)

    for alias in (symlink, hardlink):
        with pytest.raises(ValueError, match="different paths"):
            main(
                [
                    "complexity",
                    "rules",
                    "--json-output",
                    str(output),
                    "--markdown-output",
                    str(alias),
                ]
            )

import json
import subprocess
from pathlib import Path

from agentbench_frame.cli import main
from agentbench_frame.research.agentbench_catalog import AGENTBENCH_GAME_SPECS


def _make_agentbench_checkout(root: Path) -> None:
    for index, spec in enumerate(AGENTBENCH_GAME_SPECS):
        source_root = root / spec.source_root
        source_root.mkdir(parents=True)
        (source_root / "rules.py").write_text(
            f"GAME = {spec.game_id!r}\nRULE = {index}\n"
        )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=AgentBench Test",
            "-c",
            "user.email=agentbench-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def test_complexity_ludi_cli_writes_ten_game_reports(tmp_path, capsys):
    source = tmp_path / "AgentBench"
    source.mkdir()
    _make_agentbench_checkout(source)
    json_output = tmp_path / "reports/result.json"
    markdown_output = tmp_path / "reports/result.md"

    main(
        [
            "complexity",
            "ludi",
            "--agentbench-repo",
            str(source),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    report = json.loads(json_output.read_text())
    output = capsys.readouterr().out
    assert len(report["games"]) == 10
    assert markdown_output.is_file()
    assert "AB-Ludi/1 complexity upper bounds" in output
    assert output.count("bits  ") == 10
    assert str(json_output) in output
    assert str(markdown_output) in output

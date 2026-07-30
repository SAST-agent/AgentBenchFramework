import json
import os

import pytest

from agentbench_frame.cli import main
from agentbench_frame.research.agentbench_catalog import AGENTBENCH_GAME_SPECS
from agentbench_frame.research.ludi_k import REFERENCE_MACHINE_ID


def _report() -> dict:
    games = [
        {
            "game_id": spec.game_id,
            "title": spec.title,
            "source_root": spec.source_root,
            "module_count": len(spec.files),
            "source_bytes": 100 + index,
            "source_bits": (100 + index) * 8,
            "canonical_bytes": 120 + index,
            "canonical_bits": (120 + index) * 8,
            "compressed_bytes": 50 + index,
            "k_upper_bits": (50 + index) * 8,
            "compression_ratio": 0.5,
            "source_sha256": "a" * 64,
            "description_sha256": "b" * 64,
            "files": [],
        }
        for index, spec in enumerate(AGENTBENCH_GAME_SPECS)
    ]
    return {
        "schema_version": "agentbench.ludi-k.v1",
        "reference_machine": {
            "family": "AB-LUDI/1",
            "id": REFERENCE_MACHINE_ID,
            "metric": "conditional_k_upper_bits",
            "decoder_constant_included": False,
        },
        "source": {
            "repository": "https://github.com/Aoraku/AgentBench",
            "commit": "a" * 40,
        },
        "games": games,
    }


def test_complexity_ludi_cli_writes_ten_game_reports(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "agentbench_frame.research.ludi_k.measure_agentbench_repository",
        lambda _: _report(),
    )
    json_output = tmp_path / "reports/result.json"
    markdown_output = tmp_path / "reports/result.md"

    main(
        [
            "complexity",
            "ludi",
            "--agentbench-repo",
            str(tmp_path / "unused"),
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


def test_complexity_ludi_cli_rejects_same_output_path(tmp_path):
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="different paths"):
        main(
            [
                "complexity",
                "ludi",
                "--agentbench-repo",
                str(tmp_path / "unused"),
                "--json-output",
                str(output),
                "--markdown-output",
                str(output),
            ]
        )


def test_complexity_ludi_cli_rejects_output_symlink_alias(tmp_path):
    output = tmp_path / "report"
    output.write_text("existing")
    alias = tmp_path / "alias"
    alias.symlink_to(output)

    with pytest.raises(ValueError, match="different paths"):
        main(
            [
                "complexity",
                "ludi",
                "--agentbench-repo",
                str(tmp_path / "unused"),
                "--json-output",
                str(output),
                "--markdown-output",
                str(alias),
            ]
        )


def test_complexity_ludi_cli_rejects_output_hardlink_alias(tmp_path):
    output = tmp_path / "report"
    output.write_text("existing")
    alias = tmp_path / "alias"
    os.link(output, alias)

    with pytest.raises(ValueError, match="different paths"):
        main(
            [
                "complexity",
                "ludi",
                "--agentbench-repo",
                str(tmp_path / "unused"),
                "--json-output",
                str(output),
                "--markdown-output",
                str(alias),
            ]
        )

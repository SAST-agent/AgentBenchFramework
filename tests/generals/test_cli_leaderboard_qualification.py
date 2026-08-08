import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser
from agentbench_frame.generals.leaderboard_qualification import (
    QUALIFICATION_RESULT_SCHEMA,
)


def _parser():
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))
    return parser


def _args(tmp_path):
    manifest = tmp_path / "qualification.toml"
    receipt = tmp_path / "receipt.json"
    manifest.write_text("qualification_id = 'test'\n", encoding="utf-8")
    receipt.write_text('{"receipt": "test"}\n', encoding="utf-8")
    output = tmp_path / "result" / "qualification.json"
    return _parser().parse_args(
        [
            "generals",
            "check-leaderboard-qualification",
            "--agentbench-root",
            "/assets",
            "--manifest",
            "/benchmark/pilot.toml",
            "--qualification-manifest",
            str(manifest),
            "--replay-skill",
            "skills/replay/SKILL.md",
            "--receipt",
            str(receipt),
            "--output",
            str(output),
        ]
    )


def test_registers_qualification_without_a_data_directory(tmp_path):
    args = _args(tmp_path)

    assert args.generals_command == "check-leaderboard-qualification"
    assert args.replay_skill == Path("skills/replay/SKILL.md")
    assert not hasattr(args, "data_dir")


def test_module_entrypoint_propagates_gate_failure_exit_code(tmp_path):
    missing = tmp_path / "missing"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentbench_frame.cli",
            "generals",
            "check-leaderboard-qualification",
            "--agentbench-root",
            str(tmp_path),
            "--manifest",
            str(missing),
            "--qualification-manifest",
            str(missing),
            "--replay-skill",
            "missing/SKILL.md",
            "--receipt",
            str(missing),
            "--output",
            str(tmp_path / "output.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Generals asset validation failed" in completed.stdout


def test_gate_rejects_output_inside_frozen_asset_tree(
    tmp_path, monkeypatch, capsys
):
    args = _args(tmp_path)
    args.agentbench_root = tmp_path
    args.output = tmp_path / "qualification-result.json"
    monkeypatch.setattr(cli, "_assets", lambda _args: (object(), object()))

    assert cli.handle(args) == 2
    assert "cannot modify the frozen AgentBench tree" in capsys.readouterr().out
    assert not args.output.exists()


@pytest.mark.parametrize(
    ("status", "qualified", "expected_exit"),
    [
        ("qualified", True, 0),
        ("unqualified", False, 1),
        ("incomplete", False, 1),
        ("invalid", False, 2),
    ],
)
def test_routes_gate_and_writes_hashed_audit_result(
    tmp_path,
    monkeypatch,
    capsys,
    status,
    qualified,
    expected_exit,
):
    args = _args(tmp_path)
    strongest_source = Path("/assets/human")
    config = SimpleNamespace(
        opponents=(SimpleNamespace(opponent_id="strongest-human"),)
    )
    layout = SimpleNamespace(
        engine_hash="e" * 64,
        opponents=(
            SimpleNamespace(
                opponent_id="strongest-human", source=strongest_source
            ),
        ),
    )
    qualification = object()
    receipt = object()
    captured = {}

    monkeypatch.setattr(cli, "_assets", lambda _args: (config, layout))
    monkeypatch.setattr(
        cli,
        "resolve_replay_skill",
        lambda root, skill: captured.update(root=root, skill=skill)
        or SimpleNamespace(sha256="s" * 64),
    )
    monkeypatch.setattr(
        cli,
        "_stable_tree_hash",
        lambda source: captured.update(source=source) or "h" * 64,
    )

    def load_config(path, pilot, **hashes):
        captured.update(path=path, pilot=pilot, hashes=hashes)
        return qualification

    monkeypatch.setattr(
        cli, "load_leaderboard_qualification_config", load_config
    )
    monkeypatch.setattr(
        cli, "load_qualification_receipt", lambda _path: receipt
    )
    monkeypatch.setattr(
        cli,
        "evaluate_leaderboard_qualification",
        lambda observed_config, observed_receipt: SimpleNamespace(
            status=status,
            qualified=qualified,
            to_dict=lambda: {
                "qualification_id": "gate-v1",
                "status": status,
                "qualified": qualified,
            },
        )
        if observed_config is qualification and observed_receipt is receipt
        else pytest.fail("gate inputs changed"),
    )

    assert cli.handle(args) == expected_exit

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload == json.loads(capsys.readouterr().out)
    assert payload["schema"] == QUALIFICATION_RESULT_SCHEMA
    assert payload["status"] == status
    assert payload["manifest_sha256"] == hashlib.sha256(
        args.qualification_manifest.read_bytes()
    ).hexdigest()
    assert payload["receipt_sha256"] == hashlib.sha256(
        args.receipt.read_bytes()
    ).hexdigest()
    assert captured["source"] == strongest_source
    assert captured["hashes"] == {
        "engine_sha256": "e" * 64,
        "opponent_tree_sha256": "h" * 64,
        "replay_skill_sha256": "s" * 64,
    }

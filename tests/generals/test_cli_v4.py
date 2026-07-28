import argparse
from types import SimpleNamespace

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def test_generals_cli_registers_complete_iterate_v4_surface():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)

    args = parser.parse_args(
        [
            "generals",
            "iterate-v4",
            "--agentbench-root",
            "/assets",
            "--manifest",
            "/benchmark/pilot.toml",
            "--learning-manifest",
            "/benchmark/v4-learning.toml",
            "--replay-skill",
            "backend_sources/corpus/28_generals/skills/replay/SKILL.md",
            "--data-dir",
            "/data",
            "--parent-run",
            "/runs/v3",
            "--expected-parent-hash",
            "a" * 64,
            "--codex-executable",
            "codex",
            "--provider-timeout",
            "900",
        ]
    )

    assert args.generals_command == "iterate-v4"
    assert str(args.learning_manifest) == "/benchmark/v4-learning.toml"
    assert str(args.replay_skill).endswith("replay/SKILL.md")
    assert str(args.parent_run) == "/runs/v3"
    assert args.expected_parent_hash == "a" * 64
    assert args.provider_timeout == 900


def test_generals_cli_routes_prior_attempt_only_to_v4(
    monkeypatch,
    capsys,
):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args([
        "generals",
        "iterate-v4",
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--learning-manifest",
        "/benchmark/v4-learning.toml",
        "--replay-skill",
        "skills/replay/SKILL.md",
        "--data-dir",
        "/data",
        "--parent-run",
        "/runs/v3",
        "--expected-parent-hash",
        "a" * 64,
        "--prior-attempt-run",
        "/runs/failed-prompt",
    ])
    captured = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            return SimpleNamespace(
                status="complete",
                run_dir="/runs/v4",
                raw_score=0.0,
                evo_score_1=0.0,
                evo_score_2=0.0,
                evo_score_3=7 / 18,
                evo_score_4=0.5,
                gain_4=0.5,
                global_act_count=5,
                round_act_count=1,
                runnable=True,
            )

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(cli, "GeneralsHLRound4Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert str(captured["prior_attempt_run_dir"]) == "/runs/failed-prompt"
    assert "prior_attempt_run_dir" in captured
    capsys.readouterr()

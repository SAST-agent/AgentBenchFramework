import argparse

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

import argparse
from pathlib import Path

from agentbench_frame.generals.cli import register_parser


def test_registers_resumable_champion_campaign_act():
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(
        [
            "generals",
            "champion-campaign-act",
            "--agentbench-root", "/assets",
            "--manifest", "/assets/pilot.toml",
            "--data-dir", "/data",
            "--campaign-manifest", "/assets/campaign.toml",
            "--qualification-manifest", "/assets/qualification.toml",
            "--replay-skill", "skills/replay/SKILL.md",
            "--decision-space", "benchmark/decision-space-v1.md",
            "--campaign-root", "/campaign",
            "--replicate-id", "replicate-2",
            "--model", "frontier-model",
            "--model-revision", "2026-08-08",
            "--codex-executable", "codex",
            "--provider-timeout", "1800",
        ]
    )

    assert args.generals_command == "champion-campaign-act"
    assert args.replicate_id == "replicate-2"
    assert args.decision_space == Path("benchmark/decision-space-v1.md")
    assert args.model_revision == "2026-08-08"


def test_registers_claude_code_campaign_provider():
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(
        [
            "generals",
            "champion-campaign-act",
            "--agentbench-root", "/assets",
            "--manifest", "/assets/pilot.toml",
            "--data-dir", "/data",
            "--campaign-manifest", "/assets/campaign.toml",
            "--qualification-manifest", "/assets/qualification.toml",
            "--replay-skill", "skills/replay/SKILL.md",
            "--decision-space", "benchmark/decision-space-v1.md",
            "--campaign-root", "/campaign",
            "--replicate-id", "replicate-2",
            "--model", "frontier-model",
            "--model-revision", "2026-08-08",
            "--provider", "claude-code",
            "--claude-executable", "claude",
            "--claude-permission-mode", "acceptEdits",
            "--provider-timeout", "1800",
        ]
    )

    assert args.provider == "claude-code"
    assert args.claude_executable == "claude"

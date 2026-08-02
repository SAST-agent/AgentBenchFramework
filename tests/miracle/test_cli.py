"""Miracle CLI 的唯一用户入口契约。"""

import pytest

from agentbench_frame.miracle import agent_bridge
from agentbench_frame.miracle.cli import build_parser


def test_match_accepts_dynamically_registered_agent_names():
    class LlmV1(agent_bridge.EndRoundAgent):
        name = "llm_v1"

    agent_bridge.AGENTS["llm_v1"] = LlmV1
    parser = build_parser()

    try:
        args = parser.parse_args([
            "match", "--agent0", "llm_v1", "--agent1", "sample",
        ])
    finally:
        agent_bridge.AGENTS.pop("llm_v1")

    assert args.agent0 == "llm_v1"
    assert args.agent1 == "sample"


def test_match_has_one_output_directory(tmp_path):
    parser = build_parser()
    output_dir = tmp_path / "match-output"

    args = parser.parse_args([
        "match",
        "--agent0", "sample",
        "--agent1", "endround",
        "--seed", "23",
        "--output-dir", str(output_dir),
        "--tag", "candidate-v3",
    ])

    assert args.seed == 23
    assert args.output_dir == output_dir
    assert args.tag == "candidate-v3"


def test_cli_only_exposes_four_core_commands():
    parser = build_parser()

    assert set(parser._subparsers._group_actions[0].choices) == {"match", "replay", "ig", "loop"}


def test_ig_has_one_output_directory_and_explicit_versions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "ig", "--trace", str(tmp_path / "x.trace.jsonl"),
        "--old", "endround", "--new", "sample", "--camp", "0",
        "--iteration", "2", "--output-dir", str(tmp_path / "ig"),
    ])

    assert args.old == "endround"
    assert args.new == "sample"
    assert args.iteration == 2
    assert args.output_dir == tmp_path / "ig"


def test_legacy_agent_flag_is_not_an_alias():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["match", "--agent", "sample"])


def test_loop_has_one_config_and_optional_data_directory(tmp_path):
    args = build_parser().parse_args([
        "loop", "--config", str(tmp_path / "loop.toml"),
        "--data-dir", str(tmp_path / "results"),
    ])

    assert args.config == tmp_path / "loop.toml"
    assert args.data_dir == tmp_path / "results"

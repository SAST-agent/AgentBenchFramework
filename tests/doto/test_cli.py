from agentbench_frame.doto.cli import build_parser


def test_only_five_public_subcommands_exist():
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "cmd")

    assert set(action.choices) == {"build", "match", "replay", "ig", "loop"}

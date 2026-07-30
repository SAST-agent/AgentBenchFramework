from __future__ import annotations
import io
from contextlib import redirect_stdout

from agentbench_frame.hl.cli import build_parser


def test_help_advertises_new_flags_not_old():
    p = build_parser()
    buf = io.StringIO()
    p.parse_args(["--help"]) if False else None  # build_parser only; check help text
    actions = {a.option_strings[0] for a in p._actions if a.option_strings}
    assert "--model-key" in actions
    assert "--max-turns" in actions
    assert "--claude-path" not in actions
    assert "--dangerously-skip-permissions" not in actions


def test_model_key_default_none():
    args = build_parser().parse_args(["--logic", "dummy", "--reference", "dummy"])
    assert args.model_key is None
    assert args.max_turns == 6

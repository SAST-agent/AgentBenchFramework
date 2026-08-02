import subprocess
import tomllib
from pathlib import Path

from agentbench_frame.doto.cli import build_parser


ROOT = Path(__file__).parents[2]


def test_documented_doto_commands_use_only_five_entry_choices():
    choices = set(build_parser()._subparsers._group_actions[0].choices)
    assert choices == {"build", "match", "replay", "ig", "loop"}
    for name in ("docs/doto-harness.md", "docs/doto-official-acceptance.md"):
        text = (ROOT / name).read_text()
        commands = [line.split("agentbench_frame.doto ", 1)[1].split()[0]
                    for line in text.splitlines() if "agentbench_frame.doto " in line]
        assert commands and set(commands) <= choices | {"<build|match|replay|ig|loop>"}


def test_example_loop_toml_has_required_sections_and_defaults():
    with (ROOT / "examples/doto-loop.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    assert {"llm", "evaluation", "budget"} <= raw.keys()
    assert raw["llm"]["stream"] is True
    assert raw["llm"]["max_context_tokens"] == 1_000_000
    assert "max_tokens" not in raw["llm"]


def test_runtime_artifacts_are_gitignored():
    paths = ["uv.lock", "agentbench_data/x/main.out", "AgentBenchResults/runs/x/summary.json",
             "tmp/match.zip", "tmp/match.trace.jsonl"]
    completed = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(paths) + "\n",
                               text=True, stdout=subprocess.PIPE, check=False, cwd=ROOT)
    assert set(completed.stdout.splitlines()) == set(paths)

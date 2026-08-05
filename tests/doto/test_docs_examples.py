import subprocess
from pathlib import Path

from agentbench_frame.doto.cli import build_parser


ROOT = Path(__file__).parents[2]


def test_documented_doto_commands_include_atomic_population_entry():
    choices = set(build_parser()._subparsers._group_actions[0].choices)
    assert choices == {
        "build", "population", "evaluate", "match", "replay", "ig",
        "run", "iteration",
    }
    for name in ("docs/doto-harness.md", "docs/doto-official-acceptance.md"):
        text = (ROOT / name).read_text()
        commands = [line.split("agentbench_frame.doto ", 1)[1].split()[0]
                    for line in text.splitlines() if "agentbench_frame.doto " in line]
        assert commands and set(commands) <= choices


def test_docs_name_codex_as_orchestrator_and_four_skills():
    text = (ROOT / "docs/doto-harness.md").read_text(encoding="utf-8")
    assert all(value in text for value in (
        "Codex decides", "DotoResults", "AgentBenchResults projection",
        "doto-benchmark-run", "doto-game-rules", "doto-agent-authoring",
        "doto-replay-reader", "30-cell", "56-cell", "70%",
    ))
    assert "Chat Completions" not in text and "doto loop" not in text


def test_readme_has_no_old_doto_harness_claims():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Codex-orchestrated" in text
    assert "DotoResults" in text
    assert "OpenAI-compatible 多轮迭代闭环" not in text


def test_runtime_artifacts_are_gitignored():
    paths = ["uv.lock", "agentbench_data/x/main.out", "AgentBenchResults/runs/x/summary.json",
             "tmp/match.zip", "tmp/match.trace.jsonl"]
    completed = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(paths) + "\n",
                               text=True, stdout=subprocess.PIPE, check=False, cwd=ROOT)
    assert set(completed.stdout.splitlines()) == set(paths)

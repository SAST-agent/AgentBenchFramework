from pathlib import Path
import re
import shlex


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CURRENT_V6_RUNBOOKS = {
    Path("docs/superpowers/plans/2026-07-29-generals-v6-macro-planner.md"): 1,
    Path("docs/experiments/2026-07-29-generals-v6-protocol.md"): 2,
}
V2_SKILL = (
    "backend_sources/corpus/28_generals/"
    "skills/replay-analysis-v2/SKILL.md"
)
PARENT_HASH = (
    "facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e"
)


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_current_v6_runbook_commands_use_frozen_production_inputs():
    for relative_path, expected_count in CURRENT_V6_RUNBOOKS.items():
        text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        command_blocks = [
            block
            for block in re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
            if re.search(r"\bgenerals (?:iterate|recover)-v6\b", block)
        ]

        assert len(command_blocks) == expected_count, relative_path
        for block in command_blocks:
            arguments = shlex.split(block.replace("\\\n", " "))
            assert _option(arguments, "--replay-skill") == V2_SKILL
            assert _option(arguments, "--expected-parent-hash") == PARENT_HASH
            assert _option(arguments, "--provider-timeout") == "1800"

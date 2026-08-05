from pathlib import Path
import json
import re

import pytest

from agentbench_frame.doto.build import build_candidate
from agentbench_frame.doto.cli import build_parser
from agentbench_frame.doto.replay import summarize_replay


ROOT = Path(__file__).parents[2]
RULES_SKILL = ROOT / "skills/doto-game-rules"
AUTHORING_SKILL = ROOT / "skills/doto-agent-authoring"
REPLAY_SKILL = ROOT / "skills/doto-replay-reader"
WORKFLOW_SKILL = ROOT / "skills/doto-benchmark-run"


def test_rules_match_official_constants():
    text = (RULES_SKILL / "references/rules.md").read_text(encoding="utf-8")
    assert all(value in text for value in ("320 × 320", "6000 frames", "20 FPS", "80 points"))
    assert "seed 11" not in text.lower()
    assert all(source in text for source in (
        "official_server/Arguments.py", "official_server/Maps/0.json",
        "official_server/main.py",
    ))


def test_decision_reference_names_joint_action():
    text = (RULES_SKILL / "references/decision-space.md").read_text(encoding="utf-8")
    assert all(value in text for value in ("move[5]", "shoot[5]", "meteor[5]", "flash[5]"))
    assert all(value in text for value in (
        "observation", "action mask", "terminal", "strict KL",
        "decision_space.py",
    ))


def test_rules_skill_routes_both_references():
    text = (RULES_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: doto-game-rules\n")
    assert "references/rules.md" in text
    assert "references/decision-space.md" in text
    assert (RULES_SKILL / "agents/openai.yaml").is_file()


def test_authoring_owns_only_complete_player_ai():
    text = (AUTHORING_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "complete playerAI.cpp" in text
    assert "void playerAI()" in text
    assert all(value in text for value in ("official_server", "sdk/main.cpp", "sealed test"))
    assert "patch fragment" in text


def test_sdk_example_compiles(tmp_path):
    reference = (AUTHORING_SKILL / "references/sdk-api.md").read_text(encoding="utf-8")
    examples = re.findall(r"```cpp\n(.*?)```", reference, re.DOTALL)
    assert len(examples) == 1
    source = tmp_path / "playerAI.cpp"
    source.write_text(examples[0], encoding="utf-8")
    result = build_candidate(source, tmp_path / "build")
    assert result.exit_code == 0, result.stderr


def test_authoring_reference_covers_safe_policy_contract():
    text = (AUTHORING_SKILL / "references/safe-policy-patterns.md").read_text(encoding="utf-8")
    assert all(value in text for value in (
        "i*2+faction", "absolute coordinates", "(-1,-1)", "finite",
        "cooldown", "persistent state",
    ))


def test_replay_claims_match_parser():
    text = (REPLAY_SKILL / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- fixture-claims\n(.*?)\n-->", text, re.DOTALL)
    assert match is not None
    claims = json.loads(match.group(1))
    summary = summarize_replay(ROOT / "tests/doto/fixtures/real_short_replay.zip")
    assert claims["final_scores"] == list(summary.final_scores)
    assert claims["last_frame"] == summary.last_frame
    assert claims["event_counts"] == summary.event_counts


def test_replay_skill_routes_schema_events_and_diagnosis():
    text = (REPLAY_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "events are replay-only" in text
    for name in ("frame-schema.md", "events.md", "diagnosis.md"):
        assert f"references/{name}" in text
        assert (REPLAY_SKILL / "references" / name).is_file()
    assert "doto-game-rules" in text


def test_workflow_names_atomic_commands():
    text = (WORKFLOW_SKILL / "SKILL.md").read_text(encoding="utf-8")
    for command in (
        "run init", "run status", "iteration begin", "iteration build",
        "iteration evaluate", "iteration compare", "iteration close",
        "run finalize", "run export",
    ):
        assert command in text


def test_workflow_invariants_and_companion_skills():
    text = (WORKFLOW_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "final test exactly once" in text
    assert "Do not edit authoritative result files" in text
    assert "OpenAI API" not in text and "Chat Completions" not in text
    for name in ("doto-game-rules", "doto-agent-authoring", "doto-replay-reader"):
        assert name in text
    for name in ("lifecycle.md", "results.md"):
        assert f"references/{name}" in text
        assert (WORKFLOW_SKILL / "references" / name).is_file()


@pytest.mark.parametrize("command", (
    "run init", "run status", "iteration begin", "iteration build",
    "iteration evaluate", "iteration compare", "iteration close",
    "run finalize", "run export",
))
def test_workflow_atomic_command_examples_reach_help(command):
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args([*command.split(), "--help"])
    assert stopped.value.code == 0

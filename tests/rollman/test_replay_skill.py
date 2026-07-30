import json
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).with_name("fixtures")
SKILL_ROOT = (
    Path(__file__).parents[2]
    / "src/agentbench_frame/games/rollman/assets/replay-skill/rollman-replay"
)
SCRIPT = SKILL_ROOT / "scripts/summarize_replay.py"


def test_replay_script_emits_exact_round_level_and_event_evidence():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURES / "replay_complete.jsonl"),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result["valid"] is True
    assert result["levels"] == [1, 2]
    assert result["round_frames"] == 5
    assert result["final_score"] == [-58, 135]
    assert result["stop_reason"] == "time is up"
    assert result["event_counts"] == {
        "EATEN_BY_GHOST": 1,
        "FINISH_LEVEL": 1,
        "SHIELD_DESTROYED": 1,
        "TIMEOUT": 1,
    }
    assert result["round_gaps"] == [
        {
            "level": 2,
            "after_round": 0,
            "before_round": 400,
            "missing_count": 399,
        }
    ]
    assert result["evidence"][0]["location"] == "level 1 round 2"
    assert result["evidence"][0]["events"] == ["SHIELD_DESTROYED"]
    assert result["evidence"][1]["location"] == "level 1 round 3"
    assert result["score_deltas"][1]["rollman"] == 0
    assert result["score_deltas"][1]["ghosts"] == 10


def test_replay_script_markdown_translates_numbers_without_inventing_intent():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURES / "replay_complete.jsonl"),
            "--format",
            "markdown",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "level 1 round 2: SHIELD_DESTROYED" in completed.stdout
    assert "level 1 round 3: EATEN_BY_GHOST" in completed.stdout
    assert "Final score: Rollman -58, Ghosts 135" in completed.stdout
    assert "level 2 is missing rounds 1-399" in completed.stdout
    assert "target item" not in completed.stdout.lower()
    assert "intended" not in completed.stdout.lower()


def test_replay_script_fails_closed_on_malformed_or_incomplete_input():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURES / "replay_malformed.jsonl"),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "line 2" in completed.stderr

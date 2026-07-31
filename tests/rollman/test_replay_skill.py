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
WINDOW_SCRIPT = SKILL_ROOT / "scripts/inspect_trace_window.py"


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


def test_trace_window_returns_bounded_decision_and_outcome_without_board(
    tmp_path,
):
    trace = tmp_path / "trace.jsonl"
    records = [
        {
            "type": "action",
            "player": 0,
            "action": 4,
            "decision": {
                "action": 4,
                "memory_id": "route:r7",
                "state": {
                    "level": 2,
                    "round": 7,
                    "board": [[0, 1], [1, 0]],
                    "board_size": 2,
                    "pacman_coord": [3, 4],
                    "ghosts_coord": [[5, 6]],
                    "pacman_skill_status": [0, 1, 0, 0, 0],
                    "score": [12, 9],
                    "beannumber": 33,
                    "portal_available": True,
                    "portal_coord": [8, 8],
                },
            },
        },
        {
            "type": "watch",
            "content": {
                "level": 2,
                "round": 8,
                "pacman_step_block": [[3, 4], [3, 5]],
                "pacman_coord": [3, 5],
                "pacman_skills": [0, 1, 0, 0, 0],
                "ghosts_step_block": [[[5, 6], [4, 6]]],
                "ghosts_coord": [[4, 6]],
                "score": [13, 9],
                "events": [1],
                "portal_available": True,
                "StopReason": None,
            },
        },
        {
            "type": "watch",
            "content": {"level": 3, "round": 300, "events": []},
        },
        {
            "type": "watch",
            "content": {"level": 3, "round": 300, "events": []},
        },
    ]
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(WINDOW_SCRIPT),
            str(trace),
            "--level",
            "2",
            "--round",
            "7",
            "--radius",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result["decision_count"] == 1
    assert result["decisions"][0]["action"] == {
        "code": 4,
        "name": "RIGHT",
    }
    assert result["decisions"][0]["pre_state"]["pacman_coord"] == [3, 4]
    assert result["decisions"][0]["outcome"]["round"] == 8
    assert result["decisions"][0]["outcome"]["events"] == [
        "SHIELD_DESTROYED"
    ]
    assert '"board":' not in completed.stdout
    assert len(completed.stdout.encode("utf-8")) < 16 * 1024


def test_trace_window_rejects_unbounded_requests(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(WINDOW_SCRIPT),
            str(trace),
            "--level",
            "1",
            *sum((["--round", str(value)] for value in range(21)), []),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "at most 20" in completed.stderr

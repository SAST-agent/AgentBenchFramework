import json
from pathlib import Path

from agentbench_frame.doto.cli import main
from agentbench_frame.doto.ig import build_ig_curve, compare_policies_on_trace


FIXTURES = Path(__file__).parent / "fixtures"


def make_policy(tmp_path, mode):
    path = tmp_path / f"{mode}.py"
    path.symlink_to(FIXTURES / "scripted_ai.py")
    return path


def one_frame_trace(tmp_path):
    humans = [
        [human_id, 10.0, 175.0, 100, 1, 0, 1, 0, 0, -1, 0]
        for human_id in range(10)
    ]
    rows = [
        {"seq": 0, "kind": "observation", "faction": 0, "frame": 0,
         "payload": {"frame": 0, "map": 0, "faction": 0}},
        {"seq": 1, "kind": "observation", "faction": 0, "frame": 1,
         "payload": {"frame": 1, "humans": humans, "fireballs": [], "meteors": [],
                     "balls": [[20, 20, -1, 0], [30, 30, -1, 1]],
                     "scores": [0, 0], "bonus": [0, 0]}},
    ]
    path = tmp_path / "episode.trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def compare(tmp_path, old_mode, new_mode):
    return compare_policies_on_trace(
        one_frame_trace(tmp_path),
        make_policy(tmp_path, old_mode),
        make_policy(tmp_path, new_mode),
        faction=0,
        iteration=1,
        old_version="old",
        new_version="new",
    )


def test_identical_joint_action_has_zero_strict_kl(tmp_path):
    row = compare(tmp_path, "same", "same-copy")
    assert row["unchanged_ratio"] == 1.0
    assert row["finite_kl_mean"] == 0.0


def test_one_coordinate_change_is_infinite(tmp_path):
    row = compare(tmp_path, "left", "right")
    assert row["infinite_ratio"] == 1.0
    assert row["finite_kl_mean"] is None


def test_crashed_policy_is_missing_not_infinite(tmp_path):
    row = compare(tmp_path, "same", "crash")
    assert row["missing_ratio"] == 1.0
    assert row["decisions"][0]["missing_reason"] == "new_process_exit"


def test_curve_keeps_missing_iteration_and_versions():
    curve = build_ig_curve([], versions={0: "base", 1: "new"})
    assert curve["points"][0]["status"] == "baseline"
    assert curve["points"][1]["status"] == "missing"
    assert curve["points"][1]["version"] == "new"


def test_ig_cli_writes_episode_decisions_and_curve(tmp_path):
    output = tmp_path / "ig"
    assert main([
        "ig", "--trace", str(one_frame_trace(tmp_path)),
        "--old", str(make_policy(tmp_path, "same")),
        "--new", str(make_policy(tmp_path, "left")),
        "--faction", "0", "--iteration", "1",
        "--old-version", "v0", "--new-version", "v1",
        "--output-dir", str(output),
    ]) == 0

    episode = json.loads((output / "iteration-0001" / "episode.ig.json").read_text())
    decisions = (output / "iteration-0001" / "episode.ig.jsonl").read_text().splitlines()
    curve = json.loads((output / "ig_curve.json").read_text())
    assert episode["infinite_ratio"] == 1.0
    assert len(decisions) == 1
    assert curve["points"][1]["version"] == "v1"

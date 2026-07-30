import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


TRANSITIONS = (
    ("v0", "v1"),
    ("v1", "v2"),
    ("v2", "v3"),
    ("v3", "v4"),
    ("v4", "v5"),
    ("v5", "v6"),
)

SENSITIVITY = {
    "0.001": (0.0, 0.0, 6.6, 11.1, 11.1, 8.1),
    "0.01": (0.0, 0.0, 5.4, 8.9, 8.9, 6.5),
    "0.05": (0.0, 0.0, 4.4, 7.1, 7.1, 5.2),
    "0.1": (0.0, 0.0, 3.8, 6.1, 6.1, 4.5),
}


def write_complete_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    transitions = []
    for index, (before, after) in enumerate(TRANSITIONS):
        transitions.append({
            "version_before": before,
            "version_after": after,
            "mean_kl_nats": SENSITIVITY["0.01"][index],
            "coverage": {"complete": 12, "total": 12},
            "sensitivity": {
                epsilon: {
                    "mean_kl_nats": values[index],
                    "coverage": {"complete": 12, "total": 12},
                }
                for epsilon, values in SENSITIVITY.items()
            },
        })
    summary = {
        "run_id": "measurement-1",
        "status": "complete",
        "controlled_reference_policy_kl": {
            "metric": "controlled_reference_policy_kl",
            "primary_epsilon": "0.01",
            "epsilons": ["0.001", "0.01", "0.05", "0.1"],
            "reference_state_count": 12,
            "transitions": transitions,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    events = []
    support = 7
    for decision in (10, 2):
        for seed in (289101, 289202, 289303):
            for seat in (0, 1):
                events.append({
                    "event_type": "action_space_count",
                    "measurement_state_id": (
                        f"seed-{seed}-seat-{seat}-decision-{decision}"
                    ),
                    "seed": seed,
                    "seat": seat,
                    "decision_number": decision,
                    "support_size": str(support),
                    "status": "complete",
                })
                support += 1
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return run_dir


def test_loads_complete_figure_data(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
    )

    data = load_policy_kl_figure_data(write_complete_run(tmp_path))

    assert data.transitions == (
        "v0→v1",
        "v1→v2",
        "v2→v3",
        "v3→v4",
        "v4→v5",
        "v5→v6",
    )
    assert data.primary_kl == (0.0, 0.0, 5.4, 8.9, 8.9, 6.5)
    assert tuple(data.sensitivity) == ("0.001", "0.01", "0.05", "0.1")
    assert len(data.support_states) == 12
    assert data.support_states[0].decision_number == 2
    assert data.support_states[-1].decision_number == 10


def test_rejects_duplicate_support_state(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
    )

    run_dir = write_complete_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    first = events_path.read_text().splitlines()[0]
    events_path.write_text(events_path.read_text() + first + "\n")

    with pytest.raises(ValueError, match="duplicate measurement_state_id"):
        load_policy_kl_figure_data(run_dir)


def test_rejects_missing_transition(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
    )

    run_dir = write_complete_run(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["controlled_reference_policy_kl"]["transitions"].pop()
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="transition order"):
        load_policy_kl_figure_data(run_dir)


def test_rejects_wrong_metric_name(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
    )

    run_dir = write_complete_run(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["controlled_reference_policy_kl"]["metric"] = "information_gain"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="metric"):
        load_policy_kl_figure_data(run_dir)


def test_rejects_non_null_partial_transition_aggregate(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
    )

    run_dir = write_complete_run(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    transition = summary["controlled_reference_policy_kl"]["transitions"][0]
    transition["coverage"] = {"complete": 11, "total": 12}
    transition["mean_kl_nats"] = 0.25
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="coverage"):
        load_policy_kl_figure_data(run_dir)


def test_renders_three_panel_svg_and_300_dpi_png(tmp_path):
    from agentbench_frame.generals.paper_figure import (
        load_policy_kl_figure_data,
        render_policy_kl_three_panel,
    )

    data = load_policy_kl_figure_data(write_complete_run(tmp_path))
    svg_path, png_path = render_policy_kl_three_panel(
        data,
        tmp_path / "figure" / "policy-kl",
    )

    svg = svg_path.read_text()
    png = png_path.read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    assert "Controlled-reference Policy KL over Iterations" in svg
    assert "Epsilon Sensitivity" in svg
    assert "Exact Canonical Support Size" in svg
    assert svg.count("epsilon = ") >= 4
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert width >= 6000
    assert height >= 1800


def test_figure_cli_writes_both_explicit_outputs(tmp_path):
    run_dir = write_complete_run(tmp_path)
    output_prefix = tmp_path / "output" / "paper"
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts/plot_generals_controlled_policy_kl.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert str(output_prefix.with_suffix(".svg")) in completed.stdout
    assert str(output_prefix.with_suffix(".png")) in completed.stdout
    assert output_prefix.with_suffix(".svg").is_file()
    assert output_prefix.with_suffix(".png").is_file()

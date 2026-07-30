from __future__ import annotations
import json
from pathlib import Path

from agentbench_frame.hl import plot_kl_compare


def _write_events(path: Path, kls_by_act):
    """kls_by_act: {act_index: (kl_mean_or_None, failed_bool)}"""
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "version", "act_id": "x-000001",
                            "edit_type": "initial"}) + "\n")
        for i, (kl, failed) in sorted(kls_by_act.items()):
            if failed:
                f.write(json.dumps({"event_type": "version", "act_id": f"x-{i:06d}",
                                    "failure_reason": "api_error"}) + "\n")
            if kl is not None:
                f.write(json.dumps({"event_type": "policy_kl", "act_id": f"x-{i:06d}",
                                    "local_policy_kl_trace": [kl, kl]}) + "\n")
            else:
                f.write(json.dumps({"event_type": "eval", "act_id": f"x-{i:06d}",
                                    "win_rate": None}) + "\n")


def test_load_experiment_two_models(tmp_path):
    exp = tmp_path / "exp1"
    (exp / "glm").mkdir(parents=True)
    (exp / "deepseek").mkdir(parents=True)
    _write_events(exp / "glm" / "events.jsonl", {2: (0.5, False), 3: (0.7, False), 4: (None, False)})
    _write_events(exp / "deepseek" / "events.jsonl", {2: (0.3, False), 3: (0.9, False)})

    series = plot_kl_compare.load_experiment(exp)
    assert set(series) == {"glm", "deepseek"}
    glm = {p.iteration: p.policy_kl_mean for p in series["glm"]}
    assert glm[2] == 0.5 and glm[3] == 0.7
    assert glm[4] is None                      # gap, not 0


def test_failed_act_marked(tmp_path):
    exp = tmp_path / "exp1"
    (exp / "glm").mkdir(parents=True)
    _write_events(exp / "glm" / "events.jsonl", {2: (None, True)})  # failed, no kl
    series = plot_kl_compare.load_experiment(exp)
    pt = [p for p in series["glm"] if p.iteration == 2][0]
    assert pt.failed is True
    assert pt.policy_kl_mean is None


def test_plot_overlay_writes_png(tmp_path):
    from agentbench_frame.hl.plot_curves import IterationPoint
    series = {
        "glm": [IterationPoint(iteration=2, act_id="a", policy_kl_mean=0.5),
                IterationPoint(iteration=3, act_id="b", policy_kl_mean=0.7)],
        "deepseek": [IterationPoint(iteration=2, act_id="a", policy_kl_mean=0.3)],
    }
    out = tmp_path / "kl_overlay.png"
    plot_kl_compare.plot_overlay(series, out)
    assert out.is_file() and out.stat().st_size > 0

"""Tests for hl/report.py — three-group HL report classification (doc Fix-E).

Groups (honest heuristics, documented in the module):
- control_weight_tuning: pure parametrize edits with IG≈0 (manual-RL control).
- hl_main: at least one structural edit with real behavior change.
- rejected_hl: structured edit landed but no eval ever showed win_rate>0, or
  nothing usable (all noop / no versions).
"""
import json
from pathlib import Path

from agentbench_frame.hl.events import HLEventWriter
from agentbench_frame.hl.report import classify_run, generate, GROUPS


def _write_events(path: Path, run_id: str, *, versions, evals, kls):
    w = HLEventWriter(run_id=run_id, path=path)
    for i, et in enumerate(versions):
        w.write("agent_act", act_id=f"{run_id}-{i:05d}")
        w.write("version", act_id=f"{run_id}-{i:05d}",
                version_id=f"v{i}", edit_type=et)
    for i, wr in enumerate(evals):
        w.write("eval", act_id=f"{run_id}-{i:05d}",
                evaluation_status="complete", win_rate=wr)
    for i, kl in enumerate(kls):
        w.write("policy_kl", act_id=f"{run_id}-{i:05d}",
                local_policy_kl_trace=[{"kl": kl, "status": "ok", "reason": None}],
                per_sample_status=["ok"], n_ok=1, n_missing=0,
                missing_reasons=[], kl_mean=kl, epsilon=0.1)
    w.close()
    return path


def test_groups_declared():
    assert set(GROUPS) == {"control_weight_tuning", "hl_main", "rejected_hl"}


def test_classify_control_weight_tuning(tmp_path):
    p = _write_events(tmp_path / "e.jsonl", "ctrl",
                      versions=["parametrize", "parametrize"],
                      evals=[0.0, 0.0], kls=[0.0, 0.0])
    assert classify_run(p) == "control_weight_tuning"


def test_classify_hl_main_with_score_gain(tmp_path):
    p = _write_events(tmp_path / "e.jsonl", "main",
                      versions=["replace", "add_rule"],
                      evals=[0.0, 0.5], kls=[1.2, 0.8])
    assert classify_run(p) == "hl_main"


def test_classify_hl_main_downgraded_to_rejected_when_no_gain(tmp_path):
    p = _write_events(tmp_path / "e.jsonl", "rej",
                      versions=["replace", "parametrize"],
                      evals=[0.0, 0.0], kls=[1.5, 0.9])
    assert classify_run(p) == "rejected_hl"


def test_classify_all_noop_is_rejected(tmp_path):
    p = _write_events(tmp_path / "e.jsonl", "nop",
                      versions=["noop", "noop"], evals=[None, None], kls=[])
    assert classify_run(p) == "rejected_hl"


def test_classify_no_versions_is_rejected(tmp_path):
    p = _write_events(tmp_path / "e.jsonl", "empty", versions=[], evals=[], kls=[])
    assert classify_run(p) == "rejected_hl"


def test_generate_writes_three_group_artifacts(tmp_path):
    exp = tmp_path / "exp"
    for label, cfg in (
        ("ctrl", dict(versions=["parametrize"], evals=[0.0], kls=[0.0])),
        ("main", dict(versions=["replace"], evals=[0.4], kls=[1.1])),
        ("rej", dict(versions=["add_rule"], evals=[0.0], kls=[0.6])),
    ):
        run_dir = exp / label
        run_dir.mkdir(parents=True)
        _write_events(run_dir / "events.jsonl", label, **cfg)
    out = generate(exp)
    assert out == exp / "reports"
    assert (out / "README.md").exists()
    curves = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    assert set(curves) == set(GROUPS)
    assert set(curves["control_weight_tuning"]) == {"ctrl"}
    assert set(curves["hl_main"]) == {"main"}
    assert set(curves["rejected_hl"]) == {"rej"}
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "control_weight_tuning" in readme
    assert "rejected_hl" in readme

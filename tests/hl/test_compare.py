from __future__ import annotations
import json
from pathlib import Path

from agentbench_frame.hl import compare


def test_each_model_writes_isolated_stream(tmp_path, monkeypatch):
    """Two models -> two isolated events.jsonl under <experiment>/<model>/."""
    # Fake the per-model round runner so we don't need the real logic/ν here.
    written = {}

    def fake_run_one(*, label, entry, experiment, acts, codebase_root, **kw):
        d = codebase_root / experiment / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text(
            json.dumps({"event_type": "agent_act", "act_id": f"{label}-000001"}) + "\n",
            encoding="utf-8")
        written[label] = d / "events.jsonl"
        return written[label]

    monkeypatch.setattr(compare, "run_one_round", fake_run_one)

    models = {
        "glm": object(),      # entry value unused by the fake
        "deepseek": object(),
    }
    out = compare.run_experiment(
        models=models, experiment="exp1", acts=2,
        codebase_root=tmp_path, hl_args={})
    assert set(out) == {"glm", "deepseek"}
    assert all(p.parent.name in {"glm", "deepseek"} for p in out.values())
    assert out["glm"] != out["deepseek"]              # isolated streams


def test_one_model_failure_does_not_abort(tmp_path, monkeypatch):
    def fake_run_one(*, label, entry, experiment, acts, codebase_root, **kw):
        if label == "qwen":
            raise RuntimeError("boom")
        d = codebase_root / experiment / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text("{}", encoding="utf-8")
        return d / "events.jsonl"

    monkeypatch.setattr(compare, "run_one_round", fake_run_one)
    out = compare.run_experiment(
        models={"glm": object(), "qwen": object()},
        experiment="exp1", acts=1, codebase_root=tmp_path, hl_args={})
    assert set(out) == {"glm"}                        # qwen failed, skipped

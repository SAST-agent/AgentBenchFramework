import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agentbench_frame.doto.llm_client import PolicyProposal
from agentbench_frame.doto.loop import run_loop
from agentbench_frame.doto.loop_config import LoopConfig
from tests.doto.test_loop_config import write_config


class FakeClient:
    def __init__(self):
        self.calls = 0

    def propose(self, messages):
        self.calls += 1
        source = f"void playerAI() {{ /* iteration {self.calls} */ }}\n"
        return PolicyProposal("changed", source,
                              {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                              .01, {"stream": True}, {"choices": []})


def fake_builder(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = output_dir / "main.out"
    executable.write_text(Path(source).read_text())
    return SimpleNamespace(exit_code=0, executable=executable, seconds=.01,
                           source_hash="hash", to_json=lambda: {})


def fake_match(agent0, agent1, *, seed, output_dir, tag):
    output_dir.mkdir(parents=True, exist_ok=True)
    trace, replay = output_dir / f"{tag}.trace.jsonl", output_dir / f"{tag}.zip"
    rows = [
        {"kind": "observation", "faction": 0, "frame": 1, "payload": {"frame": 1}},
        {"kind": "action", "faction": 0, "frame": 1, "payload": {"move": []}},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows))
    replay.write_bytes(b"fixture")
    return SimpleNamespace(winner=0, scores=(2.0, 1.0), frames=1, duration=.01,
                           terminated_by="normal", errors=(), replay_path=replay,
                           trace_path=trace)


def fake_ig(trace, old, new, **kwargs):
    return {"episode_id": Path(trace).stem, "iteration": kwargs["iteration"],
            "old_version": kwargs["old_version"], "new_version": kwargs["new_version"],
            "counts": {"infinite": 1}, "decisions": [{"finite_kl": None}],
            "unchanged_ratio": 0.0, "infinite_ratio": 1.0,
            "missing_ratio": 0.0, "finite_kl_mean": None}


def test_complete_baseline_update_battle_ig_loop(tmp_path):
    config = LoopConfig.load(write_config(tmp_path))
    config = replace(config, budget=replace(config.budget, max_iterations=1))
    run_dir = run_loop(config, client=FakeClient(), builder=fake_builder,
                       match_runner=fake_match, ig_runner=fake_ig,
                       data_dir=tmp_path / "results", run_id="loop-test")
    baseline = json.loads((run_dir / "iterations/iteration-0000/iteration.json").read_text())
    evolved = json.loads((run_dir / "iterations/iteration-0001/iteration.json").read_text())
    assert baseline["version"] != evolved["version"]
    assert evolved["status"] == "accepted"
    assert (run_dir / "iterations/iteration-0001/llm_request.json").is_file()
    assert (run_dir / "score_curve.json").is_file()
    assert json.loads((run_dir / "ig_curve.json").read_text())["points"][1]["status"] == "measured"

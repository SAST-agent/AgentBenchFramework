import json
from pathlib import Path

from agentbench_frame.games.rollman.research_measurement import (
    RollmanMeasurementRunner,
)
from agentbench_frame.hl.codebase import VersionStore
from agentbench_frame.hl.evaluator import CandidateEvaluation


def _trace(path, states):
    with path.open("w", encoding="utf-8") as handle:
        for index, state in enumerate(states):
            handle.write(
                json.dumps(
                    {
                        "type": "action",
                        "player": 0,
                        "decision": {
                            "state": state,
                            "state_id": f"rollout-{index}",
                            "action": 0,
                            "memory_id": None,
                        },
                    }
                )
                + "\n"
            )


def test_measurement_uses_frozen_origin_contexts_and_separate_occupancy(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("OLD\n", encoding="utf-8")
    store = VersionStore(workspace, tmp_path / "versions")
    old = store.snapshot(parent_version_id=None, act_id="origin")
    (workspace / "ai.py").write_text("NEW\n", encoding="utf-8")
    new = store.snapshot(parent_version_id=old.version_id, act_id="act-1")
    states = [{"level": 1, "round": 0}, {"level": 1, "round": 1}]
    origin_trace = tmp_path / "origin.trace.jsonl"
    new_trace = tmp_path / "new.trace.jsonl"
    _trace(origin_trace, states)
    _trace(new_trace, list(reversed(states)))
    old_eval = CandidateEvaluation(
        status="complete",
        score=0.5,
        matches=({"status": "complete", "trace": str(origin_trace)},),
    )
    new_eval = CandidateEvaluation(
        status="complete",
        score=0.5,
        matches=({"status": "complete", "trace": str(new_trace)},),
    )

    def fake_probe(*, workspace, states, **kwargs):
        is_new = (Path(workspace) / "ai.py").read_text(encoding="utf-8") == "NEW\n"
        return tuple(
            {
                "reference_index": index,
                "state_id": f"visible-{index}",
                "action": 4 if is_new and index == 1 else 0,
                "memory_id": None,
            }
            for index, _ in enumerate(states)
        )

    runner = RollmanMeasurementRunner(
        root=tmp_path / "measurement",
        sdk_root=tmp_path,
        epsilon=0.05,
        probe_runner=fake_probe,
    )
    runner.freeze_reference(old_eval)
    result = runner.measure(
        new_version=new,
        old_version=old,
        version_store=store,
        new_evaluation=new_eval,
        old_evaluation=old_eval,
    )

    assert len(result["local_policy_kl_trace"]) == 2
    assert result["local_policy_kl_trace"][0] == 0
    assert result["local_policy_kl_trace"][1] > 0
    assert result["occupancy_shift"] == 0

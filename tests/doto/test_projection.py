import json

import pytest

from agentbench_frame.doto.projection import export_agentbench_projection
from agentbench_frame.doto.run_store import InvalidTransition


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sealed_run(tmp_path):
    run = tmp_path / "DotoResults/runs/23_doto/codex/r1"
    run.mkdir(parents=True)
    (run / "run.toml").write_text(
        'schema_version = 1\ngame = "23_doto"\nrun_id = "r1"\nagent = "codex"\n'
        'state = "finalized"\ncreated_at = "2026-08-05T00:00:00Z"\n'
        'benchmark_version = "fixture-v1"\ntrain_pool_sha256 = "' + "a" * 64 + '"\n'
        'test_pool_sha256 = "' + "b" * 64 + '"\n', encoding="utf-8"
    )
    _json(run / "summary.json", {
        "schema_version": 1, "run_id": "r1", "agent": "codex", "state": "finalized",
        "defeated_human_policy_count": 17, "test_policy_count": 28,
        "wall_hours": 1.25, "total_steps": 86, "win_rate": 0.7,
    })
    _json(run / "score_curve.json", {"schema_version": 1, "metric": "score", "points": []})
    _json(run / "ig_curve.json", {"schema_version": 1, "metric": "strict_kl", "points": []})
    _json(run / "final-test.json", {"state": "finalized"})
    return run


def test_projection_file_set_and_idempotence(tmp_path):
    sealed = _sealed_run(tmp_path)
    target = tmp_path / "AgentBenchResults"
    result = export_agentbench_projection(sealed, target)
    assert {path.name for path in result.path.iterdir()} == {
        "run.toml", "summary.json", "score_curve.json", "ig_curve.json",
        "doto_results_ref.json",
    }
    assert export_agentbench_projection(sealed, target).path == result.path
    assert json.loads((sealed / "projection.json").read_text())["state"] == "exported"


def test_projection_refuses_different_existing_destination(tmp_path):
    sealed = _sealed_run(tmp_path)
    target = tmp_path / "AgentBenchResults"
    result = export_agentbench_projection(sealed, target)
    _json(result.path / "summary.json", {"wall_hours": 999})
    with pytest.raises(InvalidTransition, match="different projection"):
        export_agentbench_projection(sealed, target)

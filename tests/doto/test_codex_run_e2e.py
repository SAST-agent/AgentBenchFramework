import json
import shutil
import subprocess
from pathlib import Path

from agentbench_frame.doto.evaluation import MatrixSpec, evaluate_matrix
from agentbench_frame.doto.ig import compare_policies_on_trace
from agentbench_frame.doto.lifecycle import (
    begin_iteration,
    close_iteration,
    finalize_run,
    init_run,
    record_build,
    record_evaluation,
    record_ig,
)
from agentbench_frame.doto.population import BuiltPolicy, PopulationBundle
from agentbench_frame.doto.projection import export_agentbench_projection
from agentbench_frame.doto.run_store import hash_directory
from agentbench_frame.doto.scheduler import AdaptiveScheduler, SchedulerConfig
from tests.doto.test_run_store import make_inputs


def _scheduler():
    return AdaptiveScheduler(
        SchedulerConfig(target_cpu=70, lower_cpu=65, upper_cpu=75,
                        max_workers=8, sample_seconds=0.001),
        cpu_sampler=lambda: 70.0,
    )


def _bundle(split, count, executable, bundle_sha256=None):
    policies = tuple(
        BuiltPolicy(f"human-{index:02d}", f"Human {index}", executable, f"{index:064x}")
        for index in range(count)
    )
    return PopulationBundle("fixture-v1", split, 11, (0, 1), policies, bundle_sha256)


def _fixture_matrix(spec, candidate, scheduler, output_dir, fixtures, staging):
    matrix = evaluate_matrix(
        spec, candidate, scheduler, staging,
        server_dir=fixtures / "fake_server", test_only=True,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staging, output_dir)
    for episode in matrix["episodes"]:
        for key in ("replay_path", "trace_path"):
            if episode[key] is not None:
                relative = Path(episode[key]).relative_to(staging)
                episode[key] = str(output_dir / relative)
    return matrix


def _record_formal(run, iteration, bundle, fixtures, staging):
    candidate = run.iteration_dir(iteration) / "build/main.out"
    matrix = _fixture_matrix(
        MatrixSpec.formal_train(bundle), candidate, _scheduler(),
        run.iteration_dir(iteration) / "formal",
        fixtures, staging,
    )
    record_evaluation(run, iteration, matrix)
    return matrix


def test_atomic_workflow_produces_both_targets(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    source, train_root, test_root, skills = make_inputs(tmp_path)
    run = init_run(tmp_path / "DotoResults", "codex", source, train_root, test_root,
                   skills, run_id="r1")

    executable = tmp_path / "candidate"
    executable.write_text(
        "#!/usr/bin/env python3\n" + (fixtures / "fake_ai.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    train_bundle = _bundle("train", 15, fixtures / "fake_ai.py")

    record_build(run, 0, {"status": "ready", "executable": executable})
    _record_formal(run, 0, train_bundle, fixtures, tmp_path / "fixture-train-0")
    record_ig(run, 0, {"status": "missing", "value": None,
                       "reason": "baseline has no parent"})
    close_iteration(run, 0)

    child_source = tmp_path / "child-playerAI.cpp"
    child_source.write_text("// Codex iteration: preserve the verified baseline behavior\n"
                            "void playerAI() {}\n", encoding="utf-8")
    child = begin_iteration(run, 0, child_source,
                            "Codex reviewed the replay and retained the stable action policy")
    record_build(run, 1, {"status": "ready", "executable": executable})
    child_matrix = _record_formal(
        run, 1, train_bundle, fixtures, tmp_path / "fixture-train-1"
    )
    first = child_matrix["episodes"][0]
    ig_row = compare_policies_on_trace(
        Path(first["trace_path"]),
        run.iteration_dir(0) / "build/main.out",
        run.iteration_dir(1) / "build/main.out",
        faction=first["candidate_seat"], iteration=1,
        old_version=json.loads((run.iteration_dir(0) / "iteration.json").read_text())["version"],
        new_version=child["version"],
    )
    run.write_json_atomic(run.iteration_dir(1) / "ig/episodes/e2e.ig.json", ig_row)
    record_ig(run, 1, {
        "status": "computed", "value": 0.0, "reason": None,
        "counts": ig_row["counts"], "decision_count": len(ig_row["decisions"]),
        "episode_count": 1, "episodes": [ig_row["episode_id"]],
    })
    close_iteration(run, 1)

    sealed = _bundle("test", 28, fixtures / "fake_ai.py", hash_directory(test_root))
    def final_evaluator(spec, candidate, scheduler, output_dir, **kwargs):
        return _fixture_matrix(
            spec, candidate, scheduler, output_dir, fixtures, tmp_path / "fixture-final"
        )

    final = finalize_run(run, 1, sealed, _scheduler(), evaluation_runner=final_evaluator)
    assert final["attempted_test_matches"] == 56

    doto_results = Path(__file__).resolve().parents[3] / "DotoResults"
    validator = doto_results / ".venv/bin/doto-results"
    if validator.is_file():
        validation = subprocess.run(
            [str(validator), "validate", str(run.run_dir)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr
        assert json.loads(validation.stdout)["valid"]

    result = export_agentbench_projection(run.run_dir, tmp_path / "AgentBenchResults")
    assert (result.path / "summary.json").is_file()
    assert {path.name for path in result.path.iterdir()} == {
        "run.toml", "summary.json", "score_curve.json", "ig_curve.json",
        "doto_results_ref.json",
    }

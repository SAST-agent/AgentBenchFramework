"""Single command-line surface for the DOTO harness."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import tomllib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .build import DotoBuildError, build_candidate
from .evaluation import MatrixSpec, evaluate_matrix
from .ig import compare_policies_on_trace, write_ig_artifacts
from .lifecycle import (
    begin_iteration,
    close_iteration,
    init_run,
    finalize_run,
    record_build,
    record_evaluation,
    record_ig,
)
from .match import run_match
from .population import (
    audit_population,
    build_training_bundle,
    load_population,
    load_sealed_bundle,
    load_training_bundle,
)
from .projection import export_agentbench_projection
from .replay import iter_replay, summarize_replay
from .run_store import DotoRunStore, InvalidTransition, hash_directory
from .scheduler import AdaptiveScheduler, SchedulerConfig


def _build(args: argparse.Namespace) -> int:
    result = build_candidate(args.player_ai, args.output_dir)
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    return 0 if result.exit_code == 0 else 1


def _population(args: argparse.Namespace) -> int:
    manifest = load_population(args.manifest)
    if args.population_cmd == "verify":
        result = audit_population(args.source_root, manifest)
        success = not result["missing"] and not result["hash_mismatches"]
    elif args.population_cmd == "build-train":
        built = build_training_bundle(manifest, args.source_root, args.output_dir)
        result = built.to_json()
        success = len(built.policies) == 15 and all(row.status == "ready" for row in built.policies)
    else:
        bundle = load_sealed_bundle(Path(f"/proc/self/fd/{args.bundle_fd}"), manifest.benchmark_version)
        result = {
            "benchmark_version": bundle.benchmark_version,
            "split": bundle.split,
            "seed": bundle.seed,
            "seats": list(bundle.seats),
            "policies": [
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "sha256": policy.executable_sha256,
                    "status": "ready",
                }
                for policy in bundle.policies
            ],
        }
        success = len(bundle.policies) == 28
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if success else 1


def _match(args: argparse.Namespace) -> int:
    result = run_match(
        args.agent0,
        args.agent1,
        seed=args.seed,
        output_dir=args.output_dir,
        tag=args.tag,
        frame_timeout=args.frame_timeout,
        server_timeout=args.server_timeout,
        server_dir=args.server_dir,
        test_only=args.test_only,
    )
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if args.split == "train":
        if args.pool is None or args.sealed_bundle_fd is not None:
            raise ValueError("train evaluation requires --pool and forbids --sealed-bundle-fd")
        bundle = load_training_bundle(args.pool)
        spec = MatrixSpec.formal_train(bundle, args.seed)
    else:
        if args.sealed_bundle_fd is None or args.pool is not None:
            raise ValueError("test evaluation requires --sealed-bundle-fd and forbids --pool")
        bundle = load_sealed_bundle(Path(f"/proc/self/fd/{args.sealed_bundle_fd}"))
        spec = MatrixSpec.final_test(bundle, args.seed)
    scheduler = AdaptiveScheduler(SchedulerConfig(
        target_cpu=args.cpu_target,
        lower_cpu=max(0.0, args.cpu_target - 5.0),
        upper_cpu=min(100.0, args.cpu_target + 5.0),
        max_workers=args.workers or max(1, os.cpu_count() or 1),
    ))
    document = evaluate_matrix(
        spec, args.candidate, scheduler, args.output_dir,
        server_dir=args.server_dir, test_only=args.test_only,
    )
    payload = {
        "split": args.split,
        **document["summary"],
        "cpu_history": document["scheduler"]["cpu_history"],
        "worker_history": document["scheduler"]["worker_history"],
    }
    if args.split == "train":
        payload["matrix"] = str(args.output_dir / "matrix.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["completion_rate"] == 1.0 else 1


def _skill_roots(root: Path) -> dict[str, Path]:
    names = (
        "doto-benchmark-run", "doto-game-rules",
        "doto-agent-authoring", "doto-replay-reader",
    )
    return {name: Path(root) / name for name in names}


def _run_command(args: argparse.Namespace) -> int:
    if args.run_cmd == "init":
        store = init_run(
            args.doto_results, args.agent, args.initial_player_ai,
            args.train_bundle, args.test_bundle, _skill_roots(args.skills_root),
            run_id=args.run_id,
        )
        payload = {"run_dir": str(store.run_dir), "run_id": store.run_id,
                   "state": store.state}
    elif args.run_cmd == "finalize":
        store = DotoRunStore.open(args.run_dir)
        with (store.run_dir / "run.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        bundle = load_sealed_bundle(
            Path(f"/proc/self/fd/{args.sealed_bundle_fd}"), metadata["benchmark_version"]
        )
        scheduler = AdaptiveScheduler(SchedulerConfig(
            target_cpu=args.cpu_target,
            lower_cpu=max(0.0, args.cpu_target - 5.0),
            upper_cpu=min(100.0, args.cpu_target + 5.0),
            max_workers=args.workers or max(1, os.cpu_count() or 1),
        ))
        payload = finalize_run(
            store, args.candidate_iteration, bundle, scheduler,
            server_dir=args.server_dir, test_only=args.test_only,
        )
    elif args.run_cmd == "export":
        result = export_agentbench_projection(args.run_dir, args.agentbench_results)
        payload = {"path": str(result.path),
                   "sealed_content_sha256": result.sealed_content_sha256}
    else:
        store = DotoRunStore.open(args.run_dir)
        iterations = []
        for path in sorted((store.run_dir / "iterations").glob("iteration-*/iteration.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            iterations.append({
                "iteration": document["iteration"], "version": document["version"],
                "parent_iteration": document["parent_iteration"], "state": document["state"],
                "build_status": document["build"]["status"], "ig_status": document["ig"]["status"],
            })
        payload = {"run_dir": str(store.run_dir), "run_id": store.run_id,
                   "state": store.state, "iterations": iterations}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _iteration_command(args: argparse.Namespace) -> int:
    store = DotoRunStore.open(args.run_dir)
    if args.iteration_cmd == "begin":
        analysis = args.analysis_file.read_text(encoding="utf-8")
        document = begin_iteration(store, args.parent, args.source, analysis)
        print(json.dumps(document, ensure_ascii=False, indent=2))
        return 0
    if args.iteration_cmd == "evaluate":
        with (store.run_dir / "run.toml").open("rb") as stream:
            run_metadata = tomllib.load(stream)
        if hash_directory(args.pool) != run_metadata["train_pool_sha256"]:
            raise ValueError("training pool identity differs from initialized Run")
        bundle = load_training_bundle(args.pool, run_metadata["benchmark_version"])
        candidate = store.iteration_dir(args.iteration) / "build" / "main.out"
        if not candidate.is_file():
            raise InvalidTransition("iteration evaluate requires a ready candidate build")
        scheduler = AdaptiveScheduler(SchedulerConfig(
            target_cpu=args.cpu_target,
            lower_cpu=max(0.0, args.cpu_target - 5),
            upper_cpu=min(100.0, args.cpu_target + 5),
            max_workers=args.workers or max(1, os.cpu_count() or 1),
        ))
        matrix = evaluate_matrix(
            MatrixSpec.formal_train(bundle), candidate, scheduler,
            store.iteration_dir(args.iteration) / "formal",
            server_dir=args.server_dir, test_only=args.test_only,
        )
        document = record_evaluation(store, args.iteration, matrix)
        print(json.dumps({"iteration": args.iteration, "state": document["state"],
                          **matrix["summary"]}, ensure_ascii=False, indent=2))
        return 0 if matrix["summary"]["completion_rate"] == 1.0 else 1
    if args.iteration_cmd == "compare":
        current = json.loads(_iteration_json(store, args.iteration).read_text(encoding="utf-8"))
        parent_index = current.get("parent_iteration")
        if parent_index is None:
            document = record_ig(store, args.iteration, {
                "status": "missing", "value": None, "reason": "baseline has no parent",
                "counts": {}, "episodes": [],
            })
            print(json.dumps({"iteration": args.iteration, "state": document["state"],
                              "status": "missing", "reason": "baseline has no parent"}, indent=2))
            return 0
        parent = json.loads(_iteration_json(store, parent_index).read_text(encoding="utf-8"))
        matrix_path = store.iteration_dir(args.iteration) / "matrix.json"
        if not matrix_path.is_file():
            raise InvalidTransition("iteration compare requires a recorded evaluation")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        old_executable = store.iteration_dir(parent_index) / "build" / "main.out"
        new_executable = store.iteration_dir(args.iteration) / "build" / "main.out"
        ig_dir = store.iteration_dir(args.iteration) / "ig" / "episodes"
        ig_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for episode in matrix.get("episodes", []):
            task_id = episode["task_id"]
            path = ig_dir / ("".join(char if char.isalnum() or char in "._-" else "-"
                                     for char in task_id) + ".json")
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            trace = store.run_dir / episode["trace_path"]
            try:
                row = compare_policies_on_trace(
                    trace, old_executable, new_executable,
                    faction=int(episode["candidate_seat"]), iteration=args.iteration,
                    old_version=parent["version"], new_version=current["version"],
                    timeout=args.timeout,
                )
            except Exception as error:
                row = {
                    "episode_id": task_id, "iteration": args.iteration,
                    "old_version": parent["version"], "new_version": current["version"],
                    "decisions": [], "counts": {"missing": 1},
                    "missing_reason": f"{type(error).__name__}: {error}",
                }
            store.write_json_atomic(path, row)
            rows.append(row)
        counts: Counter[str] = Counter()
        decision_count = 0
        for row in rows:
            counts.update(row.get("counts", {}))
            decision_count += len(row.get("decisions", []))
        if not rows or counts["missing"]:
            status, value = "missing", None
            reason = "strict KL unavailable because one or more decisions are missing"
        elif counts["infinite"]:
            status, value = "computed", None
            reason = "strict KL is infinite because deterministic action supports differ"
        else:
            status, value, reason = "computed", 0.0, None
        summary = {
            "status": status, "value": value, "reason": reason,
            "counts": dict(counts), "decision_count": decision_count,
            "episode_count": len(rows), "episodes": [row["episode_id"] for row in rows],
        }
        document = record_ig(store, args.iteration, summary)
        print(json.dumps({"iteration": args.iteration, "state": document["state"],
                          **summary}, ensure_ascii=False, indent=2))
        return 0 if status == "computed" else 1
    if args.iteration_cmd == "close":
        document = close_iteration(store, args.iteration)
        print(json.dumps({"iteration": args.iteration, "version": document["version"],
                          "state": document["state"]}, ensure_ascii=False, indent=2))
        return 0
    iteration_dir = store.iteration_dir(args.iteration)
    source = iteration_dir / "playerAI.cpp"
    try:
        with tempfile.TemporaryDirectory(prefix="doto-candidate-build-") as temporary:
            result = build_candidate(source, Path(temporary) / "build")
            status = "ready" if result.exit_code == 0 and result.executable is not None else "failed"
            document = record_build(store, args.iteration, {
                "status": status, "executable": result.executable,
                "stdout": result.stdout, "stderr": result.stderr, "seconds": result.seconds,
                "error": None if status == "ready" else "candidate compilation failed",
            })
    except (DotoBuildError, OSError, ValueError) as error:
        document = record_build(store, args.iteration, {"status": "failed", "error": str(error)})
        print(json.dumps({"iteration": args.iteration, "state": document["state"],
                          "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "iteration": args.iteration, "version": document["version"],
        "state": document["state"], "executable_sha256": document["build"]["executable_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if document["state"] == "candidate_built" else 1


def _iteration_json(store: DotoRunStore, iteration: int) -> Path:
    return store.run_dir / "iterations" / f"iteration-{iteration:04d}" / "iteration.json"


def _replay(args: argparse.Namespace) -> int:
    summary = summarize_replay(args.path)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as stream:
            for frame in iter_replay(args.path):
                for event in frame.events:
                    row = {"frame": frame.frame, "type": event[0], "args": event[1:]}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary.to_json(), ensure_ascii=False, indent=2))
    return 0


def _ig(args: argparse.Namespace) -> int:
    row = compare_policies_on_trace(
        args.trace, args.old, args.new,
        faction=args.faction,
        iteration=args.iteration,
        old_version=args.old_version,
        new_version=args.new_version,
        timeout=args.timeout,
    )
    paths = write_ig_artifacts(row, args.output_dir)
    result = {
        "episode_id": row["episode_id"],
        "unchanged_ratio": row["unchanged_ratio"],
        "infinite_ratio": row["infinite_ratio"],
        "missing_ratio": row["missing_ratio"],
        "finite_kl_mean": row["finite_kl_mean"],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.doto",
        description="Atomic DOTO build, match, replay, evaluation, IG, and Run tools",
    )
    subcommands = parser.add_subparsers(dest="cmd", required=True)
    build = subcommands.add_parser("build", help="compile a native playerAI.cpp")
    build.add_argument("--player-ai", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(func=_build)
    population = subcommands.add_parser("population", help="verify and build human-policy pools")
    population_commands = population.add_subparsers(dest="population_cmd", required=True)
    verify = population_commands.add_parser("verify", help="verify complete-source identities")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.set_defaults(func=_population)
    build_train = population_commands.add_parser("build-train", help="build the public training pool")
    build_train.add_argument("--manifest", type=Path, required=True)
    build_train.add_argument("--source-root", type=Path, required=True)
    build_train.add_argument("--output-dir", type=Path, required=True)
    build_train.set_defaults(func=_population)
    verify_sealed = population_commands.add_parser("verify-sealed", help="verify an evaluator-owned pool")
    verify_sealed.add_argument("--manifest", type=Path, required=True)
    verify_sealed.add_argument("--bundle-fd", type=int, required=True)
    verify_sealed.set_defaults(func=_population)
    evaluate = subcommands.add_parser("evaluate", help="run a complete seat-balanced matrix")
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--pool", type=Path)
    evaluate.add_argument("--sealed-bundle-fd", type=int)
    evaluate.add_argument("--split", choices=("train", "test"), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=11)
    evaluate.add_argument("--cpu-target", type=float, default=70.0)
    evaluate.add_argument("--workers", type=int)
    evaluate.add_argument("--server-dir", type=Path)
    evaluate.add_argument("--test-only", action="store_true")
    evaluate.set_defaults(func=_evaluate)
    run = subcommands.add_parser("run", help="manage an authoritative DOTO Run")
    run_commands = run.add_subparsers(dest="run_cmd", required=True)
    run_init = run_commands.add_parser("init", help="initialize an authoritative Run")
    run_init.add_argument("--agent", required=True)
    run_init.add_argument("--initial-player-ai", type=Path, required=True)
    run_init.add_argument("--doto-results", type=Path, required=True)
    run_init.add_argument("--run-id", required=True)
    run_init.add_argument("--train-bundle", type=Path, required=True)
    run_init.add_argument("--test-bundle", type=Path, required=True)
    run_init.add_argument("--skills-root", type=Path,
                          default=Path(__file__).resolve().parents[3] / "skills")
    run_init.set_defaults(func=_run_command)
    run_status = run_commands.add_parser("status", help="read Run state")
    run_status.add_argument("--run-dir", type=Path, required=True)
    run_status.set_defaults(func=_run_command)
    run_finalize = run_commands.add_parser(
        "finalize", help="lock one candidate and run the sealed 56-match test"
    )
    run_finalize.add_argument("--run-dir", type=Path, required=True)
    run_finalize.add_argument("--candidate-iteration", type=int, required=True)
    run_finalize.add_argument("--sealed-bundle-fd", type=int, required=True)
    run_finalize.add_argument("--cpu-target", type=float, default=70.0)
    run_finalize.add_argument("--workers", type=int)
    run_finalize.add_argument("--server-dir", type=Path)
    run_finalize.add_argument("--test-only", action="store_true")
    run_finalize.set_defaults(func=_run_command)
    run_export = run_commands.add_parser(
        "export", help="write the five-file AgentBenchResults projection"
    )
    run_export.add_argument("--run-dir", type=Path, required=True)
    run_export.add_argument("--agentbench-results", type=Path, required=True)
    run_export.set_defaults(func=_run_command)
    iteration = subcommands.add_parser("iteration", help="manage explicit DOTO iterations")
    iteration_commands = iteration.add_subparsers(dest="iteration_cmd", required=True)
    iteration_begin = iteration_commands.add_parser("begin", help="create a child iteration")
    iteration_begin.add_argument("--run-dir", type=Path, required=True)
    iteration_begin.add_argument("--parent", type=int, required=True)
    iteration_begin.add_argument("--source", type=Path, required=True)
    iteration_begin.add_argument("--analysis-file", type=Path, required=True)
    iteration_begin.set_defaults(func=_iteration_command)
    iteration_build = iteration_commands.add_parser("build", help="compile and attach one iteration")
    iteration_build.add_argument("--run-dir", type=Path, required=True)
    iteration_build.add_argument("--iteration", type=int, required=True)
    iteration_build.set_defaults(func=_iteration_command)
    iteration_evaluate = iteration_commands.add_parser("evaluate", help="run and attach 30 formal matches")
    iteration_evaluate.add_argument("--run-dir", type=Path, required=True)
    iteration_evaluate.add_argument("--iteration", type=int, required=True)
    iteration_evaluate.add_argument("--pool", type=Path, required=True)
    iteration_evaluate.add_argument("--cpu-target", type=float, default=70.0)
    iteration_evaluate.add_argument("--workers", type=int)
    iteration_evaluate.add_argument("--server-dir", type=Path)
    iteration_evaluate.add_argument("--test-only", action="store_true")
    iteration_evaluate.set_defaults(func=_iteration_command)
    iteration_compare = iteration_commands.add_parser("compare", help="record strict KL on formal traces")
    iteration_compare.add_argument("--run-dir", type=Path, required=True)
    iteration_compare.add_argument("--iteration", type=int, required=True)
    iteration_compare.add_argument("--timeout", type=float, default=1.0)
    iteration_compare.set_defaults(func=_iteration_command)
    iteration_close = iteration_commands.add_parser("close", help="seal an iteration and rebuild curves")
    iteration_close.add_argument("--run-dir", type=Path, required=True)
    iteration_close.add_argument("--iteration", type=int, required=True)
    iteration_close.set_defaults(func=_iteration_command)
    match = subcommands.add_parser("match", help="run one official-protocol match")
    match.add_argument("--agent0", type=Path, required=True)
    match.add_argument("--agent1", type=Path, required=True)
    match.add_argument("--seed", type=int, default=11)
    match.add_argument("--output-dir", type=Path, required=True)
    match.add_argument("--tag", required=True)
    match.add_argument("--frame-timeout", type=float, default=180.0)
    match.add_argument("--server-timeout", type=float, default=900.0)
    match.add_argument("--server-dir", type=Path)
    match.add_argument("--test-only", action="store_true")
    match.set_defaults(func=_match)
    replay = subcommands.add_parser("replay", help="parse an official replay ZIP")
    replay.add_argument("--path", type=Path, required=True)
    replay.add_argument("--jsonl", type=Path)
    replay.set_defaults(func=_replay)
    ig = subcommands.add_parser("ig", help="compare native policies on one trace")
    ig.add_argument("--trace", type=Path, required=True)
    ig.add_argument("--old", type=Path, required=True)
    ig.add_argument("--new", type=Path, required=True)
    ig.add_argument("--faction", type=int, choices=(0, 1), required=True)
    ig.add_argument("--iteration", type=int, required=True)
    ig.add_argument("--old-version", required=True)
    ig.add_argument("--new-version", required=True)
    ig.add_argument("--output-dir", type=Path, required=True)
    ig.add_argument("--timeout", type=float, default=1.0)
    ig.set_defaults(func=_ig)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InvalidTransition, ValueError) as error:
        print(json.dumps({"error": str(error), "type": type(error).__name__}, ensure_ascii=False))
        return 2

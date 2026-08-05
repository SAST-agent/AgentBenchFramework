"""Atomic, orchestration-free Run and iteration state transitions."""

from __future__ import annotations

import json
import shutil
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .evaluation import MatrixSpec, evaluate_matrix
from .population import PopulationBundle
from .run_store import (
    DotoRunStore,
    DOTO_SKILL_NAMES,
    InvalidTransition,
    IterationState,
    RunState,
    hash_directory,
    hash_file,
    snapshot_skills,
    utc_now,
)


REQUIRED_SKILLS = set(DOTO_SKILL_NAMES)


def _run_metadata(store: DotoRunStore) -> dict:
    with (store.run_dir / "run.toml").open("rb") as stream:
        return tomllib.load(stream)


def _write_run_metadata(store: DotoRunStore, metadata: dict) -> None:
    ordered = (
        "schema_version", "game", "run_id", "agent", "state", "created_at",
        "benchmark_version", "train_pool_sha256", "test_pool_sha256",
    )
    lines = []
    for key in ordered:
        value = metadata[key]
        lines.append(f"{key} = {value}" if isinstance(value, int) else f"{key} = {json.dumps(value)}")
    store.write_text_atomic(store.run_dir / "run.toml", "\n".join(lines) + "\n")


def _manifest_version(bundle: Path, names: tuple[str, ...]) -> str:
    for name in names:
        path = bundle / name
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            version = str(raw.get("benchmark_version", "")).strip()
            if version:
                return version
    raise ValueError(f"bundle has no benchmark manifest: {bundle}")


def _iteration_path(store: DotoRunStore, iteration: int) -> Path:
    return store.run_dir / "iterations" / f"iteration-{iteration:04d}" / "iteration.json"


def _load_iteration(store: DotoRunStore, iteration: int) -> dict:
    path = _iteration_path(store, iteration)
    if not path.is_file():
        raise InvalidTransition(f"iteration {iteration} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_mutable(document: dict) -> None:
    if document["state"] == IterationState.CLOSED:
        raise InvalidTransition("closed iteration cannot be overwritten")


def init_run(results_root: Path, agent: str, initial_player_ai: Path,
             train_bundle: Path, test_bundle: Path,
             skill_roots: Mapping[str, Path], *, run_id: str) -> DotoRunStore:
    agent = str(agent).strip()
    run_id = str(run_id).strip()
    if not agent or not run_id:
        raise ValueError("agent and run_id must be nonempty")
    source = Path(initial_player_ai)
    if not source.is_file() or not source.read_bytes():
        raise ValueError("initial playerAI.cpp must be a nonempty file")
    if set(skill_roots) != REQUIRED_SKILLS:
        raise ValueError("Run requires exactly the four DOTO Skill packages")
    train_bundle, test_bundle = Path(train_bundle), Path(test_bundle)
    benchmark_version = _manifest_version(train_bundle, ("bundle-manifest.json", "manifest.json"))
    test_version = _manifest_version(test_bundle, ("sealed-manifest.json", "test-manifest.json"))
    if benchmark_version != test_version:
        raise ValueError("training and test bundles use different benchmark versions")

    run_dir = Path(results_root) / "runs" / "23_doto" / agent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    store = DotoRunStore(run_dir)
    train_hash, test_hash = hash_directory(train_bundle), hash_directory(test_bundle)
    metadata = {
        "schema_version": 1, "game": "23_doto", "run_id": run_id, "agent": agent,
        "state": RunState.CREATED, "created_at": utc_now(),
        "benchmark_version": benchmark_version,
        "train_pool_sha256": train_hash, "test_pool_sha256": test_hash,
    }
    _write_run_metadata(store, metadata)

    snapshot_skills(run_dir, skill_roots)
    pools = run_dir / "pools"
    pools.mkdir()
    store.write_json_atomic(pools / "train.json", {
        "benchmark_version": benchmark_version, "split": "train", "sha256": train_hash,
    })
    store.write_json_atomic(pools / "test.json", {
        "benchmark_version": benchmark_version, "split": "test", "sha256": test_hash,
    })

    iteration_dir = store.iteration_dir(0)
    store.write_text_atomic(iteration_dir / "playerAI.cpp", source.read_text(encoding="utf-8"))
    source_sha256 = hash_file(iteration_dir / "playerAI.cpp")
    store.write_json_atomic(iteration_dir / "iteration.json", {
        "schema_version": 1, "iteration": 0, "version": source_sha256[:12],
        "parent_iteration": None, "state": IterationState.OPEN, "analysis": "baseline",
        "source_sha256": source_sha256, "created_at": utc_now(), "closed_at": None,
        "build": {"status": "pending", "executable_sha256": None},
        "ig": {"status": "pending", "reason": None},
    })
    with store.locked():
        store._append_event_unlocked("run_initialized", run_id=run_id, iteration=0)
    return store


def begin_iteration(store: DotoRunStore, parent_iteration: int, source: Path,
                    analysis: str) -> dict:
    analysis = str(analysis).strip()
    if not analysis:
        raise ValueError("iteration analysis must be nonempty")
    source = Path(source)
    if not source.is_file() or not source.read_bytes():
        raise ValueError("iteration source must be a nonempty file")
    with store.locked():
        parent = _load_iteration(store, parent_iteration)
        matrix_path = store.run_dir / f"iterations/iteration-{parent_iteration:04d}/matrix.json"
        matrix = json.loads(matrix_path.read_text()) if matrix_path.is_file() else {}
        if (parent.get("state") != IterationState.CLOSED
                or parent.get("build", {}).get("status") != "ready"
                or matrix.get("summary", {}).get("formal_score") is None):
            raise InvalidTransition("child requires a complete evaluated parent")
        existing = [int(path.name.split("-")[-1]) for path in (store.run_dir / "iterations").glob("iteration-*")]
        iteration = max(existing, default=-1) + 1
        directory = store.iteration_dir(iteration)
        store.write_text_atomic(directory / "playerAI.cpp", source.read_text(encoding="utf-8"))
        source_sha256 = hash_file(directory / "playerAI.cpp")
        document = {
            "schema_version": 1, "iteration": iteration, "version": source_sha256[:12],
            "parent_iteration": parent_iteration, "state": IterationState.OPEN,
            "analysis": analysis, "source_sha256": source_sha256,
            "created_at": utc_now(), "closed_at": None,
            "build": {"status": "pending", "executable_sha256": None},
            "ig": {"status": "pending", "reason": None},
        }
        store.write_json_atomic(directory / "iteration.json", document)
        metadata = _run_metadata(store)
        metadata["state"] = RunState.ITERATING
        _write_run_metadata(store, metadata)
        store._append_event_unlocked("iteration_begun", iteration=iteration,
                                     parent_iteration=parent_iteration)
        return document


def record_build(store: DotoRunStore, iteration: int, result: dict) -> dict:
    with store.locked():
        document = _load_iteration(store, iteration)
        _assert_mutable(document)
        status = str(result.get("status", ""))
        build_dir = store.iteration_dir(iteration) / "build"
        build_dir.mkdir(exist_ok=True)
        if status == "ready":
            executable = Path(result["executable"])
            if not executable.is_file():
                raise ValueError("ready build executable does not exist")
            target = build_dir / "main.out"
            shutil.copy2(executable, target)
            if (executable.parent / "Maps").is_dir():
                shutil.copytree(executable.parent / "Maps", build_dir / "Maps", dirs_exist_ok=True)
            executable_sha256 = hash_file(target)
            document["build"] = {"status": "ready", "executable_sha256": executable_sha256}
            document["state"] = IterationState.BUILT
        elif status == "failed":
            (build_dir / "main.out").unlink(missing_ok=True)
            executable_sha256 = None
            document["build"] = {"status": "failed", "executable_sha256": None}
            document["state"] = IterationState.BUILD_FAILED
        else:
            raise ValueError("build status must be ready or failed")
        store.write_json_atomic(build_dir / "build.json", {
            "status": status, "executable_sha256": executable_sha256,
            "error": result.get("error"), "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""), "seconds": result.get("seconds"),
        })
        store.write_json_atomic(_iteration_path(store, iteration), document)
        store._append_event_unlocked("build_recorded", iteration=iteration, status=status)
        return document


def record_evaluation(store: DotoRunStore, iteration: int, matrix: dict) -> dict:
    with store.locked():
        document = _load_iteration(store, iteration)
        _assert_mutable(document)
        if document.get("build", {}).get("status") != "ready":
            raise InvalidTransition("evaluation requires a ready build")
        matrix = json.loads(json.dumps(matrix, allow_nan=False))
        for episode in matrix.get("episodes", []):
            for key in ("replay_path", "trace_path"):
                raw = episode.get(key)
                if raw is None:
                    continue
                path = Path(raw)
                absolute = path.resolve() if path.is_absolute() else (store.run_dir / path).resolve()
                try:
                    episode[key] = absolute.relative_to(store.run_dir.resolve()).as_posix()
                except ValueError as error:
                    raise ValueError(f"evaluation evidence escapes Run: {raw}") from error
        expected = matrix.get("expected_task_ids", [])
        if len(expected) != 30 or len(set(expected)) != 30:
            raise ValueError("formal training evaluation must declare 30 unique tasks")
        matrix_path = store.iteration_dir(iteration) / "matrix.json"
        if matrix_path.is_file():
            existing = json.loads(matrix_path.read_text(encoding="utf-8"))
            if existing != matrix:
                raise InvalidTransition("recorded evaluation is immutable")
            return document
        summary = matrix.get("summary", {})
        complete = summary.get("completion_rate") == 1.0 and summary.get("formal_score") is not None
        document["state"] = IterationState.EVALUATED if complete else IterationState.INCOMPLETE
        store.write_json_atomic(matrix_path, matrix)
        store.write_json_atomic(_iteration_path(store, iteration), document)
        store._append_event_unlocked("evaluation_recorded", iteration=iteration,
                                     complete=complete)
        return document


def record_ig(store: DotoRunStore, iteration: int, ig: dict) -> dict:
    with store.locked():
        document = _load_iteration(store, iteration)
        _assert_mutable(document)
        status = ig.get("status")
        if status not in ("computed", "missing"):
            raise ValueError("IG status must be computed or missing")
        reason = ig.get("reason")
        if status == "missing" and not str(reason or "").strip():
            raise ValueError("missing strict KL requires an explicit reason")
        document["ig"] = {"status": status, "reason": reason}
        document["state"] = IterationState.IG_RECORDED if status == "computed" else IterationState.IG_MISSING
        store.write_json_atomic(store.iteration_dir(iteration) / "ig" / "summary.json", ig)
        store.write_json_atomic(_iteration_path(store, iteration), document)
        store._append_event_unlocked("ig_recorded", iteration=iteration, status=status)
        return document


def _rebuild_curves_unlocked(store: DotoRunStore) -> None:
    score_points = []
    ig_points = []
    for path in sorted((store.run_dir / "iterations").glob("iteration-*/iteration.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        iteration = int(document["iteration"])
        directory = path.parent
        matrix_path = directory / "matrix.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
        summary = matrix.get("summary", {})
        score = summary.get("formal_score")
        completion_rate = float(summary.get("completion_rate", 0.0) or 0.0)
        score_points.append({
            "iteration": iteration,
            "version": document["version"],
            "parent_iteration": document["parent_iteration"],
            "status": "measured" if score is not None else (
                "build_failed" if document.get("build", {}).get("status") == "failed"
                else "incomplete"
            ),
            "evo": score,
            "completion_rate": completion_rate,
        })
        ig_path = directory / "ig" / "summary.json"
        ig = json.loads(ig_path.read_text(encoding="utf-8")) if ig_path.is_file() else {
            "status": "missing", "value": None, "reason": "strict KL was not recorded"
        }
        ig_points.append({
            "iteration": iteration,
            "version": document["version"],
            "parent_iteration": document["parent_iteration"],
            "status": ig.get("status", "missing"),
            "value": ig.get("value"),
            "reason": ig.get("reason"),
        })
    store.write_json_atomic(store.run_dir / "score_curve.json", {
        "schema_version": 1, "metric": "candidate_score_diff", "points": score_points,
    })
    store.write_json_atomic(store.run_dir / "ig_curve.json", {
        "schema_version": 1, "metric": "strict_kl", "points": ig_points,
    })


def close_iteration(store: DotoRunStore, iteration: int) -> dict:
    with store.locked():
        document = _load_iteration(store, iteration)
        _assert_mutable(document)
        if document["state"] == IterationState.OPEN:
            raise InvalidTransition("open iteration must record a build result before close")
        if document.get("ig", {}).get("status") == "pending":
            reason = (
                "candidate build failed" if document.get("build", {}).get("status") == "failed"
                else "strict KL was not recorded"
            )
            document["ig"] = {"status": "missing", "reason": reason}
            store.write_json_atomic(store.iteration_dir(iteration) / "ig" / "summary.json", {
                "status": "missing", "value": None, "reason": reason,
            })
        document["state"] = IterationState.CLOSED
        document["closed_at"] = utc_now()
        store.write_json_atomic(_iteration_path(store, iteration), document)
        metadata = _run_metadata(store)
        metadata["state"] = RunState.ITERATING
        _write_run_metadata(store, metadata)
        _rebuild_curves_unlocked(store)
        store._append_event_unlocked("iteration_closed", iteration=iteration)
        return document


def _relative_evidence(store: DotoRunStore, episodes: list[dict]) -> list[dict]:
    normalized = json.loads(json.dumps(episodes, allow_nan=False))
    for episode in normalized:
        for key in ("replay_path", "trace_path"):
            raw = episode.get(key)
            if raw is None:
                continue
            path = Path(raw)
            absolute = path.resolve() if path.is_absolute() else (store.run_dir / path).resolve()
            try:
                episode[key] = absolute.relative_to(store.run_dir.resolve()).as_posix()
            except ValueError as error:
                raise ValueError(f"final-test evidence escapes Run: {raw}") from error
    return normalized


def _assert_final_candidate(store: DotoRunStore, final: dict, iteration: int) -> None:
    if final["candidate_iteration"] != iteration:
        raise InvalidTransition("final test selected candidate cannot be changed")
    document = _load_iteration(store, iteration)
    executable = store.run_dir / f"iterations/iteration-{iteration:04d}/build/main.out"
    if (document.get("version") != final["candidate_version"]
            or not executable.is_file()
            or hash_file(executable) != final["candidate_sha256"]):
        raise InvalidTransition("final test selected candidate identity changed")


def _summary_total_steps(store: DotoRunStore, final_attempted: int) -> int:
    total = final_attempted
    for path in (store.run_dir / "iterations").glob("iteration-*/matrix.json"):
        matrix = json.loads(path.read_text(encoding="utf-8"))
        total += int(matrix.get("summary", {}).get("attempted_matches",
                                                   len(matrix.get("episodes", []))))
    return total


def finalize_run(store: DotoRunStore, candidate_iteration: int,
                 sealed_bundle: PopulationBundle, scheduler, *,
                 evaluation_runner=evaluate_matrix, server_dir: Path | None = None,
                 test_only: bool = False) -> dict:
    """Lock one candidate, run/resume the hidden matrix, and seal the Run."""

    final_path = store.run_dir / "final-test.json"
    with store.locked():
        metadata = _run_metadata(store)
        if (sealed_bundle.split != "test" or sealed_bundle.seed != 11
                or sealed_bundle.seats != (0, 1) or len(sealed_bundle.policies) != 28):
            raise ValueError("final test requires the complete 28-policy seed-11 two-seat bundle")
        if sealed_bundle.benchmark_version != metadata["benchmark_version"]:
            raise ValueError("sealed test benchmark version differs from initialized Run")
        if sealed_bundle.bundle_sha256 != metadata["test_pool_sha256"]:
            raise ValueError("sealed test pool identity differs from initialized Run")

        if final_path.is_file():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            _assert_final_candidate(store, final, candidate_iteration)
            if metadata["state"] == RunState.FINALIZED:
                summary = json.loads((store.run_dir / "summary.json").read_text(encoding="utf-8"))
                matrix_path = store.run_dir / "final" / "matrix.json"
                matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
                return {**summary, "attempted_test_matches": int(
                    matrix.get("summary", {}).get("attempted_matches", len(final["episodes"])))
                }
        else:
            if metadata["state"] == RunState.FINALIZED:
                raise InvalidTransition("finalized Run is missing its final-test record")
            document = _load_iteration(store, candidate_iteration)
            executable = store.run_dir / f"iterations/iteration-{candidate_iteration:04d}/build/main.out"
            matrix_path = store.run_dir / f"iterations/iteration-{candidate_iteration:04d}/matrix.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
            if (document.get("state") != IterationState.CLOSED
                    or document.get("build", {}).get("status") != "ready"
                    or matrix.get("summary", {}).get("formal_score") is None
                    or not executable.is_file()):
                raise InvalidTransition("final test requires a closed, completely evaluated candidate")
            spec = MatrixSpec.final_test(sealed_bundle)
            final = {
                "schema_version": 1, "state": "started",
                "candidate_iteration": candidate_iteration,
                "candidate_version": document["version"],
                "candidate_sha256": hash_file(executable),
                "test_pool_sha256": sealed_bundle.bundle_sha256,
                "seed": 11, "seats": [0, 1],
                "expected_task_ids": [task.task_id for task in spec.tasks],
                "episodes": [], "defeated_policy_count": 0,
                "test_policy_count": 28,
            }
            store.write_json_atomic(final_path, final)
            metadata["state"] = RunState.FINAL_TEST_STARTED
            _write_run_metadata(store, metadata)
            store._append_event_unlocked(
                "final_test_started", candidate_iteration=candidate_iteration,
                candidate_sha256=final["candidate_sha256"],
            )

    # The durable selection above survives interruption. Per-cell evaluator artifacts
    # make this call resumable without rerunning completed tasks.
    spec = MatrixSpec.final_test(sealed_bundle)
    candidate = store.run_dir / f"iterations/iteration-{candidate_iteration:04d}/build/main.out"
    matrix = evaluation_runner(
        spec, candidate, scheduler, store.run_dir / "final",
        server_dir=server_dir, test_only=test_only,
    )

    with store.locked():
        persisted = json.loads(final_path.read_text(encoding="utf-8"))
        _assert_final_candidate(store, persisted, candidate_iteration)
        episodes = _relative_evidence(store, matrix.get("episodes", []))
        summary_data = matrix.get("summary", {})
        final = {
            **persisted,
            "state": "finalized",
            "episodes": episodes,
            "defeated_policy_count": int(summary_data.get("defeated_policy_count", 0)),
        }
        store.write_json_atomic(store.run_dir / "final" / "matrix.json", matrix)
        store.write_json_atomic(final_path, final)
        metadata = _run_metadata(store)
        metadata["state"] = RunState.FINALIZED
        _write_run_metadata(store, metadata)
        attempted = int(summary_data.get("attempted_matches", len(episodes)))
        try:
            created = datetime.strptime(metadata["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            ).timestamp()
        except (TypeError, ValueError):
            created = time.time()
        summary = {
            "schema_version": 1,
            "run_id": metadata["run_id"],
            "agent": metadata["agent"],
            "state": "finalized",
            "defeated_human_policy_count": final["defeated_policy_count"],
            "test_policy_count": 28,
            "wall_hours": round(max(0.0, time.time() - created) / 3600, 6),
            "total_steps": _summary_total_steps(store, attempted),
            "win_rate": summary_data.get("win_rate"),
        }
        store.write_json_atomic(store.run_dir / "summary.json", summary)
        store._append_event_unlocked(
            "run_finalized", candidate_iteration=candidate_iteration,
            defeated_policy_count=final["defeated_policy_count"], attempted_matches=attempted,
        )
        return {**summary, "attempted_test_matches": attempted}

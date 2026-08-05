"""Strict deterministic KL status for native DOTO policies on one trace."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assets import official_server_dir
from .decision_space import (
    ActionValidationError,
    FrameObservation,
    action_mask,
    canonicalize_action,
    parse_observation,
)
from .process import ManagedProcess
from .protocol import DotoProtocolError, read_ai_frame, write_ai_observation


def _argv(path: Path) -> list[str]:
    return [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_trace(path: Path, faction: int) -> tuple[dict, list[dict]]:
    init = None
    observations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "observation" or row.get("faction") != faction:
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("frame") == 0:
            init = payload
        elif isinstance(payload.get("frame"), int) and payload["frame"] > 0:
            observations.append(payload)
    if init is None:
        raise ValueError("trace has no initialization observation for faction")
    return init, observations


def _request_action(
    process: ManagedProcess,
    payload: dict,
    timeout: float,
    side: str,
):
    try:
        write_ai_observation(process.stdin, _canonical_json(payload).encode())
        raw = read_ai_frame(process.stdout, timeout, side)
    except (BrokenPipeError, DotoProtocolError) as exc:
        detail = str(exc)
        suffix = "process_exit" if "exited" in detail or process.process.poll() is not None else "timeout"
        return None, f"{side}_{suffix}"
    try:
        decoded = json.loads(raw)
        return canonicalize_action(decoded), None
    except (json.JSONDecodeError, ActionValidationError) as exc:
        return None, f"{side}_invalid_action:{exc}"


def compare_policies_on_trace(
    trace: Path,
    old_executable: Path,
    new_executable: Path,
    *,
    faction: int,
    iteration: int,
    old_version: str,
    new_version: str,
    timeout: float = 1.0,
) -> dict[str, Any]:
    if faction not in (0, 1):
        raise ValueError("faction must be 0 or 1")
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    init, payloads = _read_trace(Path(trace), faction)
    map_data = json.loads((official_server_dir() / "Maps" / "0.json").read_text())
    # Keep a symlink's entry name: policy launchers may legitimately dispatch on
    # argv[0].  ``absolute`` still gives ManagedProcess an unambiguous path
    # without collapsing distinct registered policy entries onto one target.
    old_path, new_path = Path(old_executable).absolute(), Path(new_executable).absolute()
    old = ManagedProcess.start(_argv(old_path), cwd=old_path.parent, label="old")
    new = ManagedProcess.start(_argv(new_path), cwd=new_path.parent, label="new")
    decisions = []
    counts: Counter[str] = Counter()
    try:
        init_bytes = _canonical_json(init).encode()
        write_ai_observation(old.stdin, init_bytes)
        write_ai_observation(new.stdin, init_bytes)
        for payload in payloads:
            parsed = parse_observation(payload, faction, map_data)
            if not isinstance(parsed, FrameObservation):
                continue
            old_action, old_error = _request_action(old, payload, timeout, "old")
            new_action, new_error = _request_action(new, payload, timeout, "new")
            missing_reason = old_error or new_error
            if missing_reason is None:
                old_mask = action_mask(parsed, old_action)
                new_mask = action_mask(parsed, new_action)
                if not old_mask.valid:
                    missing_reason = "old_action_outside_mask"
                elif not new_mask.valid:
                    missing_reason = "new_action_outside_mask"
            if missing_reason is not None:
                status, finite_kl = "missing", None
            elif old_action == new_action:
                status, finite_kl = "unchanged", 0.0
            else:
                status, finite_kl = "infinite", None
            counts[status] += 1
            decisions.append({
                "frame": parsed.frame,
                "faction": faction,
                "observation_hash": hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
                "old_action": old_action.to_json() if old_action else None,
                "new_action": new_action.to_json() if new_action else None,
                "status": status,
                "finite_kl": finite_kl,
                "missing_reason": missing_reason,
            })
    finally:
        old.terminate()
        new.terminate()
    total = len(decisions)
    finite = [row["finite_kl"] for row in decisions if row["finite_kl"] is not None]
    return {
        "episode_id": Path(trace).name.removesuffix(".trace.jsonl"),
        "iteration": iteration,
        "faction": faction,
        "old_version": old_version,
        "new_version": new_version,
        "decisions": decisions,
        "counts": dict(counts),
        "unchanged_ratio": counts["unchanged"] / total if total else None,
        "infinite_ratio": counts["infinite"] / total if total else None,
        "missing_ratio": counts["missing"] / total if total else 1.0,
        "finite_kl_mean": sum(finite) / len(finite) if finite else None,
    }


def build_ig_curve(
    episode_rows: Sequence[dict[str, Any]], versions: Mapping[int, str]
) -> dict[str, Any]:
    points = []
    for iteration in sorted(versions):
        rows = [row for row in episode_rows if int(row["iteration"]) == iteration]
        if iteration == 0:
            points.append({
                "iteration": 0, "version": versions[iteration], "status": "baseline",
                "finite_kl_mean": None, "unchanged_ratio": None,
                "infinite_ratio": None, "missing_ratio": None,
            })
            continue
        if not rows:
            points.append({
                "iteration": iteration, "version": versions[iteration], "status": "missing",
                "finite_kl_mean": None, "unchanged_ratio": None,
                "infinite_ratio": None, "missing_ratio": 1.0,
            })
            continue
        totals: Counter[str] = Counter()
        finite = []
        for row in rows:
            totals.update(row.get("counts", {}))
            finite.extend(
                decision["finite_kl"] for decision in row.get("decisions", [])
                if decision.get("finite_kl") is not None
            )
        denominator = sum(totals.values())
        points.append({
            "iteration": iteration,
            "version": versions[iteration],
            "status": "measured" if denominator else "missing",
            "finite_kl_mean": sum(finite) / len(finite) if finite else None,
            "unchanged_ratio": totals["unchanged"] / denominator if denominator else None,
            "infinite_ratio": totals["infinite"] / denominator if denominator else None,
            "missing_ratio": totals["missing"] / denominator if denominator else 1.0,
        })
    return {
        "metric": "strict_deterministic_kl_status",
        "note": "DOTO has continuous coordinates: equal Dirac actions give KL=0; different actions diverge; invalid comparisons are missing.",
        "points": points,
    }


def write_ig_artifacts(row: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Atomically persist one comparison and its aligned two-version curve."""
    iteration = int(row["iteration"])
    episode_id = str(row["episode_id"])
    iteration_dir = Path(output_dir) / f"iteration-{iteration:04d}"
    episode_path = iteration_dir / f"{episode_id}.ig.json"
    decisions_path = iteration_dir / f"{episode_id}.ig.jsonl"
    curve_path = Path(output_dir) / "ig_curve.json"
    _atomic_write(episode_path, json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    decision_text = "".join(
        json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n"
        for decision in row["decisions"]
    )
    _atomic_write(decisions_path, decision_text)
    curve = build_ig_curve(
        [row], {0: str(row["old_version"]), iteration: str(row["new_version"])}
    )
    _atomic_write(curve_path, json.dumps(curve, ensure_ascii=False, indent=2) + "\n")
    return {"episode": episode_path, "decisions": decisions_path, "curve": curve_path}

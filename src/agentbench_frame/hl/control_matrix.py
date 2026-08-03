"""Provider-free execution of explicit positive-control match matrices."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.codebase import VersionStore
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.game_profile import get_game_profile
from agentbench_frame.hl.local_config import LocalHLConfig
from agentbench_frame.hl.profile_runner import frozen_config


_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")


@dataclasses.dataclass(frozen=True)
class ControlMatrixGroup:
    """One explicit Cartesian product of opponents, roles, and seeds."""

    name: str
    opponent_ids: tuple[str, ...]
    roles: tuple[str, ...]
    seeds: tuple[int, ...]


def _unique_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _unique_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("control matrix seeds must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("control matrix seeds must contain integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError("control matrix seeds must contain unique values")
    return result


def load_control_matrix(path: str | Path) -> tuple[ControlMatrixGroup, ...]:
    """Load a strict matrix; no implicit opponents or seed expansion is allowed."""

    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("control matrix must be a JSON object")
    unknown = sorted(set(value) - {"schema_version", "groups"})
    if unknown:
        raise ValueError(f"unknown control matrix fields: {unknown}")
    if value.get("schema_version") != "1.0":
        raise ValueError("control matrix schema_version must be 1.0")
    raw_groups = value.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("control matrix groups must be a non-empty list")

    groups: list[ControlMatrixGroup] = []
    for raw in raw_groups:
        if not isinstance(raw, Mapping):
            raise ValueError("control matrix group must be an object")
        group_unknown = sorted(
            set(raw) - {"name", "opponent_ids", "roles", "seeds"}
        )
        if group_unknown:
            raise ValueError(
                f"unknown control matrix group fields: {group_unknown}"
            )
        missing = sorted(
            {"name", "opponent_ids", "roles", "seeds"} - set(raw)
        )
        if missing:
            raise ValueError(f"missing control matrix group fields: {missing}")
        name = raw["name"]
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise ValueError("control matrix group name must be a safe lowercase name")
        opponent_ids = _unique_strings(
            raw["opponent_ids"], field="control matrix opponent_ids"
        )
        roles = _unique_strings(raw["roles"], field="control matrix roles")
        if any(role not in {"P0", "P1"} for role in roles):
            raise ValueError("control matrix roles must contain only P0/P1")
        groups.append(
            ControlMatrixGroup(
                name=name,
                opponent_ids=opponent_ids,
                roles=roles,
                seeds=_unique_seeds(raw["seeds"]),
            )
        )
    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        raise ValueError("control matrix group names must be unique")
    return tuple(groups)


def summarize_group(evaluation: CandidateEvaluation) -> dict[str, Any]:
    """Keep valid games separate from packaging/runtime failures."""

    matches = tuple(evaluation.matches)
    complete = tuple(match for match in matches if match.get("status") == "complete")
    results = tuple(str(match.get("result")) for match in complete)
    margins = tuple(
        float(match["dense_margin"])
        for match in complete
        if match.get("dense_margin") is not None
    )
    wins = sum(result == "win" for result in results)
    draws = sum(result == "draw" for result in results)
    losses = sum(result == "loss" for result in results)
    return {
        "status": evaluation.status,
        "attempted": len(matches),
        "valid_games": len(complete),
        "package_failures": len(matches) - len(complete),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate_valid": (
            (wins + 0.5 * draws) / len(complete) if complete else None
        ),
        "mean_dense_margin_valid": (
            sum(margins) / len(margins) if margins else None
        ),
    }


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        _json_native(value), ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_control_matrix(
    config: LocalHLConfig,
    *,
    matrix_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate one imported version without constructing a model provider."""

    if config.run.origin.mode != "imported_version":
        raise ValueError("control-matrix requires origin.mode=imported_version")
    assert config.run.origin.source_run is not None
    assert config.run.origin.source_version is not None
    matrix_source = Path(matrix_path).resolve()
    groups = load_control_matrix(matrix_source)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)

    profile = get_game_profile(config.run.game)
    bindings = profile.build_bindings(config=config, run_root=root)
    evaluator = bindings.evaluator
    if not callable(getattr(evaluator, "evaluate_matrix", None)):
        raise ValueError(
            f"game profile {config.run.game} does not support control matrices"
        )
    versions = VersionStore(root / "workspace", root / "versions")
    source, imported = versions.import_version(
        Path(config.run.origin.source_run) / "versions",
        config.run.origin.source_version,
    )

    results: list[dict[str, Any]] = []
    for group in groups:
        evaluation = evaluator.evaluate_matrix(
            imported,
            opponent_ids=group.opponent_ids,
            roles=group.roles,
            seeds=group.seeds,
            phase=f"control-{group.name}",
        )
        results.append(
            {
                "group": dataclasses.asdict(group),
                "summary": summarize_group(evaluation),
                "error": evaluation.error,
                "matches": list(evaluation.matches),
            }
        )

    payload = {
        "schema_version": "1.0",
        "game": config.run.game,
        "provider_used": False,
        "source_config": str(config.source_path),
        "source_config_sha256": _sha256(config.source_path),
        "matrix_path": str(matrix_source),
        "matrix_sha256": _sha256(matrix_source),
        "source_run": str(Path(config.run.origin.source_run).resolve()),
        "source_version": source.version_id,
        "source_content_hash": source.content_hash,
        "evaluated_version": imported.version_id,
        "evaluated_content_hash": imported.content_hash,
        "frozen_config": frozen_config(config),
        "results": results,
    }
    _atomic_json(root / "control-matrix-results.json", payload)
    return payload

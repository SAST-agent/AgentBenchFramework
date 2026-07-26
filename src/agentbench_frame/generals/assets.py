"""Load and validate the frozen Generals HL pilot assets."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import os
import shutil
import subprocess
import tomllib

from .models import AgentProcessSpec, AssetLayout, OpponentSpec, PilotConfig, ProcessLimits


BENCHMARK_ID = "generals-hl-pilot-v1"
FROZEN_LIMITS = ProcessLimits(10.0, 2.0, 600.0, 65_536, 10_485_760)
EXPECTED_TIERS = ("high", "medium", "low")


class AssetValidationError(ValueError):
    """The pilot manifest or its referenced assets are invalid."""


def _tuple_strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssetValidationError(f"{field} must be an array of strings")
    return tuple(value)


def load_pilot_config(path: Path) -> PilotConfig:
    """Parse the strict, versioned pilot manifest."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AssetValidationError(f"cannot read pilot manifest: {exc}") from exc

    try:
        limits_raw = raw["limits"]
        limits = ProcessLimits(
            float(limits_raw["startup_timeout_s"]),
            float(limits_raw["decision_timeout_s"]),
            float(limits_raw["match_timeout_s"]),
            int(limits_raw["max_packet_bytes"]),
            int(limits_raw["max_artifact_bytes"]),
        )
        opponents = tuple(
            OpponentSpec(
                opponent_id=str(item["opponent_id"]),
                tier=str(item["tier"]),
                language=str(item["language"]),
                source=Path(item["source"]),
                argv=_tuple_strings(item["argv"], "opponent argv"),
                build_argv=_tuple_strings(item.get("build_argv", []), "opponent build_argv"),
                learning=bool(item.get("learning", False)),
            )
            for item in raw["opponents"]
        )
        config = PilotConfig(
            benchmark_id=str(raw["benchmark_id"]),
            engine_path=Path(raw["engine_path"]),
            baseline_path=Path(raw["baseline_path"]),
            evaluation_seeds=tuple(int(item) for item in raw["evaluation_seeds"]),
            learning_seeds=tuple(int(item) for item in raw["learning_seeds"]),
            opponents=opponents,
            limits=limits,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AssetValidationError(f"invalid pilot manifest field: {exc}") from exc

    _validate_config(config)
    return config


def _validate_config(config: PilotConfig) -> None:
    if config.benchmark_id != BENCHMARK_ID:
        raise AssetValidationError(f"benchmark_id must be {BENCHMARK_ID}")
    if config.limits != FROZEN_LIMITS:
        raise AssetValidationError("process limits must equal the frozen values")
    if len(config.opponents) != 3:
        raise AssetValidationError("pilot must contain exactly three opponents")
    tiers = tuple(item.tier for item in config.opponents)
    if tiers != EXPECTED_TIERS or len(set(tiers)) != len(tiers):
        raise AssetValidationError("opponent tiers must be unique and ordered high, medium, low")
    ids = tuple(item.opponent_id for item in config.opponents)
    if len(set(ids)) != len(ids):
        raise AssetValidationError("opponent IDs must be unique")
    if set(config.evaluation_seeds) & set(config.learning_seeds):
        raise AssetValidationError("evaluation and learning seeds must be disjoint")
    if len(config.evaluation_seeds) != 3 or len(config.learning_seeds) != 3:
        raise AssetValidationError("each frozen seed set must contain exactly three seeds")
    if config.opponents[0].learning:
        raise AssetValidationError("high-tier opponent cannot be a learning opponent")
    if tuple(item.learning for item in config.opponents) != (False, True, True):
        raise AssetValidationError("only medium and low opponents must be learning opponents")
    if any(not item.argv for item in config.opponents):
        raise AssetValidationError("every opponent must define argv")


def _resolve_below(root: Path, relative: Path, field: str) -> Path:
    root = root.resolve()
    if relative.is_absolute():
        raise AssetValidationError(f"{field} escapes AgentBench root: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise AssetValidationError(f"{field} escapes AgentBench root: {relative}")
    return resolved


def _stable_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return ""
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_assets(config: PilotConfig, agentbench_root: Path) -> AssetLayout:
    """Resolve every path and reject symlink/path traversal escapes."""
    root = agentbench_root.resolve()
    engine_root = _resolve_below(root, config.engine_path, "engine_path")
    baseline_root = _resolve_below(root, config.baseline_path, "baseline_path")
    opponents = tuple(
        replace(item, source=_resolve_below(root, item.source, f"opponent {item.opponent_id} source"))
        for item in config.opponents
    )
    return AssetLayout(
        root=root,
        engine_root=engine_root,
        baseline_root=baseline_root,
        opponents=opponents,
        engine_hash=_stable_tree_hash(engine_root),
    )


def validate_assets(layout: AssetLayout) -> list[str]:
    """Return stable diagnostics; callers choose whether they are fatal."""
    diagnostics: list[str] = []
    if not (layout.engine_root / "main.py").is_file():
        diagnostics.append("missing official engine main.py")
    if not (layout.engine_root / "main_for_player_test.py").is_file():
        diagnostics.append("missing official player SDK")
    if not layout.engine_hash:
        diagnostics.append("official engine hash is empty")
    for opponent in layout.opponents:
        if not opponent.source.is_dir():
            diagnostics.append(f"missing opponent source: {opponent.opponent_id}")
        elif opponent.language == "python" and not (opponent.source / "main.py").is_file():
            diagnostics.append(f"missing opponent main.py: {opponent.opponent_id}")
        elif opponent.language == "cpp" and not (opponent.source / "makefile").is_file():
            diagnostics.append(f"missing opponent makefile: {opponent.opponent_id}")
    return diagnostics


def require_valid_assets(layout: AssetLayout) -> None:
    diagnostics = validate_assets(layout)
    if diagnostics:
        raise AssetValidationError("; ".join(diagnostics))


def prepare_opponents(
    layout: AssetLayout,
    preparation_root: Path,
    python_executable: Path,
) -> tuple[AgentProcessSpec, ...]:
    """Copy preserved submissions, build the C++ entry, and return local specs."""
    preparation_root.mkdir(parents=True, exist_ok=True)
    prepared: list[AgentProcessSpec] = []
    for opponent in layout.opponents:
        target = preparation_root / opponent.opponent_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(opponent.source, target)
        if opponent.build_argv:
            completed = subprocess.run(
                opponent.build_argv,
                cwd=target,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            (target / "build.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (target / "build.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise AssetValidationError(
                    f"opponent build failed: {opponent.opponent_id} ({completed.returncode})"
                )
        argv = tuple(
            str(python_executable) if value in {"python", "python3"} else value
            for value in opponent.argv
        )
        if opponent.language == "cpp" and not (target / argv[0]).is_file():
            raise AssetValidationError(f"missing built executable: {opponent.opponent_id}")
        prepared.append(
            AgentProcessSpec(
                agent_id=opponent.opponent_id,
                argv=argv,
                cwd=target,
                env={
                    "PYTHONPATH": str(target),
                    "PYTHONUNBUFFERED": "1",
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                },
            )
        )
    return tuple(prepared)

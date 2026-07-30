"""Local filesystem bindings around the portable HL run schema."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from agentbench_frame.hl.config import HLRunConfig


@dataclasses.dataclass(frozen=True)
class LocalPaths:
    agentbench_root: Path
    official_logic_root: Path
    pacman_sdk_root: Path
    human_manifest: Path
    workspace: Path
    runs_root: Path
    opponent_build_root: Path

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        config_dir: Path,
    ) -> "LocalPaths":
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown:
            raise ValueError(f"unknown paths fields: {unknown}")
        if missing:
            raise ValueError(f"missing paths fields: {missing}")
        values = {}
        for name in allowed:
            path = Path(str(raw[name])).expanduser()
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            values[name] = path
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class LocalHLConfig:
    schema_version: str
    run: HLRunConfig
    paths: LocalPaths
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "LocalHLConfig":
        source = Path(path).resolve()
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("HL config must be a mapping")
        unknown = sorted(set(value) - {"schema_version", "run", "paths"})
        if unknown:
            raise ValueError(f"unknown local config fields: {unknown}")
        if str(value.get("schema_version")) != "1.0":
            raise ValueError("local config schema_version must be 1.0")
        if not isinstance(value.get("run"), Mapping):
            raise ValueError("run config must be a mapping")
        if not isinstance(value.get("paths"), Mapping):
            raise ValueError("paths config must be a mapping")
        run = HLRunConfig.from_mapping(value["run"])
        if run.game != "29_rollman":
            raise ValueError("this local harness currently supports 29_rollman")
        return cls(
            schema_version="1.0",
            run=run,
            paths=LocalPaths.from_mapping(
                value["paths"],
                config_dir=source.parents[2],
            ),
            source_path=source,
        )

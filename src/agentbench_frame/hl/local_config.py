"""Local filesystem bindings around the portable HL run schema."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from agentbench_frame.hl.config import HLRunConfig
from agentbench_frame.hl.game_profile import get_game_profile


@dataclasses.dataclass(frozen=True)
class LocalPaths:
    values: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def require(self, name: str) -> Path:
        try:
            return self.values[name]
        except KeyError as exc:
            raise KeyError(f"local path is not configured: {name}") from exc

    def __getattr__(self, name: str) -> Path:
        return self.require(name)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        config_dir: Path,
        required_names: tuple[str, ...],
    ) -> "LocalPaths":
        allowed = set(required_names)
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown:
            raise ValueError(f"unknown paths fields: {unknown}")
        if missing:
            raise ValueError(f"missing paths fields: {missing}")
        values = {}
        for name in allowed:
            raw_path = str(raw[name])
            expanded = os.path.expandvars(raw_path)
            if "$" in expanded:
                raise ValueError(
                    f"unresolved environment variable in paths.{name}: {raw_path}"
                )
            path = Path(expanded).expanduser()
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            values[name] = path
        return cls(values=values)


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
        profile = get_game_profile(run.game)
        if (
            run.origin.mode == "imported_version"
            and run.origin.source_run is not None
        ):
            source_run = Path(run.origin.source_run).expanduser()
            if not source_run.is_absolute():
                source_run = (source.parents[2] / source_run).resolve()
            run = dataclasses.replace(
                run,
                origin=dataclasses.replace(
                    run.origin,
                    source_run=str(source_run),
                ),
            )
        return cls(
            schema_version="1.0",
            run=run,
            paths=LocalPaths.from_mapping(
                value["paths"],
                config_dir=source.parents[2],
                required_names=profile.required_local_paths,
            ),
            source_path=source,
        )

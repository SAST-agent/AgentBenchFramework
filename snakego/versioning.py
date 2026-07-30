"""Strategy versioning: save / load / diff agent versions.

Each version is a standalone ``strategy.py`` that defines::

    def create_agent() -> BaseAgent: ...

This file captures the *complete* strategy logic, so a coding agent can
produce a new version by writing one file, and we can load any historical
version back to play or evaluate it deterministically.

A ``VersionStore`` manages a directory tree::

    store_root/
    └── {strategy_name}/
        └── v{n}/
            ├── strategy.py      # the code
            └── meta.json        # version, parent, created, description, score
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agents import BaseAgent


# --------------------------------------------------------------------------- #
# version record
# --------------------------------------------------------------------------- #


@dataclass
class StrategyVersion:
    """Metadata for one saved strategy version."""
    strategy_name: str
    version: int
    created: float           # unix timestamp
    description: str = ""
    parent_version: Optional[int] = None
    score: Optional[float] = None        # optional: Elo or win-rate
    score_detail: Optional[dict] = None  # optional: full eval results
    code_hash: str = ""                  # sha256 of strategy.py

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyVersion":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def version_id(self) -> str:
        return f"{self.strategy_name}/v{self.version}"


# --------------------------------------------------------------------------- #
# version store
# --------------------------------------------------------------------------- #


class VersionStore:
    """Filesystem-backed store of versioned strategies."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------

    def _strategy_dir(self, name: str) -> Path:
        return self.root / name

    def _version_dir(self, name: str, version: int) -> Path:
        return self._strategy_dir(name) / f"v{version}"

    def _code_path(self, name: str, version: int) -> Path:
        return self._version_dir(name, version) / "strategy.py"

    def _meta_path(self, name: str, version: int) -> Path:
        return self._version_dir(name, version) / "meta.json"

    # -- save ---------------------------------------------------------------

    def save(self, strategy_name: str, code: str,
             description: str = "",
             parent_version: Optional[int] = None,
             score: Optional[float] = None,
             score_detail: Optional[dict] = None) -> StrategyVersion:
        """Save *code* as the next version of *strategy_name*.

        Returns the :class:`StrategyVersion` record.
        """
        version = self._next_version(strategy_name)
        vdir = self._version_dir(strategy_name, version)
        vdir.mkdir(parents=True, exist_ok=True)

        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        meta = StrategyVersion(
            strategy_name=strategy_name,
            version=version,
            created=time.time(),
            description=description,
            parent_version=parent_version,
            score=score,
            score_detail=score_detail,
            code_hash=code_hash,
        )
        self._code_path(strategy_name, version).write_text(code, encoding="utf-8")
        self._meta_path(strategy_name, version).write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    def _next_version(self, name: str) -> int:
        sdir = self._strategy_dir(name)
        if not sdir.exists():
            return 1
        versions = []
        for d in sdir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    pass
        return max(versions) + 1 if versions else 1

    # -- load ---------------------------------------------------------------

    def load_code(self, name: str, version: int) -> str:
        return self._code_path(name, version).read_text(encoding="utf-8")

    def load_meta(self, name: str, version: int) -> StrategyVersion:
        return StrategyVersion.from_dict(
            json.loads(self._meta_path(name, version).read_text(encoding="utf-8"))
        )

    def load_agent(self, name: str, version: int) -> BaseAgent:
        """Import the saved ``strategy.py`` and return a fresh agent.

        The file must define ``create_agent() -> BaseAgent``.
        """
        code = self.load_code(name, version)
        mod_name = f"_snakego_strategy_{name}_v{version}_{abs(hash(code))}"
        spec = importlib.util.spec_from_loader(mod_name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        code_path = self._code_path(name, version)
        mod.__dict__["__file__"] = str(code_path)
        exec(compile(code, str(code_path), "exec"), mod.__dict__)
        if not hasattr(mod, "create_agent"):
            raise ValueError(
                f"{name}/v{version}: strategy.py must define create_agent()"
            )
        agent = mod.create_agent()
        agent.name = f"{name}_v{version}"
        return agent

    # -- list / query -------------------------------------------------------

    def list_strategies(self) -> List[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def list_versions(self, name: str) -> List[StrategyVersion]:
        sdir = self._strategy_dir(name)
        if not sdir.exists():
            return []
        versions = []
        for d in sorted(sdir.iterdir()):
            if d.is_dir() and d.name.startswith("v"):
                mp = d / "meta.json"
                if mp.exists():
                    versions.append(
                        StrategyVersion.from_dict(
                            json.loads(mp.read_text(encoding="utf-8"))
                        )
                    )
        return versions

    def latest_version(self, name: str) -> Optional[int]:
        vs = self.list_versions(name)
        return vs[-1].version if vs else None

    def load_latest_agent(self, name: str) -> Optional[BaseAgent]:
        v = self.latest_version(name)
        if v is None:
            return None
        return self.load_agent(name, v)

    # -- update score -------------------------------------------------------

    def update_score(self, name: str, version: int,
                     score: float, score_detail: Optional[dict] = None) -> None:
        meta = self.load_meta(name, version)
        meta.score = score
        meta.score_detail = score_detail
        self._meta_path(name, version).write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- diff ---------------------------------------------------------------

    def diff(self, name: str, v1: int, v2: int) -> Dict[str, Any]:
        """Compare two versions: metadata + a simple line-level code diff."""
        m1, m2 = self.load_meta(name, v1), self.load_meta(name, v2)
        c1 = self.load_code(name, v1).splitlines()
        c2 = self.load_code(name, v2).splitlines()
        added = len(c2) - len(c1)
        return {
            "v1": m1.to_dict(), "v2": m2.to_dict(),
            "lines_v1": len(c1), "lines_v2": len(c2),
            "delta_lines": added,
            "code_changed": m1.code_hash != m2.code_hash,
        }


__all__ = ["VersionStore", "StrategyVersion"]

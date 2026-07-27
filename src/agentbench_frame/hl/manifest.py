"""
HL codebase manifest — declares the agent's on-disk shape.

Two shapes:
- ``single_file``: the agent is one opaque ``agent.py`` (the existing
  lostspace candidate shape). EditType collapses to ``replace``/``noop``.
- ``package``: a ``rules/`` directory of modules + an ordered rule list in
  ``manifest.toml``. Enables add/reorder/parametrize as first-class diffs
  and clean increment-only enforcement (enforcement itself is deferred).

manifest.toml schema:

    shape = "package"            # or "single_file"
    entrypoint = "agent.py"     # the Saiblo-stdio entry script
    [rules]
    order = ["expand", "attack", "escape", "fallback"]

For ``single_file``, ``[rules]`` is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class HLManifest:
    shape: str            # "single_file" | "package"
    entrypoint: str
    rule_order: List[str] = field(default_factory=list)


def load_manifest(path) -> HLManifest:
    """Load a manifest.toml. Minimal hand-rolled TOML reader (the framework
    has no tomllib dep on 3.10; we only need a flat key + one [rules] table)."""
    import os
    p = os.fspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"manifest not found: {p}")

    shape: Optional[str] = None
    entrypoint: Optional[str] = None
    rule_order: List[str] = []
    in_rules = False
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                in_rules = (line == "[rules]")
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # strip inline comments
            if "#" in val:
                val = val.split("#", 1)[0].strip()
            if in_rules and key == "order":
                # list literal ["a", "b", ...]
                inner = val.strip().strip("[]")
                rule_order = [
                    s.strip().strip('"').strip("'")
                    for s in inner.split(",") if s.strip()
                ]
            elif not in_rules:
                if key == "shape":
                    shape = val.strip('"').strip("'")
                elif key == "entrypoint":
                    entrypoint = val.strip('"').strip("'")

    if shape is None:
        raise ValueError(f"manifest {p} missing 'shape'")
    if entrypoint is None:
        raise ValueError(f"manifest {p} missing 'entrypoint'")
    if shape not in ("single_file", "package"):
        raise ValueError(f"manifest {p} has unknown shape: {shape}")

    return HLManifest(shape=shape, entrypoint=entrypoint,
                      rule_order=list(rule_order))

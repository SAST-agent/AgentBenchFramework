"""
HL adapter — run a versioned candidate through the LostSpace eval contract.

``LostSpaceEvaluator`` consumes a ``candidate_command`` shell string + the
candidates run in a subprocess with cwd at their own directory. The adapter
takes a ``VersionHandle`` (a snapshot in the codebase store) and produces that
(command, cwd) pair, staging a *copy* of the snapshot so the canonical store
is never mutated by an eval run.

Both ``single_file`` and ``package`` shapes work identically here: the
manifest's ``entrypoint`` is the script the harness executes. The shape only
matters for diffing/classification (manifest.py), not for running.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Union

from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.manifest import load_manifest


def stage_candidate(
    version: VersionHandle,
    *,
    store,
    dest,
) -> Path:
    """Copy a snapshot into a fresh staging dir and return its path.

    The staging dir is wiped first so repeated evaluations of the same version
    start clean. The canonical store snapshot is read-only here.
    """
    store = Path(store)
    dest = Path(dest)
    snap = store / version.content_hash
    if not snap.exists():
        raise FileNotFoundError(
            f"snapshot {version.content_hash} not in store {store}"
        )
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for src in snap.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(snap)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(src.read_bytes())
    return dest


def candidate_command(
    version: VersionHandle,
    *,
    store,
    dest,
    python: Union[str, None] = None,
) -> Tuple[List[str], Path]:
    """Return ``(argv, cwd)`` to run the candidate at ``version``.

    Reads the manifest from the staged copy to find ``entrypoint``. Falls back
    to ``agent.py`` if no manifest is present (legacy single-file candidates
    without a manifest still work).
    """
    staged = stage_candidate(version, store=store, dest=dest)
    manifest_path = staged / "manifest.toml"
    entrypoint = "agent.py"
    if manifest_path.exists():
        try:
            m = load_manifest(manifest_path)
            entrypoint = m.entrypoint
        except Exception:
            pass  # fall back to agent.py
    argv = [python or sys.executable, str(staged / entrypoint)]
    return argv, staged

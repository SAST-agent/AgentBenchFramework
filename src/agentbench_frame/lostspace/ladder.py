"""Registry of the 16 ranked human algorithms for LostSpace.

The algorithms live out-of-tree under ``AgentBench/top_algorithms/corpus/
25_lostspace_final_ladder`` (referenced in place — never copied into this
framework). This module turns a ``rank=NAME`` selector into the launch
command the harness can spawn:

* C++ (``make``) algorithms are compiled into a per-rank build cache (so the
  corpus tree stays clean) and launched with that cache as ``cwd`` (the
  binaries read ``mapconf2.map`` from their working directory).
* Python (single ``.py``) algorithms are launched directly with the same
  interpreter the framework uses.
* ``python_zip`` (rank 3, ShowCry) is a directory with ``.npy`` data files
  alongside ``main.py``; it is launched with that directory as ``cwd``.

The build cache defaults to ``<repo>/.cache/ladder`` and is overridable via
``LOSTSPACE_LADDER_CACHE``.
"""

from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The corpus is referenced in place — it lives outside this framework repo.
DEFAULT_CORPUS_ROOT = Path(
    r"E:\HL_Agent\AgentBench\top_algorithms\corpus\25_lostspace_final_ladder"
)
MANIFEST_NAME = "MANIFEST.tsv"


class LadderError(RuntimeError):
    """Raised when a ladder entry cannot be resolved or built."""


@dataclass(frozen=True)
class LadderEntry:
    rank: int
    username: str
    display_name: str
    language: str  # "make" | "python" | "python_zip"
    source_dir: Path  # directory the algorithm runs from / is built in
    entry: str  # file to run (python) or "main" (c++ binary name)


def _corpus_root() -> Path:
    env = os.environ.get("LOSTSPACE_LADDER_CORPUS")
    return Path(env).resolve() if env else DEFAULT_CORPUS_ROOT


def _cache_root() -> Path:
    env = os.environ.get("LOSTSPACE_LADDER_CACHE")
    if env:
        return Path(env).resolve()
    # <this file>/../../../../.cache/ladder  ->  repo root / .cache / ladder
    return Path(__file__).resolve().parents[3] / ".cache" / "ladder"


def _read_manifest(corpus: Path) -> list[dict[str, str]]:
    manifest = corpus / MANIFEST_NAME
    if not manifest.is_file():
        raise LadderError(f"MANIFEST.tsv not found at {manifest}")
    with manifest.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _extracted_dir(corpus: Path, row: dict[str, str]) -> Path | None:
    """Locate the already-extracted directory for a row, if present."""
    extracted = corpus / "extracted"
    if not extracted.is_dir():
        return None
    rank = int(row["rank"])
    prefix = f"rank{rank:02d}__"
    matches = [p for p in extracted.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return matches[0] if matches else None


def _entries(corpus: Path) -> dict[int, LadderEntry]:
    out: dict[int, LadderEntry] = {}
    for row in _read_manifest(corpus):
        rank = int(row["rank"])
        lang = row["language"].strip()
        src = _extracted_dir(corpus, row)
        if lang == "python":
            # single .py lives in archives/, not extracted/
            if src is None:
                src = corpus / "archives" / Path(row["archive"]).name
            entry = LadderEntry(
                rank=rank,
                username=row["username"],
                display_name=row["display_name"],
                language=lang,
                source_dir=src.parent,
                entry=src.name,
            )
        elif lang in ("make", "python_zip"):
            if src is None:
                raise LadderError(
                    f"rank {rank} ({lang}) has no extracted directory under "
                    f"{corpus / 'extracted'}"
                )
            if lang == "python_zip":
                # rank 3: ShowCry_v2 subdir holds main.py + .npy data
                sub = next((p for p in src.iterdir() if p.is_dir()), src)
                main_py = sub / "main.py"
                if not main_py.is_file():
                    main_py = src / "main.py"
                entry = LadderEntry(
                    rank=rank,
                    username=row["username"],
                    display_name=row["display_name"],
                    language=lang,
                    source_dir=main_py.parent,
                    entry=main_py.name,
                )
            else:  # make (C++)
                entry = LadderEntry(
                    rank=rank,
                    username=row["username"],
                    display_name=row["display_name"],
                    language=lang,
                    source_dir=src,
                    entry="main",
                )
        else:
            raise LadderError(f"rank {rank}: unknown language {lang!r}")
        out[rank] = entry
    return out


_CACHE: dict[int, LadderEntry] | None = None


def entries() -> dict[int, LadderEntry]:
    """All 16 ladder entries, keyed by rank."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _entries(_corpus_root())
    return _CACHE


def resolve(selector: str) -> LadderEntry:
    """Resolve a ``rank=NAME`` selector to a LadderEntry.

    ``NAME`` may be a rank number (``rank=1``) or the entry's username
    (``rank=omegafantasy``) or display name (``rank=最终幻想``).
    """
    if "=" in selector:
        _, name = selector.split("=", 1)
    else:
        name = selector
    name = name.strip()
    all_entries = entries()
    if name.isdigit():
        rank = int(name)
        if rank in all_entries:
            return all_entries[rank]
    for entry in all_entries.values():
        if name in (entry.username, entry.display_name):
            return entry
    raise LadderError(
        f"no ladder algorithm matches {name!r}; available ranks: "
        f"{sorted(all_entries)}"
    )


def _copy_sources(src: Path, dest: Path) -> None:
    """Copy a C++ algorithm's sources into the build cache (clean build)."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {"Python", "old.cpp", "v18.cpp", "v19.cpp", "ver9.cpp"}:
            continue  # stray alternate versions some archives ship
        target = dest / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def build(entry: LadderEntry) -> Path:
    """Build a C++ algorithm into the cache; return the binary path.

    Python algorithms are returned unchanged (no build step).
    """
    if entry.language != "make":
        return entry.source_dir / entry.entry
    cache = _cache_root() / f"rank{entry.rank:02d}__{entry.username}"
    binary = cache / ("main.exe" if sys.platform.startswith("win") else "main")
    if binary.exists():
        return binary
    _copy_sources(entry.source_dir, cache)
    make_exe = os.environ.get("MAKE", "make")
    result = subprocess.run(
        [make_exe],
        cwd=cache,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not binary.exists():
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise LadderError(
            f"rank {entry.rank} ({entry.username}): build failed\n{tail}"
        )
    return binary


def launch_command(entry: LadderEntry) -> tuple[str, str | None]:
    """Return ``(command, cwd)`` for a ladder opponent.

    The command is a shell string suitable for the harness ``_start`` (which
    understands ``cd /d <dir> && <exe>`` on Windows); ``cwd`` is returned
    separately for callers that pass it directly.
    """
    binary = build(entry)
    if entry.language == "make":
        cwd = str(binary.parent)
        exe = binary.name
        return f'cd /d "{cwd}" && {exe}' if sys.platform.startswith("win") else exe, cwd
    # python / python_zip
    cwd = str(entry.source_dir)
    py = sys.executable
    script = str(entry.source_dir / entry.entry)
    return f'"{py}" "{script}"', cwd


def command_for(selector: str) -> str:
    """Convenience: ``rank=NAME`` -> harness-ready launch command string."""
    entry = resolve(selector)
    cmd, _cwd = launch_command(entry)
    return cmd


def all_ranks() -> list[int]:
    return sorted(entries())

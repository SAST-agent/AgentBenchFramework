"""Cross-platform AI entry resolution for the 24_miracle runner.

Resolves the argv list used to launch an AI subprocess via ``subprocess.Popen``
(``shell=False``). The executable is returned as an **ABSOLUTE** path because
Windows ``CreateProcess`` does NOT search the ``cwd=`` argument's directory for a
bare executable name (``Popen(["main.exe"], cwd=dir)`` raises FileNotFoundError);
an absolute path is found regardless of the child's working directory.

Precedence:
  1. ``explicit`` (a list is kept as-is; a string is kept WHOLE as a single
     executable path — never whitespace-split, so paths with spaces survive)
     takes priority over auto-detection.
  2. Auto-detection (deterministic and documented):
       Windows (``os.name == 'nt'``): ``main.exe`` > ``main.py`` > ``main``
       POSIX:                          ``main``     > ``main.py`` > ``main.exe``
     When ``main.exe`` and ``main`` coexist, Windows picks ``main.exe`` and
     POSIX picks ``main`` (recorded behaviour).
     ``main.py`` is launched via ``sys.executable`` + absolute script path.

This never modifies the strategy directory and never reads strategy source.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

#: platform flag read at call time (not the global ``os.name``), so tests can
#: patch it via ``monkeypatch.setattr(entry, "_IS_NT", ...)`` without disturbing
#: pathlib or other stdlib code that reads the real ``os.name``.
_IS_NT = os.name == "nt"


def resolve_ai_command(
    ai_dir: Union[str, Path],
    explicit: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """Return the argv list for the AI subprocess (executable as an ABSOLUTE path).

    Raises ``FileNotFoundError`` (mentioning ``ai_dir``) when no entry exists.
    """
    # 1. explicit takes priority; a string is kept whole (paths with spaces!)
    if explicit is not None and explicit != "" and not (
        isinstance(explicit, (list, tuple)) and len(explicit) == 0
    ):
        if isinstance(explicit, (list, tuple)):
            return [str(x) for x in explicit]
        return [str(explicit)]

    ai_dir = Path(ai_dir).resolve()  # absolute, so Popen(cwd=...) finds the exe on Windows
    is_nt = _IS_NT

    # 2. auto-detection — absolute path to the entry
    if is_nt:
        if (ai_dir / "main.exe").exists():
            return [str(ai_dir / "main.exe")]
        if (ai_dir / "main.py").exists():
            return [sys.executable, str(ai_dir / "main.py")]
        if (ai_dir / "main").exists():           # unusual on Windows (no extension)
            return [str(ai_dir / "main")]
    else:
        if (ai_dir / "main").exists():
            return [str(ai_dir / "main")]
        if (ai_dir / "main.py").exists():
            return [sys.executable, str(ai_dir / "main.py")]
        if (ai_dir / "main.exe").exists():        # cross-mounted POSIX
            return [str(ai_dir / "main.exe")]

    raise FileNotFoundError(
        f"no AI entry found (main.exe on Windows / main on POSIX / main.py) in {ai_dir}"
    )

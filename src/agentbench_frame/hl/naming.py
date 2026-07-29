"""Name generation for HL rounds and iterations.

Round name:   ``hl-v<YY-MM-DD>-round<n>``   (one per CLI run / round)
Act name:     ``<round-name>-00000<m>``      (one per iteration/act in a round)

``<YY-MM-DD>`` is the round-start date, 2-digit year, ``-`` separator. Colons
are illegal in Windows paths, so the spec's ``YY:MM:DD`` is rendered with ``-``.
The date is cosmetic — uniqueness is guaranteed by the monotonic ``n`` (see
``round_state.py``), so a clock jump can never cause a collision.
"""
from __future__ import annotations

import re
from datetime import datetime

# Matches a round dir/codebase name and captures its date and round number.
ROUND_NAME_RE = re.compile(r"^hl-v(?P<date>\d{2}-\d{2}-\d{2})-round(?P<n>\d+)$")


def today_date() -> str:
    """Today's local date as ``YY-MM-DD``."""
    return datetime.now().strftime("%y-%m-%d")


def round_name(date: str, n: int) -> str:
    """Build a round name: ``hl-v<date>-round<n>``."""
    return f"hl-v{date}-round{n}"


def act_name(round_name_str: str, m: int) -> str:
    """Build a per-act name: ``<round-name>-00000<m>``.

    The spec's ``00000<m>`` is five leading zeros + the index → 6 digits total
    (``000001``, ``000013``), matching the existing ``version_id`` ``_00000N``
    suffix style.
    """
    return f"{round_name_str}-{m:06d}"


def parse_round_number(name: str) -> int | None:
    """Round number encoded in a round name/dir, or None if not a round name."""
    m = ROUND_NAME_RE.match(name)
    return int(m.group("n")) if m else None

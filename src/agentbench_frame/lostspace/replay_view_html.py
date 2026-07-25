"""HTML replay viewer for LostSpace (PLACEHOLDER — phase 2).

The text viewer in ``replay_view.py`` covers the closed loop today. This module
is the planned upgrade: a self-contained static HTML page that renders the
7x7x3 board layer-by-layer and animates the recorded action stream round by
round.

It is intentionally a stub so the import surface exists; implement when needed.

Planned interface::

    from agentbench_frame.lostspace.replay_view_html import render_html
    html = render_html("replay.json")          # str
    Path("replay.html").write_text(html)
"""

from __future__ import annotations

NOT_IMPLEMENTED = (
    "HTML replay viewer is not implemented yet; use "
    "'python -m agentbench_frame.lostspace.replay_view' for the text view."
)


def render_html(replay_path: str) -> str:  # pragma: no cover - stub
    raise NotImplementedError(NOT_IMPLEMENTED)


def main() -> int:  # pragma: no cover - stub
    raise SystemExit(NOT_IMPLEMENTED)


if __name__ == "__main__":  # pragma: no cover
    main()

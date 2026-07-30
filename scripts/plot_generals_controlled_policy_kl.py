#!/usr/bin/env python3
"""Render the Generals controlled-reference policy-KL paper figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentbench_frame.generals.paper_figure import (
    load_policy_kl_figure_data,
    render_policy_kl_three_panel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the English three-panel Generals policy-KL paper figure "
            "from one explicit finalized run."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Finalized generals-policy-kl run directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path without extension; writes .svg and .png.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = load_policy_kl_figure_data(args.run_dir)
    svg_path, png_path = render_policy_kl_three_panel(
        data,
        args.output_prefix,
    )
    print(svg_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

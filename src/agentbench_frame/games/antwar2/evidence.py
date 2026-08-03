"""Bounded replay evidence packets for AntWar2 HL acts."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.match_record import MatchRecord


def build_replay_evidence(
    matches: Sequence[MatchRecord],
    *,
    summarizer: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    output: list[dict[str, Any]] = []
    script = Path(summarizer).resolve()
    for match in matches:
        base = {
            "opponent": match.opponent,
            "candidate_role": match.candidate_role,
            "seed": match.seed,
            "status": match.status,
            "result": match.result,
            "points": match.points,
            "candidate_score": match.candidate_score,
            "opponent_score": match.opponent_score,
            "dense_margin": match.dense_margin,
            "faults": list(match.faults),
            "replay": match.replay,
            "trace": match.trace,
            "summary": None,
        }
        if match.promotable and match.replay is not None:
            replay = Path(match.replay).resolve()
            summary = replay.with_name("match_summary.json")
            completed = subprocess.run(
                (
                    sys.executable,
                    str(script),
                    str(replay),
                    "--output",
                    str(summary),
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if completed.returncode != 0 or not summary.is_file():
                diagnostic = (completed.stderr or completed.stdout)[-4000:]
                raise RuntimeError(f"AntWar2 replay summarization failed: {diagnostic}")
            value = json.loads(summary.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("AntWar2 replay summary must be an object")
            base["summary"] = str(summary)
        output.append(base)
    return tuple(output)

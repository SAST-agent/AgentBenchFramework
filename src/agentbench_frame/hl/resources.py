"""
Read-only resource views handed to the coding agent during HL iteration.

The agent never reads ``runs/`` directly; it goes through these views so the
framework controls what's exposed (and the ladder opponent source is never
leaked — only opponent *names* and match outcomes).

- ``MatchHistoryView``: aggregates the real ``runs/{game}/{agent}/*/
  {summary.json, matches.jsonl}`` layout. This is the corrected loader that
  ``mcp/QueryHistoryTool`` (which reads the wrong flat dir) should delegate to.
- ``ReplayView``: native ``artifacts/*.json`` replay + score/ranking helpers.
- ``VersionDiffView``: read-only prior-version content + structured diff.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.hl.codebase import HLCodebase, StructuredDiff


class MatchHistoryView:
    """Aggregate match history for one agent across all its runs."""

    def __init__(self, *, data_root, game: str, agent: str):
        self.data_root = Path(data_root)
        self.game = game
        self.agent = agent

    def _agent_runs(self) -> List[Path]:
        base = self.data_root / "runs" / self.game / self.agent
        if not base.is_dir():
            return []
        return sorted([d for d in base.iterdir() if d.is_dir()])

    def run_summaries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for run_dir in self._agent_runs():
            sp = run_dir / "summary.json"
            if sp.exists():
                try:
                    out.append(json.loads(sp.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        return out

    def match_rows(self) -> List[Dict[str, Any]]:
        """Every per-match record, across all runs, in run order."""
        rows: List[Dict[str, Any]] = []
        for run_dir in self._agent_runs():
            mp = run_dir / "matches.jsonl"
            if not mp.exists():
                continue
            for line in mp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["run_id"] = run_dir.name
                    rows.append(rec)
                except json.JSONDecodeError:
                    continue
        return rows

    def by_opponent(self) -> Dict[str, Dict[str, Any]]:
        """Per-opponent W/L/error counts. Errors are NOT losses (doc §12);
        ``win_rate`` is None when there are no valid games."""
        agg: Dict[str, Dict[str, Any]] = {}
        for r in self.match_rows():
            opp = r.get("opponent", "?")
            a = agg.setdefault(opp, {"wins": 0, "losses": 0, "errors": 0,
                                    "valid_games": 0, "ranks": []})
            res = r.get("candidate_result")
            if res == "win":
                a["wins"] += 1
                a["valid_games"] += 1
            elif res == "loss":
                a["losses"] += 1
                a["valid_games"] += 1
            else:  # error / missing
                a["errors"] += 1
            if r.get("candidate_rank") is not None:
                a["ranks"].append(r["candidate_rank"])
        for a in agg.values():
            if a["valid_games"] > 0:
                a["win_rate"] = a["wins"] / a["valid_games"]
                a["avg_rank"] = (sum(a["ranks"]) / len(a["ranks"])
                                 if a["ranks"] else None)
            else:
                a["win_rate"] = None
                a["avg_rank"] = None
        return agg


class ReplayView:
    """Read a native LostSpace replay (one JSON array)."""

    def __init__(self, path):
        self.path = Path(path)
        self._data: Optional[list] = None

    def _load(self) -> list:
        if self._data is None:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        return self._data

    def birthplaces(self) -> list:
        return self._load()[0]

    def rounds(self) -> list:
        # rounds are indices 1 .. len-2
        return self._load()[1:-1]

    def n_rounds(self) -> int:
        return len(self.rounds())

    def score_dic(self) -> Dict[int, int]:
        """Player-id(int) -> rank points (4=1st ... 1=4th)."""
        raw = self._load()[-1]
        return {int(k): int(v) for k, v in raw.items()}

    def ranking(self) -> List[int]:
        """Player ids sorted by score descending (4=1st ... 1=4th)."""
        sd = self.score_dic()
        return sorted(sd.keys(), key=lambda k: sd[k], reverse=True)

    def winner(self) -> Optional[int]:
        sd = self.score_dic()
        if not sd:
            return None
        return max(sd, key=lambda k: sd[k])


class VersionDiffView:
    """Read-only structured diff + content access between two versions.

    The agent can see *what changed* and read either version's files, but it
    cannot write to the store or the live workspace through this view.
    """

    def __init__(self, codebase: HLCodebase, before_hash: str, after_hash: str):
        self.codebase = codebase
        self.before_hash = before_hash
        self.after_hash = after_hash
        self._diff: Optional[StructuredDiff] = None

    @property
    def diff(self) -> StructuredDiff:
        if self._diff is None:
            self._diff = self.codebase.diff(self.before_hash, self.after_hash)
        return self._diff

    @property
    def added(self) -> List[str]:
        return self.diff.added

    @property
    def removed(self) -> List[str]:
        return self.diff.removed

    @property
    def modified(self) -> List[str]:
        return self.diff.modified

    @property
    def unchanged(self) -> List[str]:
        return self.diff.unchanged

    def read_file(self, *, after_hash: Optional[str] = None,
                  before_hash: Optional[str] = None,
                  relpath: str) -> str:
        """Read one file's content from a snapshot (read-only)."""
        h = after_hash or before_hash or self.after_hash
        snap = self.codebase.store / h / relpath
        if not snap.exists():
            raise FileNotFoundError(relpath)
        return snap.read_text(encoding="utf-8")

"""
ExperienceStore — the agent's self-summarized, persisted experience (HL std 5).

The HL loop's per-act feedback only carries the *last* act's outcome. Without a
persisted memory the agent re-derives the same lessons every act and re-tries
disproven ideas. The ExperienceStore is the missing long-term memory:

- After every act the controller appends one raw **observation** to a staging
  file (``.experience_staging.jsonl``) — opponent, outcome, KL, the seat-0
  digest, the edit type.
- The **experience document** ``EXPERIENCE.md`` lives at the round root
  (``.hl_codebase/<round>/EXPERIENCE.md``) — *outside* the snapshot store, so
  it never pollutes the ``agent.py`` content-hash diff. It is written and
  re-summarized by the coding agent itself on the consolidation cadence (the
  consolidation mission in ``context.py``), not auto-generated. That makes it a
  *self-summarized* skill: the agent distills staging observations into a
  short, deduplicated set of lessons / open questions / retired ideas, rather
  than an ever-growing log.

The store is read-only from the controller's side: ``render()`` returns the
current ``EXPERIENCE.md`` (capped, for prompt injection) and ``path`` gives the
absolute path so the consolidation mission can tell the agent where to edit.
If no document exists yet, ``render()`` falls back to a compact view of the
raw staging observations so the agent always has *some* accumulated context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperienceStore:
    """Persisted, self-summarized experience for one HL round.

    Files (under ``round_root``):
    - ``EXPERIENCE.md``           — the distilled document (agent-authored)
    - ``.experience_staging.jsonl`` — raw per-act observations (controller-appended)
    """

    def __init__(self, round_root):
        self.round_root = Path(round_root)
        self.doc_path = self.round_root / "EXPERIENCE.md"
        self.staging_path = self.round_root / ".experience_staging.jsonl"

    @property
    def path(self) -> Path:
        """Absolute path to the experience document (for the agent to edit)."""
        return self.doc_path.resolve()

    def propose_update(self, *, act_id: str, feedback: Dict[str, Any],
                       seat0_digest: Optional[Dict[str, Any]] = None) -> None:
        """Append one raw observation from this act's outcome.

        Append-only and best-effort: a missing field stays missing (never
        coerced), and an IO failure never breaks the act loop (the controller
        wraps the call in try/except).
        """
        self.round_root.mkdir(parents=True, exist_ok=True)
        rec = {
            "act_id": act_id,
            "edit_type": feedback.get("edit_type"),
            "win_rate": feedback.get("win_rate"),
            "avg_rank": feedback.get("avg_rank"),
            "avg_score": feedback.get("avg_score"),
            "active_opponents": list(feedback.get("active_opponents") or []),
            "kl_mean": feedback.get("kl_mean"),
            "n_changed": feedback.get("n_changed"),
            "n_total": feedback.get("n_total"),
            "occupancy_shift": feedback.get("occupancy_shift"),
            "seat0": seat0_digest,
        }
        with self.staging_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def observations(self) -> List[Dict[str, Any]]:
        """Raw staging observations in append order."""
        if not self.staging_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.staging_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def render(self, *, max_lines: int = 40) -> str:
        """Render the experience for prompt injection.

        If the agent has authored ``EXPERIENCE.md``, return it (capped to
        ``max_lines``). Otherwise render a compact summary of the raw staging
        observations so there is always *some* accumulated context to act on.
        """
        if self.doc_path.exists():
            text = self.doc_path.read_text(encoding="utf-8").rstrip()
            lines = text.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines.append(f"... ({len(text.splitlines()) - max_lines} more lines in "
                             f"{self.doc_path.name})")
            return "\n".join(lines)
        obs = self.observations()
        if not obs:
            return ""
        rows = []
        for o in obs[-8:]:  # last few raw observations
            opp = ", ".join(o.get("active_opponents") or []) or "?"
            wr = o.get("win_rate")
            wr_s = f"{wr:.0%}" if isinstance(wr, (int, float)) else "-"
            ar = o.get("avg_rank")
            ar_s = f"{ar:.1f}" if isinstance(ar, (int, float)) else "-"
            et = o.get("edit_type") or "-"
            rows.append(f"- {o.get('act_id')}: vs {opp}  win_rate={wr_s} "
                        f"avg_rank={ar_s}  edit={et}")
        return ("(No EXPERIENCE.md yet — raw observations so far:)\n"
                + "\n".join(rows))

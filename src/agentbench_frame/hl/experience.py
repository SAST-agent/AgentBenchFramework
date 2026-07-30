"""Compact, versioned empirical memory for the coding agent."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Mapping


_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


@dataclasses.dataclass(frozen=True)
class ExperienceUpdate:
    stable_knowledge: tuple[str, ...] = ()
    failed_hypotheses: tuple[str, ...] = ()
    replay_evidence: tuple[str, ...] = ()
    active_questions: tuple[str, ...] = ()


class ExperienceManager:
    """Maintain evidence-grounded lessons without copying source or secrets."""

    _SECTIONS = (
        ("Stable knowledge", "stable_knowledge"),
        ("Failed hypotheses", "failed_hypotheses"),
        ("Replay evidence", "replay_evidence"),
        ("Active questions", "active_questions"),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        compress_every_acts: int = 5,
        max_entries_per_section: int = 80,
    ) -> None:
        if compress_every_acts < 1:
            raise ValueError("compress_every_acts must be >= 1")
        self.root = Path(root)
        self.history = self.root / "history"
        self.path = self.root / "SKILL.md"
        self.state_path = self.root / "state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.history.mkdir(parents=True, exist_ok=True)
        self.compress_every_acts = compress_every_acts
        self.max_entries_per_section = max_entries_per_section
        self._entries = {field: [] for _, field in self._SECTIONS}
        if self.state_path.is_file():
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            allowed = set(self._entries)
            if not isinstance(value, dict) or set(value) != allowed:
                raise ValueError("invalid persisted experience state")
            for field in allowed:
                if not isinstance(value[field], list):
                    raise ValueError("invalid persisted experience entries")
                self._entries[field] = [
                    self._validate_entry(item) for item in value[field]
                ]
        self._write()

    @staticmethod
    def _validate_entry(value: str) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError("experience entry cannot be empty")
        if _SECRET.search(normalized):
            raise ValueError("experience cannot contain credential material")
        if "```" in value or len(normalized) > 800:
            raise ValueError("experience must contain concise findings, not source code blobs")
        return normalized

    def update(self, act_id: str, update: ExperienceUpdate) -> Path:
        for _, field in self._SECTIONS:
            for raw in getattr(update, field):
                value = self._validate_entry(raw)
                if value not in self._entries[field]:
                    self._entries[field].append(value)
            self._entries[field] = self._entries[field][-self.max_entries_per_section :]
        self._write()
        (self.history / f"{act_id}.md").write_text(
            self.path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return self.path

    def apply_file(self, act_id: str, path: str | Path) -> Path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("experience update must be an object")
        allowed = {field for _, field in self._SECTIONS}
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown:
            raise ValueError(f"unknown experience fields: {unknown}")
        if missing:
            raise ValueError(f"missing experience fields: {missing}")
        normalized: dict[str, tuple[str, ...]] = {}
        for field in allowed:
            items = value[field]
            if not isinstance(items, list) or not all(
                isinstance(item, str) for item in items
            ):
                raise ValueError(f"experience {field} must be a list of strings")
            normalized[field] = tuple(items)
        return self.update(act_id, ExperienceUpdate(**normalized))

    def _write(self) -> None:
        lines = [
            "# HL Experience Skill",
            "",
            "Only empirical, replay-grounded knowledge belongs here.",
        ]
        for title, field in self._SECTIONS:
            lines.extend(["", f"## {title}", ""])
            entries = self._entries[field]
            lines.extend(f"- {entry}" for entry in entries)
            if not entries:
                lines.append("- (none)")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.state_path.write_text(
            json.dumps(
                self._entries,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

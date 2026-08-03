"""Compact, versioned empirical memory for the coding agent."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentbench_frame.hl.experience_ledger import (
    ExperienceLedger,
    ExperienceRecord,
)


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
        self.ledger = ExperienceLedger(self.root / "ledger.jsonl")
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
        return self.update(act_id, self.read_file(path))

    def read_file(self, path: str | Path) -> ExperienceUpdate:
        """Load one strict, provisional candidate-authored update."""

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
        return ExperienceUpdate(**normalized)

    def consolidate_cycle(
        self,
        cycle_id: str,
        *,
        records: Iterable[ExperienceRecord],
        notes: Iterable[ExperienceUpdate] = (),
    ) -> Path:
        """Append measured outcomes, merge bounded notes, and project the Skill."""

        normalized_cycle = self._validate_entry(cycle_id)
        for record in sorted(records, key=lambda item: item.identity):
            self.ledger.append(record)
        for update in notes:
            for _, field in self._SECTIONS:
                for raw in getattr(update, field):
                    value = self._validate_entry(raw)
                    if value not in self._entries[field]:
                        self._entries[field].append(value)
                self._entries[field] = self._entries[field][
                    -self.max_entries_per_section :
                ]
        self._write()
        (self.history / f"{normalized_cycle}.md").write_text(
            self.path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return self.path

    def rebuild(
        self,
        updates: tuple[tuple[str, str | Path], ...],
    ) -> Path:
        """Reconstruct active memory from validated staged updates."""

        self._entries = {field: [] for _, field in self._SECTIONS}
        self._write()
        for act_id, path in updates:
            self.apply_file(act_id, path)
        return self.path

    def _write(self) -> None:
        self.path.write_text(self._render_skill(), encoding="utf-8")
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

    @staticmethod
    def _number(value: float) -> str:
        if float(value).is_integer():
            return f"{value:+.0f}"
        return f"{value:+.2f}".rstrip("0").rstrip(".")

    @classmethod
    def _record_line(cls, record: ExperienceRecord) -> str:
        effects = "; ".join(
            f"{row.opponent}/seed{row.seed} margin_delta={cls._number(row.margin_delta)}"
            for row in record.comparisons
        ) or "no comparable complete matches"
        selected = "selected" if record.selected else "not-selected"
        return (
            f"{record.candidate_version_id} [{record.verdict}, {selected}, "
            f"activation={record.changed_action_count}/{record.decision_count}]: "
            f"when {record.activation_condition}; use {record.mechanism}; "
            f"{effects}; preservation: {record.preservation_contract}."
        )

    def _render_skill(self) -> str:
        records = sorted(self.ledger.load(), key=lambda item: item.identity)
        groups: dict[str, list[str]] = {
            "good": [
                self._record_line(record)
                for record in records
                if record.verdict == "verified_good"
            ],
            "bad": [
                self._record_line(record)
                for record in records
                if record.verdict == "verified_bad"
            ],
            "mixed": [
                self._record_line(record)
                for record in records
                if record.verdict in {"mixed", "inconclusive", "invalid"}
            ],
            "stable": list(self._entries["stable_knowledge"]),
            "failed": list(self._entries["failed_hypotheses"]),
            "replay": list(self._entries["replay_evidence"]),
            "questions": list(self._entries["active_questions"]),
        }
        for key in groups:
            groups[key] = groups[key][-self.max_entries_per_section :]

        negative_by_opponent: dict[str, list[float]] = {}
        for record in records:
            for comparison in record.comparisons:
                if comparison.margin_delta < 0:
                    negative_by_opponent.setdefault(comparison.opponent, []).append(
                        comparison.margin_delta
                    )
        profile = [
            (
                f"{opponent}: {len(values)} measured regressions; "
                f"worst_margin_delta={self._number(min(values))}."
            )
            for opponent, values in sorted(negative_by_opponent.items())
        ]

        def bullets(values: list[str]) -> list[str]:
            return [f"- {value}" for value in values] if values else ["- (none)"]

        def render() -> str:
            lines = [
                "# HL Experience Skill",
                "",
                "Framework-measured outcomes are authoritative; candidate notes are replay-grounded observations.",
                "",
                "## Verified good conditions",
                "",
                *bullets(groups["good"]),
                "",
                "## Verified bad conditions",
                "",
                *bullets(groups["bad"]),
                "",
                "## Mixed or scope-sensitive findings",
                "",
                *bullets(groups["mixed"]),
                "",
                "## Replay-grounded observations and open questions",
                "",
                "### Stable knowledge",
                "",
                *bullets(groups["stable"]),
                "",
                "### Failed hypotheses",
                "",
                *bullets(groups["failed"]),
                "",
                "### Replay evidence",
                "",
                *bullets(groups["replay"]),
                "",
                "### Active questions",
                "",
                *bullets(groups["questions"]),
                "",
                "## Current hard-opponent failure profile",
                "",
                *bullets(profile),
            ]
            return "\n".join(lines) + "\n"

        rendered = render()
        while len(rendered.encode("utf-8")) > 65536:
            removable = [
                (sum(len(item.encode("utf-8")) for item in values), key)
                for key, values in groups.items()
                if values
            ]
            if not removable:
                raise ValueError("experience Skill fixed content exceeds 65536 bytes")
            _, largest = max(removable)
            groups[largest].pop(0)
            rendered = render()
        return rendered

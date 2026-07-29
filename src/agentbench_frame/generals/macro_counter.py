"""Exact, resumable counting for a canonical macro-action prefix graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Any

from .measurement_state import measurement_state_id


class ExactCounterError(RuntimeError):
    """The exact-count graph or persisted cache violates its contract."""


@dataclass(frozen=True)
class ExactCountResult:
    status: str
    support_size: int | None
    expanded_states: int
    legal_edges: int
    cache_hits: int
    elapsed_time_s: float
    root_state_id: str
    error: str | None = None


class ExactCountIncomplete(RuntimeError):
    """An operational guard stopped counting without approximating a value."""

    def __init__(self, result: ExactCountResult):
        self.result = result
        super().__init__(result.error or result.status)


class _OperationalStop(RuntimeError):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


class ExactMacroCounter:
    """Memoize only completed exact suffix counts in an SQLite cache."""

    def __init__(
        self,
        action_space: Any,
        cache_path: Path,
        *,
        max_wall_time_s: float | None = None,
        max_expanded_states: int | None = None,
    ):
        self.action_space = action_space
        self.spec_id = str(action_space.spec_id)
        if not self.spec_id:
            raise ValueError("action-space spec_id cannot be empty")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if max_wall_time_s is not None and max_wall_time_s < 0:
            raise ValueError("max_wall_time_s cannot be negative")
        if (
            max_expanded_states is not None
            and (
                type(max_expanded_states) is not int
                or max_expanded_states < 0
            )
        ):
            raise ValueError(
                "max_expanded_states must be a non-negative integer"
            )
        self.max_wall_time_s = max_wall_time_s
        self.max_expanded_states = max_expanded_states
        self._initialize_cache()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize_cache(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exact_counts (
                    spec_id TEXT NOT NULL,
                    actor INTEGER NOT NULL,
                    state_id TEXT NOT NULL,
                    count_decimal TEXT NOT NULL,
                    completed_at REAL NOT NULL,
                    PRIMARY KEY (spec_id, actor, state_id)
                )
                """
            )

    @staticmethod
    def _actor(state: Mapping[str, Any]) -> int:
        actor = state.get("actor")
        if type(actor) is not int or actor not in (0, 1):
            raise ExactCounterError("state actor must be 0 or 1")
        return actor

    def _cache_get(self, actor: int, state_id: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT count_decimal FROM exact_counts "
                "WHERE spec_id = ? AND actor = ? AND state_id = ?",
                (self.spec_id, actor, state_id),
            ).fetchone()
        if row is None:
            return None
        raw = row[0]
        if (
            not isinstance(raw, str)
            or not raw
            or not raw.isdecimal()
            or int(raw) < 1
        ):
            raise ExactCounterError(
                f"corrupt exact count cache entry for {state_id}"
            )
        return int(raw)

    def _cache_put(
        self,
        actor: int,
        state_id: str,
        count: int,
    ) -> None:
        if type(count) is not int or count < 1:
            raise ExactCounterError("exact support count must be positive")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO exact_counts (
                    spec_id, actor, state_id, count_decimal, completed_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(spec_id, actor, state_id)
                DO UPDATE SET
                    count_decimal = excluded.count_decimal,
                    completed_at = excluded.completed_at
                """,
                (
                    self.spec_id,
                    actor,
                    state_id,
                    str(count),
                    time.time(),
                ),
            )

    def _check_guards(self) -> None:
        elapsed = time.monotonic() - self._started
        if (
            self.max_wall_time_s is not None
            and elapsed >= self.max_wall_time_s
        ):
            raise _OperationalStop(
                "incomplete_wall_time",
                "exact count reached the wall-time guard",
            )
        if (
            self.max_expanded_states is not None
            and self._expanded_states >= self.max_expanded_states
        ):
            raise _OperationalStop(
                "incomplete_max_expanded_states",
                "exact count reached the expanded-state guard",
            )

    def _count_state(
        self,
        state: Mapping[str, Any],
        stack: set[tuple[int, str]],
    ) -> int:
        actor = self._actor(state)
        state_id = measurement_state_id(state)
        cached = self._cache_get(actor, state_id)
        if cached is not None:
            self._cache_hits += 1
            return cached

        key = (actor, state_id)
        if key in stack:
            raise ExactCounterError(
                f"cycle detected in macro prefix graph at {state_id}"
            )
        self._check_guards()
        self._expanded_states += 1
        stack.add(key)
        try:
            total = 1
            for transition in self.action_space.legal_transitions(state):
                self._legal_edges += 1
                if transition.terminal:
                    total += 1
                else:
                    total += self._count_state(
                        transition.state_after,
                        stack,
                    )
            self._cache_put(actor, state_id, total)
            return total
        finally:
            stack.remove(key)

    def count(self, state: Mapping[str, Any]) -> ExactCountResult:
        root_state_id = measurement_state_id(state)
        self._started = time.monotonic()
        self._expanded_states = 0
        self._legal_edges = 0
        self._cache_hits = 0
        try:
            support_size = self._count_state(state, set())
        except _OperationalStop as exc:
            result = ExactCountResult(
                status=exc.status,
                support_size=None,
                expanded_states=self._expanded_states,
                legal_edges=self._legal_edges,
                cache_hits=self._cache_hits,
                elapsed_time_s=time.monotonic() - self._started,
                root_state_id=root_state_id,
                error=str(exc),
            )
            raise ExactCountIncomplete(result) from exc
        return ExactCountResult(
            status="complete",
            support_size=support_size,
            expanded_states=self._expanded_states,
            legal_edges=self._legal_edges,
            cache_hits=self._cache_hits,
            elapsed_time_s=time.monotonic() - self._started,
            root_state_id=root_state_id,
        )

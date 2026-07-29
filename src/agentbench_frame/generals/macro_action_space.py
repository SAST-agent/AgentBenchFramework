"""Complete canonical behavioral macro support for official Generals."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from agentbench_frame.eval.action_space import CanonicalMacro

from .assets import _stable_tree_hash
from .engine import OfficialGeneralsEngine
from .measurement_state import MEASUREMENT_STATE_SCHEMA


ACTION_SPACE_SCHEMA = "generals-canonical-macro-action-space-v1"
CANONICALIZATION_VERSION = "generals-behavioral-canonicalization-v1"


class ActionSpaceError(ValueError):
    """A submitted macro is malformed or outside the official support."""


@dataclass(frozen=True)
class PrimitiveTransition:
    command: tuple[int, ...]
    state_after: Mapping[str, Any]
    terminal: bool


@dataclass(frozen=True)
class ActionSpaceSpec:
    schema: str
    engine_hash: str
    measurement_state_schema: str
    canonicalization_version: str
    spec_id: str


def _spec_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GeneralsMacroActionSpaceV1:
    """An official-transition prefix graph with count-only full support."""

    def __init__(
        self,
        engine_root: Path,
        *,
        engine_hash: str | None = None,
        replay_path: Path | None = None,
        counter: Any | None = None,
    ):
        self.engine_root = Path(engine_root).resolve()
        actual_engine_hash = engine_hash or _stable_tree_hash(self.engine_root)
        if not actual_engine_hash:
            raise ActionSpaceError("official engine hash is empty")
        spec_payload = {
            "schema": ACTION_SPACE_SCHEMA,
            "engine_hash": actual_engine_hash,
            "measurement_state_schema": MEASUREMENT_STATE_SCHEMA,
            "canonicalization_version": CANONICALIZATION_VERSION,
        }
        self.spec = ActionSpaceSpec(
            **spec_payload,
            spec_id=_spec_id(spec_payload),
        )
        self.spec_id = self.spec.spec_id
        if replay_path is None:
            descriptor, name = tempfile.mkstemp(
                prefix="generals-action-probe-",
                suffix=".jsonl",
            )
            Path(name).touch(exist_ok=True)
            try:
                import os

                os.close(descriptor)
            except OSError:
                pass
            replay_path = Path(name)
        self.replay_path = Path(replay_path)
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_path.touch(exist_ok=True)
        self._counter = counter

    def spec_payload(self) -> dict[str, Any]:
        return asdict(self.spec)

    @staticmethod
    def _raw_state(
        measurement_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raw = measurement_state.get("state")
        if not isinstance(raw, Mapping):
            raise ActionSpaceError("measurement state is missing state")
        return raw

    def primitive_candidates(
        self,
        state: Mapping[str, Any],
    ) -> Iterator[tuple[int, ...]]:
        """Yield the finite complete primitive request domain in stable order."""

        raw = self._raw_state(state)
        board = sorted(
            raw["board"],
            key=lambda item: tuple(item["position"]),
        )
        positions = tuple(
            (int(item["position"][0]), int(item["position"][1]))
            for item in board
        )
        generals = tuple(
            sorted(int(item["id"]) for item in raw["generals"])
        )

        for cell in board:
            row, column = (
                int(cell["position"][0]),
                int(cell["position"][1]),
            )
            for direction in range(1, 5):
                for amount in range(1, max(1, int(cell["army"]))):
                    yield (1, row, column, direction, amount)

        for general_id in generals:
            for row, column in positions:
                yield (2, general_id, row, column)

        for general_id in generals:
            for quality in range(1, 4):
                yield (3, general_id, quality)

        for general_id in generals:
            for skill in (1, 2):
                for row, column in positions:
                    yield (4, general_id, skill, row, column)
            for skill in (3, 4, 5):
                yield (4, general_id, skill)

        for technology in range(1, 5):
            yield (5, technology)

        for weapon in (1, 2):
            for row, column in positions:
                yield (6, weapon, row, column)
        for target_row, target_column in positions:
            for source_row, source_column in positions:
                yield (
                    6,
                    3,
                    target_row,
                    target_column,
                    source_row,
                    source_column,
                )
        for row, column in positions:
            yield (6, 4, row, column)

        for row, column in positions:
            yield (7, row, column)

    @staticmethod
    def _matches_filter(
        command: tuple[int, ...],
        command_prefix: tuple[int, ...] | None,
        subcode: int | None,
    ) -> bool:
        if (
            command_prefix is not None
            and command[: len(command_prefix)] != command_prefix
        ):
            return False
        if subcode is None:
            return True
        if command[0] == 4:
            return len(command) > 2 and command[2] == subcode
        return len(command) > 1 and command[1] == subcode

    def legal_transitions(
        self,
        state: Mapping[str, Any],
        *,
        command_prefix: tuple[int, ...] | None = None,
        subcode: int | None = None,
    ) -> Iterator[PrimitiveTransition]:
        """Validate every yielded successor through a cloned official engine."""

        actor = state.get("actor")
        if type(actor) is not int or actor not in (0, 1):
            raise ActionSpaceError("measurement state actor must be 0 or 1")
        for command in self.primitive_candidates(state):
            if not self._matches_filter(command, command_prefix, subcode):
                continue
            engine = OfficialGeneralsEngine.from_measurement_state(
                self.engine_root,
                state,
                self.replay_path,
            )
            outcome = engine.apply_primitive(actor, command)
            if not outcome.valid:
                continue
            yield PrimitiveTransition(
                command=command,
                state_after=engine.measurement_state(actor),
                terminal=outcome.terminal,
            )

    @staticmethod
    def _require_command(raw: Any) -> tuple[int, ...]:
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or not raw
            or not all(type(item) is int for item in raw)
        ):
            raise ActionSpaceError("commands must be non-empty integer arrays")
        return tuple(raw)

    @classmethod
    def _normalize_request(cls, raw: Any) -> tuple[int, ...]:
        command = cls._require_command(raw)
        opcode = command[0]
        if opcode == 1 and len(command) == 5:
            return command
        if opcode == 2 and len(command) >= 4:
            return command[:4]
        if opcode == 3 and len(command) >= 3:
            return command[:3]
        if opcode == 4 and len(command) >= 3:
            skill = command[2]
            if skill in (1, 2) and len(command) >= 5:
                return command[:5]
            if skill in (3, 4, 5):
                return command[:3]
        if opcode == 5 and len(command) >= 2:
            return command[:2]
        if opcode == 6 and len(command) >= 2:
            weapon = command[1]
            if weapon in (1, 2, 4) and len(command) >= 4:
                return command[:4]
            if weapon == 3 and len(command) >= 6:
                return command[:6]
        if opcode == 7 and len(command) == 3:
            return command
        raise ActionSpaceError(
            f"malformed or unknown command opcode: {opcode}"
        )

    def canonicalize(
        self,
        state: Mapping[str, Any],
        action: Sequence[Sequence[int]],
    ) -> CanonicalMacro:
        if (
            not isinstance(action, Sequence)
            or isinstance(action, (str, bytes))
            or not action
        ):
            raise ActionSpaceError("macro action must be a non-empty sequence")

        end_indices = []
        for index, raw in enumerate(action):
            try:
                if self._require_command(raw) == (8,):
                    end_indices.append(index)
            except ActionSpaceError:
                continue
        if not end_indices:
            raise ActionSpaceError("macro action requires a final end marker")
        if end_indices != [len(action) - 1]:
            raise ActionSpaceError("submitted primitives appear after end marker")

        actor = state.get("actor")
        if type(actor) is not int or actor not in (0, 1):
            raise ActionSpaceError("measurement state actor must be 0 or 1")
        engine = OfficialGeneralsEngine.from_measurement_state(
            self.engine_root,
            state,
            self.replay_path,
        )
        canonical: list[tuple[int, ...]] = []
        for raw in action[:-1]:
            command = self._normalize_request(raw)
            source_army = None
            if command[0] == 1:
                row, column = command[1], command[2]
                if not (0 <= row < 15 and 0 <= column < 15):
                    raise ActionSpaceError("illegal official primitive")
                source_army = int(engine.state.board[row][column].army)
            outcome = engine.apply_primitive(actor, command)
            if not outcome.valid:
                raise ActionSpaceError("illegal official primitive")
            if source_army is not None:
                executed = source_army - int(
                    engine.state.board[command[1]][command[2]].army
                )
                command = command[:4] + (executed,)
            canonical.append(command)
            if outcome.terminal:
                break
        canonical.append((8,))
        return tuple(canonical)

    def contains(
        self,
        state: Mapping[str, Any],
        action: Sequence[Sequence[int]],
    ) -> bool:
        try:
            self.canonicalize(state, action)
        except (ActionSpaceError, KeyError, TypeError, ValueError):
            return False
        return True

    def cardinality(self, state: Mapping[str, Any]) -> int:
        if self._counter is None:
            raise RuntimeError("exact macro counter is not configured")
        result = self._counter.count(state)
        if result.support_size is None:
            raise RuntimeError("exact macro support count is incomplete")
        return result.support_size

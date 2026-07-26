"""Round-aligned dense trajectory diagnostics for official Generals matches."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import MatchResult


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class DenseStateSample:
    sample_index: int
    official_round: int
    sample_kind: str
    evaluated_seat: int
    target_alive: bool
    opponent_alive: bool
    target_territory: int
    opponent_territory: int
    territory_margin: int
    territory_share: float | None
    target_army: int
    opponent_army: int
    army_margin: int
    army_share: float | None
    target_coins: int | None
    opponent_coins: int | None
    coin_margin: int | None
    coin_share: float | None
    target_attack_mass_near_opponent_main: int | None
    opponent_defense_mass_near_opponent_main: int | None
    opponent_attack_mass_near_target_main: int | None
    target_defense_mass_near_target_main: int | None
    pressure_for: int | None
    pressure_against: int | None
    net_main_pressure: int | None


@dataclass(frozen=True)
class DenseMetricSummary:
    terminal: float | None
    minimum: float | None
    maximum: float | None
    time_average: float | None
    auc: float | None


@dataclass(frozen=True)
class DenseEpisodeSummary:
    case_id: str
    evaluated_seat: int
    outcome: str
    terminal_round: int | None
    completed_rounds_survived: int
    termination_type: str
    terminated: bool
    truncated: bool
    sample_count: int
    target_territory: DenseMetricSummary
    opponent_territory: DenseMetricSummary
    territory_margin: DenseMetricSummary
    territory_share: DenseMetricSummary
    target_army: DenseMetricSummary
    opponent_army: DenseMetricSummary
    army_margin: DenseMetricSummary
    army_share: DenseMetricSummary
    target_coins: DenseMetricSummary
    opponent_coins: DenseMetricSummary
    coin_margin: DenseMetricSummary
    coin_share: DenseMetricSummary
    pressure_for: DenseMetricSummary
    pressure_against: DenseMetricSummary
    net_main_pressure: DenseMetricSummary


def _position(key: str) -> tuple[int, int]:
    row, column = key.split(",", 1)
    return int(row), int(column)


def _walkable(state: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        _position(key): cell
        for key, cell in state.get("cells", {}).items()
        if int(cell.get("type", 2)) != 2
    }


def _main_position(
    state: Mapping[str, Any], player: int
) -> tuple[int, int] | None:
    candidates = sorted(
        (
            general
            for general in state.get("generals", {}).values()
            if int(general.get("player", -1)) == player
            and general.get("type") == "main"
        ),
        key=lambda general: int(general.get("id", 0)),
    )
    if not candidates:
        return None
    position = candidates[0].get("position")
    if not isinstance(position, Sequence) or len(position) != 2:
        return None
    return int(position[0]), int(position[1])


def _within_distance(
    walkable: Mapping[tuple[int, int], Mapping[str, Any]],
    start: tuple[int, int],
    maximum: int = 2,
) -> set[tuple[int, int]]:
    if start not in walkable:
        return set()
    reached = {start}
    queue = deque([(start, 0)])
    while queue:
        current, distance = queue.popleft()
        if distance == maximum:
            continue
        for delta in NEIGHBORS:
            nxt = current[0] + delta[0], current[1] + delta[1]
            if nxt in walkable and nxt not in reached:
                reached.add(nxt)
                queue.append((nxt, distance + 1))
    return reached


def _share(target: int, opponent: int) -> float | None:
    total = target + opponent
    return target / total if total else None


def _coins(state: Mapping[str, Any], player: int) -> int | None:
    values = state.get("coins")
    if not isinstance(values, Sequence) or len(values) <= player:
        return None
    return int(values[player])


def _pressure(
    state: Mapping[str, Any],
    target: int,
    opponent: int,
) -> tuple[int | None, int | None, int | None, int | None, int | None, int | None, int | None]:
    walkable = _walkable(state)
    target_main = _main_position(state, target)
    opponent_main = _main_position(state, opponent)
    if target_main is None or opponent_main is None:
        return (None,) * 7
    near_opponent = _within_distance(walkable, opponent_main)
    near_target = _within_distance(walkable, target_main)
    target_attack = sum(
        max(int(cell.get("army", 0)) - 1, 0)
        for position, cell in walkable.items()
        if position in near_opponent and int(cell.get("player", -1)) == target
    )
    opponent_defense = sum(
        int(cell.get("army", 0))
        for position, cell in walkable.items()
        if position in near_opponent and int(cell.get("player", -1)) == opponent
    )
    opponent_attack = sum(
        max(int(cell.get("army", 0)) - 1, 0)
        for position, cell in walkable.items()
        if position in near_target and int(cell.get("player", -1)) == opponent
    )
    target_defense = sum(
        int(cell.get("army", 0))
        for position, cell in walkable.items()
        if position in near_target and int(cell.get("player", -1)) == target
    )
    pressure_for = target_attack - opponent_defense
    pressure_against = opponent_attack - target_defense
    return (
        target_attack,
        opponent_defense,
        opponent_attack,
        target_defense,
        pressure_for,
        pressure_against,
        pressure_for - pressure_against,
    )


def dense_sample(
    state: Mapping[str, Any],
    evaluated_seat: int,
    sample_index: int,
    sample_kind: str,
) -> DenseStateSample:
    if evaluated_seat not in (0, 1):
        raise ValueError("evaluated_seat must be 0 or 1")
    if sample_kind not in {"initial", "completed_round", "terminal"}:
        raise ValueError("invalid dense sample kind")
    opponent = 1 - evaluated_seat
    walkable = _walkable(state)
    target_cells = [
        cell
        for cell in walkable.values()
        if int(cell.get("player", -1)) == evaluated_seat
    ]
    opponent_cells = [
        cell
        for cell in walkable.values()
        if int(cell.get("player", -1)) == opponent
    ]
    target_territory = len(target_cells)
    opponent_territory = len(opponent_cells)
    target_army = sum(int(cell.get("army", 0)) for cell in target_cells)
    opponent_army = sum(int(cell.get("army", 0)) for cell in opponent_cells)
    target_coins = _coins(state, evaluated_seat)
    opponent_coins = _coins(state, opponent)
    coin_margin = (
        target_coins - opponent_coins
        if target_coins is not None and opponent_coins is not None
        else None
    )
    coin_share = (
        _share(target_coins, opponent_coins)
        if target_coins is not None and opponent_coins is not None
        else None
    )
    (
        target_attack,
        opponent_defense,
        opponent_attack,
        target_defense,
        pressure_for,
        pressure_against,
        net_pressure,
    ) = _pressure(state, evaluated_seat, opponent)
    return DenseStateSample(
        sample_index=int(sample_index),
        official_round=int(state.get("round", 0)),
        sample_kind=sample_kind,
        evaluated_seat=evaluated_seat,
        target_alive=_main_position(state, evaluated_seat) is not None,
        opponent_alive=_main_position(state, opponent) is not None,
        target_territory=target_territory,
        opponent_territory=opponent_territory,
        territory_margin=target_territory - opponent_territory,
        territory_share=_share(target_territory, opponent_territory),
        target_army=target_army,
        opponent_army=opponent_army,
        army_margin=target_army - opponent_army,
        army_share=_share(target_army, opponent_army),
        target_coins=target_coins,
        opponent_coins=opponent_coins,
        coin_margin=coin_margin,
        coin_share=coin_share,
        target_attack_mass_near_opponent_main=target_attack,
        opponent_defense_mass_near_opponent_main=opponent_defense,
        opponent_attack_mass_near_target_main=opponent_attack,
        target_defense_mass_near_target_main=target_defense,
        pressure_for=pressure_for,
        pressure_against=pressure_against,
        net_main_pressure=net_pressure,
    )


def build_dense_trace(match: MatchResult) -> tuple[DenseStateSample, ...]:
    if not match.turns:
        return ()
    samples = [
        dense_sample(
            match.turns[0].state_before,
            match.evaluated_seat,
            0,
            "initial",
        )
    ]
    for turn in match.turns:
        if (
            turn.player == 1
            and int(turn.state_after.get("round", 0))
            > int(turn.state_before.get("round", 0))
        ):
            samples.append(
                dense_sample(
                    turn.state_after,
                    match.evaluated_seat,
                    len(samples),
                    "completed_round",
                )
            )
    samples.append(
        dense_sample(
            match.turns[-1].state_after,
            match.evaluated_seat,
            len(samples),
            "terminal",
        )
    )
    return tuple(samples)


def _metric_summary(
    trace: Sequence[DenseStateSample], field: str
) -> DenseMetricSummary:
    values = [getattr(sample, field) for sample in trace]
    if not values or any(value is None for value in values):
        return DenseMetricSummary(None, None, None, None, None)
    numeric = [float(value) for value in values]
    auc = 0.0
    for previous, current, left, right in zip(
        numeric, numeric[1:], trace, trace[1:]
    ):
        delta = right.official_round - left.official_round
        if delta < 0:
            return DenseMetricSummary(None, None, None, None, None)
        auc += (previous + current) * 0.5 * delta
    return DenseMetricSummary(
        terminal=numeric[-1],
        minimum=min(numeric),
        maximum=max(numeric),
        time_average=sum(numeric) / len(numeric),
        auc=auc,
    )


def _outcome(match: MatchResult) -> str:
    if not match.valid:
        return "invalid"
    if match.winner == -1:
        return "draw"
    if match.winner == match.evaluated_seat:
        return "win"
    return "loss"


def summarize_dense_trace(
    match: MatchResult,
    trace: Sequence[DenseStateSample],
) -> DenseEpisodeSummary:
    field_names = (
        "target_territory",
        "opponent_territory",
        "territory_margin",
        "territory_share",
        "target_army",
        "opponent_army",
        "army_margin",
        "army_share",
        "target_coins",
        "opponent_coins",
        "coin_margin",
        "coin_share",
        "pressure_for",
        "pressure_against",
        "net_main_pressure",
    )
    summaries = {field: _metric_summary(trace, field) for field in field_names}
    return DenseEpisodeSummary(
        case_id=match.case_id,
        evaluated_seat=match.evaluated_seat,
        outcome=_outcome(match),
        terminal_round=trace[-1].official_round if trace else None,
        completed_rounds_survived=sum(
            sample.sample_kind == "completed_round" for sample in trace
        ),
        termination_type=match.termination_type,
        terminated=bool(match.valid),
        truncated=not match.valid,
        sample_count=len(trace),
        **summaries,
    )


def persist_dense_diagnostics(
    match: MatchResult,
    artifact_dir: Path,
) -> tuple[tuple[DenseStateSample, ...], DenseEpisodeSummary]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace = build_dense_trace(match)
    summary = summarize_dense_trace(match, trace)
    trace_target = artifact_dir / "dense-trace.jsonl"
    trace_temporary = artifact_dir / "dense-trace.jsonl.tmp"
    trace_temporary.write_text(
        "".join(
            json.dumps(asdict(sample), sort_keys=True, ensure_ascii=False) + "\n"
            for sample in trace
        ),
        encoding="utf-8",
    )
    trace_temporary.replace(trace_target)
    summary_target = artifact_dir / "dense-summary.json"
    summary_temporary = artifact_dir / "dense-summary.json.tmp"
    summary_temporary.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_temporary.replace(summary_target)
    return trace, summary

"""Frozen clean-room Generals v8 learning and validation contract."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Sequence

from agentbench_frame.eval.benchmark import BenchmarkCase

from .assets import AssetValidationError
from .challenge_v7 import (
    FROZEN_BEFORE_ROUND7,
    ROUND7_LEARNING_SEEDS,
    ROUND7_SEALED_SEEDS,
    ROUND7_VALIDATION_SEEDS,
)
from .models import PilotConfig, Round8ChallengeConfig


ROUND8_CHALLENGE_ID = "generals-hl-v8-clean-room-v1"
ROUND8_LEARNING_SEEDS = (300101, 300202, 300303)
ROUND8_VALIDATION_SEEDS = (
    301101,
    301202,
    301303,
    301404,
    301505,
    301606,
)
ROUND8_SEATS = (0, 1)
ROUND8_VALIDATION_THRESHOLD = (2, 1)
ROUND8_FORMAL_THRESHOLD = (2, 12, 1)
FROZEN_BEFORE_ROUND8 = (
    FROZEN_BEFORE_ROUND7
    | frozenset(ROUND7_LEARNING_SEEDS)
    | frozenset(ROUND7_VALIDATION_SEEDS)
    | frozenset(ROUND7_SEALED_SEEDS)
)


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise AssetValidationError(f"{field} must be an array of integers")
    return tuple(value)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise AssetValidationError(f"{field} must be an integer")
    return value


def load_round8_challenge_config(
    path: Path,
    pilot: PilotConfig,
    *,
    engine_hash: str,
    replay_skill_sha256: str,
) -> Round8ChallengeConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        config = Round8ChallengeConfig(
            challenge_id=str(raw["challenge_id"]),
            opponent_id=str(raw["opponent_id"]),
            learning_seeds=_integer_tuple(raw["learning_seeds"], "learning_seeds"),
            validation_seeds=_integer_tuple(raw["validation_seeds"], "validation_seeds"),
            seats=_integer_tuple(raw["seats"], "seats"),
            validation_min_wins=_integer(raw["validation_min_wins"], "validation_min_wins"),
            validation_min_wins_per_seat=_integer(
                raw["validation_min_wins_per_seat"], "validation_min_wins_per_seat"
            ),
            formal_high_min_wins=_integer(raw["formal_high_min_wins"], "formal_high_min_wins"),
            formal_total_min_wins=_integer(raw["formal_total_min_wins"], "formal_total_min_wins"),
            formal_high_min_wins_per_seat=_integer(
                raw["formal_high_min_wins_per_seat"], "formal_high_min_wins_per_seat"
            ),
            engine_sha256=str(raw["engine_sha256"]),
            replay_skill_sha256=str(raw["replay_skill_sha256"]),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise AssetValidationError(f"invalid round-8 clean-room manifest: {exc}") from exc

    exact = (
        (config.challenge_id, ROUND8_CHALLENGE_ID, "challenge_id"),
        (config.learning_seeds, ROUND8_LEARNING_SEEDS, "learning seeds"),
        (config.validation_seeds, ROUND8_VALIDATION_SEEDS, "validation seeds"),
        (config.seats, ROUND8_SEATS, "seats"),
        (config.validation_threshold, ROUND8_VALIDATION_THRESHOLD, "validation threshold"),
        (config.formal_success_threshold, ROUND8_FORMAL_THRESHOLD, "formal threshold"),
    )
    for observed, expected, field in exact:
        if observed != expected:
            raise AssetValidationError(f"round-8 {field} must match the frozen contract")
    highest = pilot.opponents[0]
    if highest.tier != "high" or config.opponent_id != highest.opponent_id:
        raise AssetValidationError("round-8 opponent must be the frozen highest-tier opponent")
    learning = frozenset(config.learning_seeds)
    validation = frozenset(config.validation_seeds)
    if learning & validation or learning & FROZEN_BEFORE_ROUND8 or validation & FROZEN_BEFORE_ROUND8:
        raise AssetValidationError("round-8 seeds must be new and partition-disjoint")
    if config.engine_sha256 != engine_hash:
        raise AssetValidationError("round-8 engine digest does not match")
    if config.replay_skill_sha256 != replay_skill_sha256:
        raise AssetValidationError("round-8 replay skill digest does not match")
    return config


def _cases(
    pilot: PilotConfig,
    challenge: Round8ChallengeConfig,
    seeds: Sequence[int],
    phase: str,
) -> tuple[BenchmarkCase, ...]:
    opponent = next(item for item in pilot.opponents if item.opponent_id == challenge.opponent_id)
    return tuple(
        BenchmarkCase(
            case_id=f"{phase}-{opponent.tier}-{opponent.opponent_id}-s{seed}-p{seat}",
            opponent=opponent.opponent_id,
            seed=seed,
            first_player=seat,
            metadata={"tier": opponent.tier, "phase": phase},
        )
        for seed in seeds
        for seat in challenge.seats
    )


def build_round8_learning_cases(
    pilot: PilotConfig, challenge: Round8ChallengeConfig
) -> tuple[BenchmarkCase, ...]:
    return _cases(pilot, challenge, challenge.learning_seeds, "learn8")


def build_round8_validation_cases(
    pilot: PilotConfig, challenge: Round8ChallengeConfig
) -> tuple[BenchmarkCase, ...]:
    return _cases(pilot, challenge, challenge.validation_seeds, "validate8")

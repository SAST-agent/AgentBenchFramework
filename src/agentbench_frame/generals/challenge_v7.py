"""Frozen Generals v7 champion-challenge contract."""

from __future__ import annotations

from pathlib import Path
import tomllib

from .assets import (
    AssetValidationError,
    FROZEN_BEFORE_POLICY_KL,
    POLICY_KL_REFERENCE_SEEDS,
)
from .models import PilotConfig, Round7ChallengeConfig


ROUND7_CHALLENGE_ID = "generals-hl-v7-champion-v1"
ROUND7_LEARNING_SEEDS = (290101, 290202, 290303)
ROUND7_VALIDATION_SEEDS = (
    291101,
    291202,
    291303,
    291404,
    291505,
    291606,
)
ROUND7_SEALED_SEEDS = (
    292101,
    292202,
    292303,
    292404,
    292505,
    292606,
    292707,
    292808,
    292909,
    292999,
)
ROUND7_SEATS = (0, 1)
ROUND7_VALIDATION_THRESHOLD = (7, 3)
ROUND7_SEALED_THRESHOLD = (11, 5)
FROZEN_BEFORE_ROUND7 = (
    FROZEN_BEFORE_POLICY_KL | frozenset(POLICY_KL_REFERENCE_SEEDS)
)


def _integer_tuple(raw: object, field: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not all(type(item) is int for item in raw):
        raise AssetValidationError(f"{field} must be an array of integers")
    return tuple(raw)


def _integer(raw: object, field: str) -> int:
    if type(raw) is not int:
        raise AssetValidationError(f"{field} must be an integer")
    return raw


def load_round7_challenge_config(
    path: Path,
    pilot: PilotConfig,
    *,
    engine_hash: str,
    replay_skill_sha256: str,
) -> Round7ChallengeConfig:
    """Load the exact learning/validation/sealed champion contract."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        config = Round7ChallengeConfig(
            challenge_id=str(raw["challenge_id"]),
            opponent_id=str(raw["opponent_id"]),
            learning_seeds=_integer_tuple(
                raw["learning_seeds"],
                "round-7 learning_seeds",
            ),
            validation_seeds=_integer_tuple(
                raw["validation_seeds"],
                "round-7 validation_seeds",
            ),
            sealed_seeds=_integer_tuple(
                raw["sealed_seeds"],
                "round-7 sealed_seeds",
            ),
            seats=_integer_tuple(raw["seats"], "round-7 seats"),
            validation_min_wins=_integer(
                raw["validation_min_wins"],
                "round-7 validation_min_wins",
            ),
            validation_min_wins_per_seat=_integer(
                raw["validation_min_wins_per_seat"],
                "round-7 validation_min_wins_per_seat",
            ),
            sealed_min_wins=_integer(
                raw["sealed_min_wins"],
                "round-7 sealed_min_wins",
            ),
            sealed_min_wins_per_seat=_integer(
                raw["sealed_min_wins_per_seat"],
                "round-7 sealed_min_wins_per_seat",
            ),
            engine_sha256=str(raw["engine_sha256"]),
            replay_skill_sha256=str(raw["replay_skill_sha256"]),
        )
    except (
        OSError,
        tomllib.TOMLDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise AssetValidationError(
            f"invalid round-7 champion manifest: {exc}"
        ) from exc

    exact_values = (
        (config.challenge_id, ROUND7_CHALLENGE_ID, "challenge_id"),
        (
            config.learning_seeds,
            ROUND7_LEARNING_SEEDS,
            "learning seeds",
        ),
        (
            config.validation_seeds,
            ROUND7_VALIDATION_SEEDS,
            "validation seeds",
        ),
        (config.sealed_seeds, ROUND7_SEALED_SEEDS, "sealed seeds"),
        (config.seats, ROUND7_SEATS, "seats"),
        (
            config.validation_threshold,
            ROUND7_VALIDATION_THRESHOLD,
            "validation threshold",
        ),
        (
            config.sealed_threshold,
            ROUND7_SEALED_THRESHOLD,
            "sealed threshold",
        ),
    )
    for observed, expected, field in exact_values:
        if observed != expected:
            raise AssetValidationError(
                f"round-7 {field} must match the frozen contract"
            )

    highest = pilot.opponents[0]
    if highest.tier != "high" or config.opponent_id != highest.opponent_id:
        raise AssetValidationError(
            "round-7 opponent must be the frozen highest-tier opponent"
        )

    partitions = (
        frozenset(config.learning_seeds),
        frozenset(config.validation_seeds),
        frozenset(config.sealed_seeds),
    )
    if any(partition & FROZEN_BEFORE_ROUND7 for partition in partitions):
        raise AssetValidationError(
            "round-7 seeds overlap a previously frozen seed set"
        )
    if any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise AssetValidationError(
            "round-7 learning, validation, and sealed seeds must be disjoint"
        )

    if config.engine_sha256 != engine_hash:
        raise AssetValidationError(
            "round-7 engine digest does not match the resolved engine"
        )
    if config.replay_skill_sha256 != replay_skill_sha256:
        raise AssetValidationError(
            "round-7 replay skill digest does not match the resolved skill"
        )
    return config

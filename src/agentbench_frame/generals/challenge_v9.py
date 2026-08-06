"""Frozen Generals v9 scientific-attribution and validation contract."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Sequence

from agentbench_frame.eval.benchmark import BenchmarkCase

from .assets import AssetValidationError
from .challenge_v8 import (
    FROZEN_BEFORE_ROUND8,
    ROUND8_LEARNING_SEEDS,
    ROUND8_VALIDATION_SEEDS,
)
from .models import PilotConfig, Round9ChallengeConfig


ROUND9_CHALLENGE_ID = "generals-hl-v9-scientific-attribution-v1"
ROUND9_ATTRIBUTION_SEEDS = (
    302101,
    302202,
    302303,
    302404,
    302505,
    302606,
)
ROUND9_VALIDATION_SEEDS = (
    303101,
    303202,
    303303,
    303404,
    303505,
    303606,
)
ROUND9_SEATS = (0, 1)
ROUND9_DIAGNOSTIC_MAX_STATES = 48
ROUND9_VALIDATION_THRESHOLD = (2, 1)
ROUND9_FORMAL_THRESHOLD = (13, 2, 1)
ROUND9_PARENT_RUN_ID = "20260730_1739_680b1632"
ROUND9_PARENT_HASH = (
    "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
)
ROUND9_PREDECESSOR_RUN_ID = "20260806_1126_8e1b6795"
ROUND9_PREDECESSOR_HASH = (
    "3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a"
)
FROZEN_BEFORE_ROUND9 = (
    FROZEN_BEFORE_ROUND8
    | frozenset(ROUND8_LEARNING_SEEDS)
    | frozenset(ROUND8_VALIDATION_SEEDS)
)


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise AssetValidationError(f"{field} must be an array of integers")
    return tuple(value)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise AssetValidationError(f"{field} must be an integer")
    return value


def load_round9_challenge_config(
    path: Path,
    pilot: PilotConfig,
    *,
    engine_hash: str,
    replay_skill_sha256: str,
) -> Round9ChallengeConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        config = Round9ChallengeConfig(
            challenge_id=str(raw["challenge_id"]),
            opponent_id=str(raw["opponent_id"]),
            attribution_seeds=_integer_tuple(
                raw["attribution_seeds"], "round-9 attribution_seeds"
            ),
            validation_seeds=_integer_tuple(
                raw["validation_seeds"], "round-9 validation_seeds"
            ),
            seats=_integer_tuple(raw["seats"], "round-9 seats"),
            diagnostic_max_states=_integer(
                raw["diagnostic_max_states"], "round-9 diagnostic_max_states"
            ),
            validation_min_wins=_integer(
                raw["validation_min_wins"], "round-9 validation_min_wins"
            ),
            validation_min_wins_per_seat=_integer(
                raw["validation_min_wins_per_seat"],
                "round-9 validation_min_wins_per_seat",
            ),
            formal_total_min_wins=_integer(
                raw["formal_total_min_wins"], "round-9 formal_total_min_wins"
            ),
            formal_high_min_wins=_integer(
                raw["formal_high_min_wins"], "round-9 formal_high_min_wins"
            ),
            formal_high_min_wins_per_seat=_integer(
                raw["formal_high_min_wins_per_seat"],
                "round-9 formal_high_min_wins_per_seat",
            ),
            parent_run_id=str(raw["parent_run_id"]),
            parent_content_hash=str(raw["parent_content_hash"]),
            predecessor_run_id=str(raw["predecessor_run_id"]),
            predecessor_content_hash=str(raw["predecessor_content_hash"]),
            engine_sha256=str(raw["engine_sha256"]),
            replay_skill_sha256=str(raw["replay_skill_sha256"]),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise AssetValidationError(
            f"invalid round-9 attribution manifest: {exc}"
        ) from exc

    exact = (
        (config.challenge_id, ROUND9_CHALLENGE_ID, "challenge_id"),
        (
            config.attribution_seeds,
            ROUND9_ATTRIBUTION_SEEDS,
            "attribution seeds",
        ),
        (
            config.validation_seeds,
            ROUND9_VALIDATION_SEEDS,
            "validation seeds",
        ),
        (config.seats, ROUND9_SEATS, "seats"),
        (
            config.diagnostic_max_states,
            ROUND9_DIAGNOSTIC_MAX_STATES,
            "diagnostic max states",
        ),
        (
            config.validation_threshold,
            ROUND9_VALIDATION_THRESHOLD,
            "validation threshold",
        ),
        (
            config.formal_success_threshold,
            ROUND9_FORMAL_THRESHOLD,
            "formal threshold",
        ),
        (config.parent_run_id, ROUND9_PARENT_RUN_ID, "parent run"),
        (config.parent_content_hash, ROUND9_PARENT_HASH, "parent hash"),
        (
            config.predecessor_run_id,
            ROUND9_PREDECESSOR_RUN_ID,
            "predecessor run",
        ),
        (
            config.predecessor_content_hash,
            ROUND9_PREDECESSOR_HASH,
            "predecessor hash",
        ),
    )
    for observed, expected, field in exact:
        if observed != expected:
            raise AssetValidationError(
                f"round-9 {field} must match the frozen contract"
            )

    highest = pilot.opponents[0]
    if highest.tier != "high" or config.opponent_id != highest.opponent_id:
        raise AssetValidationError(
            "round-9 opponent must be the frozen highest-tier opponent"
        )
    attribution = frozenset(config.attribution_seeds)
    validation = frozenset(config.validation_seeds)
    if (
        attribution & validation
        or attribution & FROZEN_BEFORE_ROUND9
        or validation & FROZEN_BEFORE_ROUND9
    ):
        raise AssetValidationError(
            "round-9 seeds must be new and partition-disjoint"
        )
    if config.engine_sha256 != engine_hash:
        raise AssetValidationError("round-9 engine digest does not match")
    if config.replay_skill_sha256 != replay_skill_sha256:
        raise AssetValidationError("round-9 replay skill digest does not match")
    return config


def _cases(
    pilot: PilotConfig,
    challenge: Round9ChallengeConfig,
    seeds: Sequence[int],
    phase: str,
) -> tuple[BenchmarkCase, ...]:
    opponent = next(
        item for item in pilot.opponents if item.opponent_id == challenge.opponent_id
    )
    return tuple(
        BenchmarkCase(
            case_id=f"{phase}-{opponent.tier}-{opponent.opponent_id}-s{seed}-p{seat}",
            opponent=opponent.opponent_id,
            seed=seed,
            first_player=seat,
            metadata={
                "tier": opponent.tier,
                "phase": phase,
                "pair_id": f"s{seed}-p{seat}",
            },
        )
        for seed in seeds
        for seat in challenge.seats
    )


def build_round9_attribution_cases(
    pilot: PilotConfig,
    challenge: Round9ChallengeConfig,
) -> tuple[BenchmarkCase, ...]:
    return _cases(pilot, challenge, challenge.attribution_seeds, "attribute9")


def build_round9_validation_cases(
    pilot: PilotConfig,
    challenge: Round9ChallengeConfig,
) -> tuple[BenchmarkCase, ...]:
    return _cases(pilot, challenge, challenge.validation_seeds, "validate9")

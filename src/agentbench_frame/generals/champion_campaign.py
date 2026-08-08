"""Frozen multi-act campaign contract for defeating the Generals champion."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
import tomllib

from agentbench_frame.eval.benchmark import BenchmarkCase

from .assets import AssetValidationError
from .leaderboard_qualification import (
    QUALIFICATION_BUDGETS,
    QUALIFICATION_ID,
    QUALIFICATION_SEEDS,
)
from .models import ChampionCampaignConfig, PilotConfig


CAMPAIGN_ID = "generals-champion-campaign-v1"
CAMPAIGN_INITIAL_VERSION = "v7"
CAMPAIGN_INITIAL_RUN_ID = "20260730_1739_680b1632"
CAMPAIGN_INITIAL_HASH = (
    "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
)
CAMPAIGN_MAX_ACTS = 32
CAMPAIGN_REPLICATES = 3
CAMPAIGN_LEARNING_SEED_BASE = 306000
CAMPAIGN_LEARNING_SEED_STRIDE = 10
CAMPAIGN_LEARNING_SEED_OFFSETS = (1, 2, 3)
CAMPAIGN_SEATS = (0, 1)
CAMPAIGN_MAX_DECISIONS = 8
CAMPAIGN_MAX_PRIMITIVES = 8
CAMPAIGN_PROMPT_MAX_BYTES = 196_608
CAMPAIGN_ACTION_SPACE_SCHEMA = "generals-canonical-macro-action-space-v1"
CAMPAIGN_ACTION_SPACE_SPEC_ID = (
    "a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e"
)
CAMPAIGN_ENGINE_SHA256 = (
    "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
)
CAMPAIGN_DECISION_SPACE_SHA256 = (
    "e927cbeef293c3f366bcfe0cf2b3c274374e1e0ab0d4c18a1780213d11db0761"
)
CAMPAIGN_RULES_SHA256 = (
    "9138bb16a27714b9be2403a28bac3f81d2de5b667a7eec1dbb3198887f516950"
)
CAMPAIGN_REPLAY_SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)
CAMPAIGN_QUALIFICATION_MANIFEST_SHA256 = (
    "70c552697ed2cb54c88bd0cbebcf7deb70fa0053ae6fd26729268a8df05876a8"
)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise AssetValidationError(f"campaign {field} must be an integer")
    return value


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise AssetValidationError(f"campaign {field} must be an integer array")
    return tuple(_integer(item, field) for item in value)


def load_champion_campaign_config(
    path: Path,
    pilot: PilotConfig,
    *,
    engine_sha256: str,
    action_space_spec_id: str,
    decision_space_sha256: str,
    rules_sha256: str,
    replay_skill_sha256: str,
    qualification_manifest_sha256: str,
) -> ChampionCampaignConfig:
    """Load and independently bind every frozen campaign authority."""

    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        config = ChampionCampaignConfig(
            campaign_id=str(raw["campaign_id"]),
            opponent_id=str(raw["opponent_id"]),
            initial_policy_version=str(raw["initial_policy_version"]),
            initial_run_id=str(raw["initial_run_id"]),
            initial_policy_hash=str(raw["initial_policy_hash"]),
            max_acts=_integer(raw["max_acts"], "max_acts"),
            checkpoint_acts=_integer_tuple(
                raw["checkpoint_acts"], "checkpoint_acts"
            ),
            replicates=_integer(raw["replicates"], "replicates"),
            learning_seed_base=_integer(
                raw["learning_seed_base"], "learning_seed_base"
            ),
            learning_seed_stride=_integer(
                raw["learning_seed_stride"], "learning_seed_stride"
            ),
            learning_seed_offsets=_integer_tuple(
                raw["learning_seed_offsets"], "learning_seed_offsets"
            ),
            seats=_integer_tuple(raw["seats"], "seats"),
            max_decisions_per_episode=_integer(
                raw["max_decisions_per_episode"],
                "max_decisions_per_episode",
            ),
            max_non_end_primitives=_integer(
                raw["max_non_end_primitives"], "max_non_end_primitives"
            ),
            prompt_max_bytes=_integer(
                raw["prompt_max_bytes"], "prompt_max_bytes"
            ),
            action_space_schema=str(raw["action_space_schema"]),
            action_space_spec_id=str(raw["action_space_spec_id"]),
            engine_sha256=str(raw["engine_sha256"]),
            decision_space_sha256=str(raw["decision_space_sha256"]),
            rules_sha256=str(raw["rules_sha256"]),
            replay_skill_sha256=str(raw["replay_skill_sha256"]),
            qualification_id=str(raw["qualification_id"]),
            qualification_manifest_sha256=str(
                raw["qualification_manifest_sha256"]
            ),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise AssetValidationError(
            f"invalid champion campaign manifest: {exc}"
        ) from exc

    if set(raw) != {item.name for item in fields(ChampionCampaignConfig)}:
        raise AssetValidationError("campaign manifest fields changed")
    highest = pilot.opponents[0]
    exact = (
        (config.campaign_id, CAMPAIGN_ID, "campaign_id"),
        (config.opponent_id, highest.opponent_id, "opponent_id"),
        (highest.tier, "high", "opponent tier"),
        (
            config.initial_policy_version,
            CAMPAIGN_INITIAL_VERSION,
            "initial policy version",
        ),
        (config.initial_run_id, CAMPAIGN_INITIAL_RUN_ID, "initial run ID"),
        (config.initial_policy_hash, CAMPAIGN_INITIAL_HASH, "initial hash"),
        (config.max_acts, CAMPAIGN_MAX_ACTS, "max acts"),
        (config.checkpoint_acts, QUALIFICATION_BUDGETS, "checkpoints"),
        (config.replicates, CAMPAIGN_REPLICATES, "replicates"),
        (
            config.learning_seed_base,
            CAMPAIGN_LEARNING_SEED_BASE,
            "learning seed base",
        ),
        (
            config.learning_seed_stride,
            CAMPAIGN_LEARNING_SEED_STRIDE,
            "learning seed stride",
        ),
        (
            config.learning_seed_offsets,
            CAMPAIGN_LEARNING_SEED_OFFSETS,
            "learning seed offsets",
        ),
        (config.seats, CAMPAIGN_SEATS, "seats"),
        (
            config.max_decisions_per_episode,
            CAMPAIGN_MAX_DECISIONS,
            "max decisions per episode",
        ),
        (
            config.max_non_end_primitives,
            CAMPAIGN_MAX_PRIMITIVES,
            "max non-end primitives",
        ),
        (
            config.prompt_max_bytes,
            CAMPAIGN_PROMPT_MAX_BYTES,
            "prompt max bytes",
        ),
        (
            config.action_space_schema,
            CAMPAIGN_ACTION_SPACE_SCHEMA,
            "action space schema",
        ),
        (
            config.action_space_spec_id,
            CAMPAIGN_ACTION_SPACE_SPEC_ID,
            "action space spec ID",
        ),
        (config.engine_sha256, engine_sha256, "engine hash"),
        (
            config.decision_space_sha256,
            decision_space_sha256,
            "decision space hash",
        ),
        (config.rules_sha256, rules_sha256, "rules hash"),
        (
            config.replay_skill_sha256,
            replay_skill_sha256,
            "replay skill hash",
        ),
        (config.qualification_id, QUALIFICATION_ID, "qualification ID"),
        (
            config.qualification_manifest_sha256,
            qualification_manifest_sha256,
            "qualification manifest hash",
        ),
        (
            config.engine_sha256,
            CAMPAIGN_ENGINE_SHA256,
            "frozen engine hash",
        ),
        (
            config.action_space_spec_id,
            action_space_spec_id,
            "resolved action space spec ID",
        ),
        (
            config.decision_space_sha256,
            CAMPAIGN_DECISION_SPACE_SHA256,
            "frozen decision space hash",
        ),
        (
            config.rules_sha256,
            CAMPAIGN_RULES_SHA256,
            "frozen rules hash",
        ),
        (
            config.replay_skill_sha256,
            CAMPAIGN_REPLAY_SKILL_SHA256,
            "frozen replay skill hash",
        ),
        (
            config.qualification_manifest_sha256,
            CAMPAIGN_QUALIFICATION_MANIFEST_SHA256,
            "frozen qualification manifest hash",
        ),
    )
    for observed, expected, field in exact:
        if observed != expected:
            raise AssetValidationError(f"campaign {field} changed")

    all_learning = {
        seed
        for act_index in range(1, config.max_acts + 1)
        for seed in campaign_learning_seeds(config, act_index)
    }
    if len(all_learning) != config.max_acts * len(
        config.learning_seed_offsets
    ):
        raise AssetValidationError("campaign learning seeds are not unique")
    forbidden = (
        set(pilot.evaluation_seeds)
        | set(pilot.learning_seeds)
        | set(QUALIFICATION_SEEDS)
    )
    if all_learning & forbidden:
        raise AssetValidationError("campaign learning seeds overlap held-out data")
    return config


def campaign_learning_seeds(
    config: ChampionCampaignConfig,
    act_index: int,
) -> tuple[int, ...]:
    if type(act_index) is not int or not 1 <= act_index <= config.max_acts:
        raise ValueError("campaign act index is outside the frozen range")
    origin = config.learning_seed_base + act_index * config.learning_seed_stride
    return tuple(origin + offset for offset in config.learning_seed_offsets)


def build_campaign_learning_cases(
    pilot: PilotConfig,
    config: ChampionCampaignConfig,
    *,
    replicate_id: str,
    act_index: int,
) -> tuple[BenchmarkCase, ...]:
    if replicate_id not in {
        f"replicate-{index}" for index in range(1, config.replicates + 1)
    }:
        raise ValueError("campaign replicate ID is not frozen")
    opponent = next(
        item
        for item in pilot.opponents
        if item.opponent_id == config.opponent_id
    )
    return tuple(
        BenchmarkCase(
            case_id=(
                f"campaign-{replicate_id}-a{act_index}-high-"
                f"s{seed}-p{seat}"
            ),
            opponent=opponent.opponent_id,
            seed=seed,
            first_player=seat,
            metadata={
                "tier": opponent.tier,
                "phase": "campaign-learning",
                "replicate_id": replicate_id,
                "act_index": act_index,
            },
        )
        for seed in campaign_learning_seeds(config, act_index)
        for seat in config.seats
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

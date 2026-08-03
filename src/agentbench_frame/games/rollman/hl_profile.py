"""Rollman-owned vocabulary and local-resource contract for HL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentbench_frame.hl.game_profile import HLGameBindings, PromptProfile


class RollmanHLProfile:
    game_id = "29_rollman"
    required_local_paths = (
        "agentbench_root",
        "official_logic_root",
        "pacman_sdk_root",
        "human_manifest",
        "workspace",
        "runs_root",
        "opponent_build_root",
    )

    def prompt_profile(self) -> PromptProfile:
        return PromptProfile(
            candidate_label="Rollman policy",
            opponent_label="human Ghost team",
            roles=("rollman",),
            policy_input="observable Pacman game state",
            output_contract="one legal movement action",
            planner_diversity=(
                "distinct observable activation condition",
                "distinct causal mechanism",
                "distinct replay-grounded falsifier",
            ),
            prohibited_information=(
                "opponent source code",
                "seed-conditioned behavior",
                "hidden engine state",
            ),
        )

    def build_bindings(
        self,
        *,
        config: Any,
        run_root: Path,
    ) -> HLGameBindings:
        raise NotImplementedError(
            "Rollman runtime bindings are constructed by the compatibility CLI"
        )

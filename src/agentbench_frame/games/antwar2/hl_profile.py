"""Declarative AntWar2 vocabulary and local-resource contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentbench_frame.hl.game_profile import HLGameBindings, PromptProfile


class AntWar2HLProfile:
    game_id = "30_antwar2"
    required_local_paths = (
        "backend_source_archive",
        "backend_executable",
        "backend_workdir",
        "sdk_root",
        "human_pool_root",
        "human_manifest",
        "workspace",
        "runs_root",
    )

    def prompt_profile(self) -> PromptProfile:
        return PromptProfile(
            candidate_label="AntWar2 policy",
            opponent_label="frozen human submission",
            roles=("P0", "P1"),
            policy_input="SDK.backend.state.BackendState public state",
            output_contract="ordered list[Operation] accepted by the official protocol",
            planner_diversity=(
                "distinct visible-state activation condition",
                "distinct causal mechanism grounded in replay evidence",
                "distinct live-match falsifier",
            ),
            prohibited_information=(
                "opponent source code",
                "opponent identity branches",
                "seed or replay identifier branches",
                "hidden engine state",
                "evaluation-only oracle policy",
            ),
            policy_entry_symbol="AI.choose_operations",
            candidate_source_relative="ai.py",
        )

    def build_bindings(
        self,
        *,
        config: Any,
        run_root: Path,
    ) -> HLGameBindings:
        raise NotImplementedError(
            "AntWar2 runtime bindings require the native evaluator adapter"
        )

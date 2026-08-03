"""Declarative AntWar2 vocabulary and local-resource contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agentbench_frame.games.antwar2.evaluator import AntWar2Evaluator, load_human_pool
from agentbench_frame.games.antwar2.evidence import build_replay_evidence
from agentbench_frame.games.antwar2.match import ProcessSpec
from agentbench_frame.games.antwar2.measurement import compare_behavior
from agentbench_frame.games.antwar2.runtime import (
    AntWarLayout,
    assemble_candidate,
    assemble_bootstrap_candidate,
    build_backend,
)
from agentbench_frame.games.antwar2 import smoke as smoke_module
from agentbench_frame.games.antwar2.smoke import verify_candidate_smoke
from agentbench_frame.hl.game_profile import (
    HLGameBindings,
    MetricSchema,
    PromptProfile,
)


class AntWar2HLProfile:
    game_id = "30_antwar2"
    required_local_paths = (
        "agentbench_root",
        "workspace",
        "runs_root",
        "build_root",
    )
    optional_local_paths = ("positive_control_root",)

    @staticmethod
    def assets_root() -> Path:
        return Path(__file__).with_name("assets")

    def layout(self, config: Any) -> AntWarLayout:
        optional = config.paths.values.get("positive_control_root")
        return AntWarLayout.from_roots(
            agentbench_root=config.paths.agentbench_root,
            build_root=config.paths.build_root,
            positive_control_root=optional,
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

    def materialize_named_origin(
        self,
        *,
        config: Any,
        name: str,
        destination: Path,
    ) -> dict[str, Any]:
        if name != "v22":
            raise ValueError("AntWar2 supports only the blinded v22 positive control")
        layout = self.layout(config)
        layout.validate(require_positive_control=True)
        assert layout.historical_versions_root is not None
        return assemble_candidate(
            destination=destination,
            policy_root=layout.historical_versions_root / "ifelse_v22",
            sdk_root=layout.sdk_root,
        )

    def build_bindings(
        self,
        *,
        config: Any,
        run_root: Path,
    ) -> HLGameBindings:
        layout = self.layout(config)
        layout.validate(
            require_positive_control=(config.run.origin.mode == "imported_version")
        )
        executable, _build_manifest = build_backend(
            archive=layout.backend_archive,
            build_root=layout.build_root,
        )
        pool = load_human_pool(
            layout.human_manifest,
            layout.human_extracted_root,
        )
        opponent_by_id = {item.opponent_id: item for item in pool}
        try:
            learning = opponent_by_id[config.run.evaluation.learning_opponent]
        except KeyError as exc:
            raise ValueError(
                "unknown AntWar2 learning opponent: "
                f"{config.run.evaluation.learning_opponent}"
            ) from exc
        fixed_seeds = tuple(config.run.evaluation.fixed_gate_seeds)
        certification_seeds = tuple(config.run.evaluation.certification_seeds)
        if not fixed_seeds:
            raise ValueError("AntWar2 evaluation.fixed_gate_seeds cannot be empty")
        if not certification_seeds:
            raise ValueError("AntWar2 evaluation.certification_seeds cannot be empty")

        def candidate_factory(version):
            root = run_root / "versions" / "objects" / version.content_hash
            return ProcessSpec((sys.executable, "main.py"), root)

        evaluator = AntWar2Evaluator(
            game=ProcessSpec((str(executable),), executable.parent.parent),
            candidate_factory=candidate_factory,
            learning_opponents=(learning,),
            human_pool=pool,
            fixed_gate_seeds=fixed_seeds,
            certification_seeds=certification_seeds,
            artifact_root=run_root / "matches",
            timeout_s=120.0,
            max_parallel_matches=config.run.evaluation.max_parallel_matches,
            quick_screen_seed_count=config.run.iteration.quick_screen_seeds,
            finalist_seed_count=config.run.iteration.finalist_seeds,
        )
        assets = self.assets_root()
        summarizer = (
            assets
            / "replay-skill/antwar2-replay/scripts/summarize_replay.py"
        )
        candidate_template = run_root / "candidate-template"
        if not candidate_template.is_dir():
            support_root = (
                layout.historical_versions_root / "ifelse_v22"
                if layout.historical_versions_root is not None
                else layout.sdk_root.parent
            )
            assemble_bootstrap_candidate(
                destination=candidate_template,
                support_root=support_root,
                sdk_root=layout.sdk_root,
                policy_template=assets / "candidate-template/ai.py",
            )
        return HLGameBindings(
            prompt_profile=self.prompt_profile(),
            metric_schema=MetricSchema(
                points_label="Live win rate",
                dense_margin_label="Terminal camp-HP margin",
                elo_label="Population Elo",
                role_labels=("P0", "P1"),
            ),
            context_sources={
                "rules": assets / "rules.md",
                "decision_space": assets / "decision_space.yaml",
                "sdk_interface": assets / "sdk_interface.md",
                "replay_skill": assets / "replay-skill/antwar2-replay",
                "smoke_fixture": Path(smoke_module.__file__),
            },
            candidate_template=candidate_template,
            evaluator=evaluator,
            smoke_verifier=lambda workspace, **_kwargs: verify_candidate_smoke(
                workspace
            ),
            replay_evidence_builder=lambda matches: build_replay_evidence(
                matches,
                summarizer=summarizer,
            ),
            behavior_comparator=lambda parent, candidate, **kwargs: compare_behavior(
                parent,
                candidate,
                references=kwargs["references"],
                epsilon=float(kwargs.get("epsilon", 0.05)),
            ),
        )

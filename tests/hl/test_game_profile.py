from pathlib import Path

import pytest


class FakeEvaluator:
    def evaluate(self, version):
        raise AssertionError(f"not called: {version}")


class FakeProfile:
    game_id = "fake"
    required_local_paths = ("backend", "workspace", "runs_root")

    def prompt_profile(self):
        from agentbench_frame.hl.game_profile import PromptProfile

        return PromptProfile(
            candidate_label="agent",
            opponent_label="opponent",
            roles=("north", "south"),
            policy_input="PublicState",
            output_contract="list[AtomicOperation]",
            planner_diversity=("distinct mechanism", "distinct falsifier"),
            prohibited_information=("opponent source", "seed lookup"),
        )

    def build_bindings(self, *, config, run_root):
        from agentbench_frame.hl.game_profile import (
            BehaviorComparison,
            HLGameBindings,
            MetricSchema,
            SmokeResult,
        )

        del config
        return HLGameBindings(
            prompt_profile=self.prompt_profile(),
            metric_schema=MetricSchema(
                points_label="Win rate",
                dense_margin_label="Score margin",
                elo_label="Agent Elo",
                role_labels=("north", "south"),
            ),
            context_sources={"rules": run_root / "rules.md"},
            candidate_template=run_root / "candidate-template",
            evaluator=FakeEvaluator(),
            smoke_verifier=lambda *_args, **_kwargs: SmokeResult(
                status="complete", error=None, artifacts={}
            ),
            replay_evidence_builder=lambda _matches: (),
            behavior_comparator=lambda *_args, **_kwargs: BehaviorComparison(
                status="complete",
                decision_count=1,
                changed_action_count=1,
                details={},
            ),
        )


@pytest.fixture(autouse=True)
def clean_registry():
    from agentbench_frame.hl.game_profile import reset_game_profiles_for_testing

    reset_game_profiles_for_testing()
    yield
    reset_game_profiles_for_testing()


def test_registry_resolves_exact_game_id_and_rejects_duplicates():
    from agentbench_frame.hl.game_profile import (
        get_game_profile,
        register_game_profile,
    )

    profile = FakeProfile()
    register_game_profile(profile)

    assert get_game_profile("fake") is profile
    with pytest.raises(ValueError, match="already registered"):
        register_game_profile(profile)


def test_registry_error_lists_available_profiles():
    from agentbench_frame.hl.game_profile import (
        get_game_profile,
        register_game_profile,
    )

    register_game_profile(FakeProfile())

    with pytest.raises(KeyError, match="registered game profiles: fake"):
        get_game_profile("missing")


def test_prompt_profile_validates_roles_and_terms():
    from agentbench_frame.hl.game_profile import PromptProfile

    profile = FakeProfile().prompt_profile()

    assert profile.roles == ("north", "south")
    with pytest.raises(ValueError, match="roles must be unique"):
        PromptProfile(
            candidate_label="agent",
            opponent_label="opponent",
            roles=("north", "north"),
            policy_input="PublicState",
            output_contract="operation",
            planner_diversity=("mechanism",),
            prohibited_information=("source",),
        )


def test_profile_builds_complete_immutable_bindings(tmp_path):
    from agentbench_frame.hl.game_profile import HLGameBindings

    bindings = FakeProfile().build_bindings(config=object(), run_root=tmp_path)

    assert isinstance(bindings, HLGameBindings)
    assert bindings.context_sources == {"rules": tmp_path / "rules.md"}
    assert bindings.metric_schema.role_labels == ("north", "south")
    assert bindings.candidate_template == tmp_path / "candidate-template"


@pytest.mark.parametrize(
    ("status", "decisions", "changed", "message"),
    (
        ("complete", 0, 1, "changed_action_count cannot exceed decision_count"),
        ("failed", 1, 1, "failed behavior comparison cannot report decisions"),
    ),
)
def test_behavior_comparison_rejects_inconsistent_counts(
    status,
    decisions,
    changed,
    message,
):
    from agentbench_frame.hl.game_profile import BehaviorComparison

    with pytest.raises(ValueError, match=message):
        BehaviorComparison(
            status=status,
            decision_count=decisions,
            changed_action_count=changed,
            details={},
        )


def test_public_hl_package_exports_profile_registry():
    from agentbench_frame.hl import get_game_profile, register_game_profile

    profile = FakeProfile()
    register_game_profile(profile)

    assert get_game_profile(profile.game_id) is profile


def test_builtin_rollman_profile_is_loaded_lazily_after_registry_reset():
    from agentbench_frame.hl.game_profile import (
        get_game_profile,
        reset_game_profiles_for_testing,
    )

    reset_game_profiles_for_testing()

    profile = get_game_profile("29_rollman")

    assert profile.game_id == "29_rollman"
    assert profile.prompt_profile().roles == ("rollman",)
    assert profile.required_local_paths == (
        "agentbench_root",
        "official_logic_root",
        "pacman_sdk_root",
        "human_manifest",
        "workspace",
        "runs_root",
        "opponent_build_root",
    )

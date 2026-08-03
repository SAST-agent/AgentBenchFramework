from pathlib import Path

import pytest


class _FakeProfile:
    game_id = "fake_paths"
    required_local_paths = ("backend", "human_pool", "workspace", "runs_root")
    optional_local_paths = ("historical_root",)

    def prompt_profile(self):
        raise AssertionError("not needed for path parsing")

    def build_bindings(self, *, config, run_root):
        raise AssertionError("not needed for path parsing")


def _fake_config(tmp_path, *, paths):
    from agentbench_frame.hl.game_profile import register_game_profile

    try:
        register_game_profile(_FakeProfile())
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
    source = tmp_path / "project" / "configs" / "hl" / "fake.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join(
            (
                'schema_version: "1.0"',
                "run:",
                '  game: "fake_paths"',
                "  provider:",
                '    kind: "codex"',
                "  origin:",
                '    mode: "model_bootstrap"',
                "  curriculum:",
                '    mode: "weakest_failed"',
                '    target_order: "lowest_rank_first"',
                "    preserve_passed_opponents: true",
                "    required_human_opponents: 1",
                "    stagnation_patience: 4",
                "paths:",
                *(f'  {name}: "{value}"' for name, value in paths.items()),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_local_config_accepts_exact_paths_declared_by_game_profile(tmp_path):
    from agentbench_frame.hl.local_config import LocalHLConfig

    source = _fake_config(
        tmp_path,
        paths={
            "backend": "backend",
            "human_pool": "humans.json",
            "workspace": "candidate",
            "runs_root": "runs",
        },
    )

    config = LocalHLConfig.load(source)

    assert config.paths.require("backend") == (
        tmp_path / "project" / "backend"
    ).resolve()
    assert config.paths.workspace == (
        tmp_path / "project" / "candidate"
    ).resolve()


def test_local_config_accepts_but_does_not_require_optional_profile_paths(tmp_path):
    from agentbench_frame.hl.local_config import LocalHLConfig

    source = _fake_config(
        tmp_path,
        paths={
            "backend": "backend",
            "human_pool": "humans.json",
            "workspace": "candidate",
            "runs_root": "runs",
            "historical_root": "positive-control",
        },
    )

    config = LocalHLConfig.load(source)

    assert config.paths.historical_root == (
        tmp_path / "project" / "positive-control"
    ).resolve()


def test_local_config_rejects_missing_profile_path(tmp_path):
    from agentbench_frame.hl.local_config import LocalHLConfig

    source = _fake_config(
        tmp_path,
        paths={
            "backend": "backend",
            "workspace": "candidate",
            "runs_root": "runs",
        },
    )

    with pytest.raises(ValueError, match="missing paths fields:.*human_pool"):
        LocalHLConfig.load(source)


def test_rollman_k4_repair_config_freezes_native_budget_and_deadlines(
    monkeypatch,
):
    from agentbench_frame.hl.local_config import LocalHLConfig

    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "hl"
        / "29_rollman-k4-repair.yaml"
    )

    monkeypatch.setenv("AGENTBENCH_SAST_ROOT", str(Path(__file__).parents[3]))
    config = LocalHLConfig.load(config_path)

    provider = config.run.provider
    assert provider.expected_cli_version == "codex-cli 0.146.0-alpha.9.2"
    assert provider.timeout_seconds == 420
    assert provider.idle_timeout_seconds == 120
    assert provider.rollout_budget.enabled is True
    assert provider.rollout_budget.limit_tokens == 70000
    assert provider.rollout_budget.reminder_at_remaining_tokens == (
        20000,
        10000,
        5000,
    )
    assert config.paths.workspace.name == "candidate-k4-repair-v4-budget"


def test_imported_source_run_resolves_from_repository_root(tmp_path):
    from agentbench_frame.hl.local_config import LocalHLConfig

    project = tmp_path / "project"
    source = project / "configs" / "hl" / "curriculum.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
schema_version: "1.0"
run:
  game: "29_rollman"
  provider:
    kind: "codex"
  origin:
    mode: "imported_version"
    source_run: ".agentbench/29_rollman/runs/source-run"
    source_version: "v000001"
  curriculum:
    mode: "weakest_failed"
    target_order: "lowest_rank_first"
    preserve_passed_opponents: true
    required_human_opponents: 16
    stagnation_patience: 4
paths:
  agentbench_root: "fixtures/AgentBench"
  official_logic_root: "fixtures/PacmanLogic"
  pacman_sdk_root: "fixtures/PacmanSDK-python"
  human_manifest: "fixtures/MANIFEST.tsv"
  workspace: ".agentbench/candidate"
  runs_root: ".agentbench/runs"
  opponent_build_root: ".agentbench/opponents"
""".lstrip(),
        encoding="utf-8",
    )

    config = LocalHLConfig.load(source)

    assert config.run.origin.source_run == str(
        (
            project
            / ".agentbench"
            / "29_rollman"
            / "runs"
            / "source-run"
        ).resolve()
    )
    assert Path(config.run.origin.source_run).is_absolute()
    assert config.run.origin.source_version == "v000001"
    assert config.run.curriculum.mode == "weakest_failed"
    assert config.run.curriculum.required_human_opponents == 16
    assert config.run.curriculum.stagnation_patience == 4


def test_machine_local_paths_expand_environment_variables(tmp_path, monkeypatch):
    from agentbench_frame.hl.local_config import LocalPaths

    external = tmp_path / "external"
    monkeypatch.setenv("AGENTBENCH_SAST_ROOT", str(external))

    paths = LocalPaths.from_mapping(
        {
            "agentbench_root": "${AGENTBENCH_SAST_ROOT}/AgentBench",
            "official_logic_root": "${AGENTBENCH_SAST_ROOT}/PacmanLogic",
            "pacman_sdk_root": "${AGENTBENCH_SAST_ROOT}/PacmanSDK-python",
            "human_manifest": "${AGENTBENCH_SAST_ROOT}/AgentBench/MANIFEST.tsv",
            "workspace": ".agentbench/candidate",
            "runs_root": ".agentbench/runs",
            "opponent_build_root": ".agentbench/opponents",
        },
        config_dir=tmp_path / "project",
        required_names=(
            "agentbench_root",
            "official_logic_root",
            "pacman_sdk_root",
            "human_manifest",
            "workspace",
            "runs_root",
            "opponent_build_root",
        ),
    )

    assert paths.agentbench_root == external / "AgentBench"
    assert paths.official_logic_root == external / "PacmanLogic"


def test_antwar2_positive_control_config_is_k4_and_profile_driven(monkeypatch):
    from agentbench_frame.hl.local_config import LocalHLConfig

    root = Path(__file__).parents[2]
    monkeypatch.setenv("AGENTBENCH_SAST_ROOT", "/fixtures/sast")
    monkeypatch.setenv("ANTWAR2_POSITIVE_CONTROL_ROOT", "/fixtures/handoff")

    config = LocalHLConfig.load(
        root / "configs/hl/30_antwar2-positive-control.yaml"
    )

    assert config.run.game == "30_antwar2"
    assert config.run.iteration.candidates_per_cycle == 4
    assert config.run.iteration.planner_enabled is True
    assert config.run.provider.structured_output_mode == "validated_file"
    assert config.run.selection.source_size_penalty is False
    assert config.run.rollback.enabled is True
    assert config.run.measurement.epsilon == 0.05
    assert config.paths.agentbench_root == Path("/fixtures/sast/AgentBench")
    assert config.paths.positive_control_root == Path("/fixtures/handoff")

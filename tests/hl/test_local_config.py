from pathlib import Path


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

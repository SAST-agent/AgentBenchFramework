import json
from pathlib import Path

import pytest


def _static_files(root: Path) -> dict[str, Path]:
    values = {
        "rules": root / "rules.md",
        "decision_space": root / "decision_space.yaml",
        "replay_skill": root / "replay-skill" / "SKILL.md",
    }
    values["replay_skill"].parent.mkdir(parents=True)
    values["rules"].write_text("RULE-MARKER-LONG-CONTENT", encoding="utf-8")
    values["decision_space"].write_text("DECISION-MARKER-LONG-CONTENT", encoding="utf-8")
    values["replay_skill"].write_text("SKILL-MARKER-LONG-CONTENT", encoding="utf-8")
    return values


def test_context_references_static_files_and_current_workspace_instead_of_embedding_them(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    files = _static_files(tmp_path / "assets")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "agent.py").write_text("SOURCE-MARKER-DO-NOT-PASTE", encoding="utf-8")
    bundle = ContextBundle.create(tmp_path / "bundle", files)
    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0001",
        branch_index=0,
        branch_count=1,
        parent_version_id="v000001",
        workspace=workspace,
        replay_evidence=[{"replay_id": "r-7", "fact": "level 1 round 20: shield destroyed"}],
        previous_measurements={"benchmark_score": 0.25, "mean_local_policy_kl": 0.1},
        experience_path=tmp_path / "experience" / "SKILL.md",
    )

    assert str(bundle.manifest_path) in prompt
    assert str(workspace) in prompt
    assert "r-7" in prompt
    assert "shield destroyed" in prompt
    assert "RULE-MARKER-LONG-CONTENT" not in prompt
    assert "DECISION-MARKER-LONG-CONTENT" not in prompt
    assert "SOURCE-MARKER-DO-NOT-PASTE" not in prompt


def test_prompt_requires_replay_grounded_causal_change_and_blocks_grid_search(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(tmp_path / "bundle", _static_files(tmp_path / "assets"))
    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0002",
        branch_index=1,
        branch_count=3,
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        replay_evidence=[{"replay_id": "r-8", "fact": "round 61: ignored open portal"}],
        previous_measurements={},
        experience_path=tmp_path / "experience.md",
    )

    assert "可证伪的因果诊断" in prompt
    assert "禁止无依据的参数枚举或 grid search" in prompt
    assert "机制上不同" in prompt
    assert "候选 2/3" in prompt
    assert "压缩或整合" in prompt


def test_curriculum_prompt_names_target_and_locked_pool(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )
    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0003",
        branch_index=0,
        branch_count=1,
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        replay_evidence=[
            {
                "opponent": "rank15",
                "seed": 101,
                "replay": "/matches/rank15/replay.jsonl",
            }
        ],
        previous_measurements={"benchmark_score": 0.0},
        experience_path=tmp_path / "experience.md",
        active_target="rank15",
        locked_opponents=("rank01", "rank16"),
    )

    assert "当前学习目标：rank15" in prompt
    assert "rank01" in prompt
    assert "rank16" in prompt
    assert "if/else" in prompt
    assert "固定回放坐标" in prompt
    assert "grid search" in prompt
    assert "只针对当前目标" in prompt
    assert "科研隔离边界" in prompt
    assert "其他 run" in prompt
    assert "其他候选目录" in prompt
    assert "审计工具调用路径" in prompt


def test_curriculum_prompt_rejects_evidence_from_another_opponent(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )

    with pytest.raises(ValueError, match="active target"):
        IterationContext(bundle).build_prompt(
            act_id="act-0003",
            branch_index=0,
            branch_count=1,
            parent_version_id="v000001",
            workspace=tmp_path / "candidate",
            replay_evidence=[
                {
                    "opponent": "rank14",
                    "seed": 101,
                    "replay": "/matches/rank14/replay.jsonl",
                }
            ],
            previous_measurements={"benchmark_score": 0.0},
            experience_path=tmp_path / "experience.md",
            active_target="rank15",
            locked_opponents=("rank01",),
        )


def test_bootstrap_prompt_creates_interpretable_origin_without_fake_replay(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(tmp_path / "bundle", _static_files(tmp_path / "assets"))
    workspace = tmp_path / "candidate"
    workspace.mkdir()

    prompt = IterationContext(bundle).build_bootstrap_prompt(
        act_id="act-000001-b00",
        workspace=workspace,
        experience_path=tmp_path / "experience" / "SKILL.md",
    )

    assert str(bundle.manifest_path) in prompt
    assert str(workspace) in prompt
    assert "初始算法" in prompt
    assert "可解释" in prompt
    assert "没有比赛回放" in prompt
    assert "不要虚构回放证据" in prompt
    assert "grid search" in prompt


def test_checkpoint_records_hashes_and_recovery_inputs(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, write_checkpoint

    bundle = ContextBundle.create(tmp_path / "bundle", _static_files(tmp_path / "assets"))
    experience = tmp_path / "experience.md"
    experience.write_text("stable experience", encoding="utf-8")
    checkpoint = write_checkpoint(
        tmp_path / "checkpoint.json",
        act_id="act-0003",
        parent_version_id="v000002",
        bundle=bundle,
        experience_path=experience,
        replay_ids=["r-2", "r-5"],
        prompt="incremental prompt",
        thread_id="thread-123",
        provider_fingerprint="provider-sha",
    )
    persisted = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))

    assert persisted == checkpoint
    assert persisted["context_bundle_hash"] == bundle.bundle_hash
    assert persisted["experience_hash"]
    assert persisted["replay_ids"] == ["r-2", "r-5"]
    assert persisted["thread_id"] == "thread-123"
    assert persisted["prompt"] == "incremental prompt"

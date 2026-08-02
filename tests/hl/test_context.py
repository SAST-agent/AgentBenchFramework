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


def _digest_files(root: Path) -> dict[str, Path]:
    values = _static_files(root)
    values["rules"].write_text(
        "# Rollman rules\n\n## Collision\n\nPaths collide.\n",
        encoding="utf-8",
    )
    values["decision_space"].write_text(
        """
policy_interface:
  input_type: core.gamedata.GameState
  object_fields: [level, round, board, pacman_pos, ghosts_pos]
  normalized_state_mapping:
    pacman_coord: pacman_pos
    ghosts_coord: ghosts_pos
roles:
  rollman:
    role_id: 0
    output_shape: one integer
    actions:
      - {id: 0, name: STAY}
      - {id: 1, name: UP}
      - {id: 2, name: LEFT}
      - {id: 3, name: DOWN}
      - {id: 4, name: RIGHT}
  ghosts:
    role_id: 1
    output_shape: ordered triple
    component_support: [0, 1, 2, 3, 4]
""".lstrip(),
        encoding="utf-8",
    )
    values["replay_skill"].write_text(
        "---\nname: rollman-replay\ndescription: Diagnose Rollman JSONL.\n---\n\n# Skill\n",
        encoding="utf-8",
    )
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


def test_context_bundle_copies_complete_skill_package(tmp_path):
    from agentbench_frame.hl.context import ContextBundle

    skill = tmp_path / "source-skill"
    script = skill / "scripts" / "summarize.py"
    script.parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Replay skill", encoding="utf-8")
    script.write_text("print('summary')\n", encoding="utf-8")

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        {"replay_skill": skill},
    )

    copied_skill = bundle.files["replay_skill"]
    assert copied_skill.name == "SKILL.md"
    assert (copied_skill.parent / "scripts" / "summarize.py").read_text(
        encoding="utf-8"
    ) == "print('summary')\n"
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["replay_skill"]["resources"] == [
        "SKILL.md",
        "scripts/summarize.py",
    ]


def test_game_digest_is_deterministic_and_contains_primitive_actions(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, compile_game_digest

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _digest_files(tmp_path / "assets"),
    )

    first = compile_game_digest(bundle, tmp_path / "first.json")
    second = compile_game_digest(bundle, tmp_path / "second.json")

    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text(encoding="utf-8"))
    assert value["context_bundle_hash"] == bundle.bundle_hash
    assert [item["id"] for item in value["roles"]["rollman"]["actions"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert value["rule_sections"] == ["Rollman rules", "Collision"]
    assert value["replay_skill"]["name"] == "rollman-replay"
    assert value["policy_interface"]["input_type"] == "core.gamedata.GameState"
    assert value["policy_interface"]["normalized_state_mapping"] == {
        "ghosts_coord": "ghosts_pos",
        "pacman_coord": "pacman_pos",
    }


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
    assert "允许策略代码增长" in prompt
    assert "压缩或整合被替代的策略" not in prompt


def test_prompt_triggers_opponent_distillation_after_three_stagnant_rollouts(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    files = _static_files(tmp_path / "assets")
    scripts = files["replay_skill"].parent / "scripts"
    scripts.mkdir()
    (scripts / "inspect_trace_window.py").write_text("", encoding="utf-8")
    (scripts / "distill_opponent_policy.py").write_text("", encoding="utf-8")
    bundle = ContextBundle.create(tmp_path / "bundle", files)

    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0004",
        branch_index=0,
        branch_count=1,
        parent_version_id="v000003",
        workspace=tmp_path / "candidate",
        replay_evidence=[{"opponent": "rank15", "trace": "trace.jsonl"}],
        previous_measurements={"curriculum_stagnation_count": 3},
        experience_path=tmp_path / "experience.md",
        active_target="rank15",
    )

    assert "停滞干预（连续无提升 3 轮）" in prompt
    assert "distill_opponent_policy.py" in prompt
    assert "不能复制 Ghost 动作" in prompt
    assert "KL 决策空间保持不变" in prompt


def test_prompt_reuses_shared_opponent_distillation_without_rerunning_tool(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    files = _static_files(tmp_path / "assets")
    scripts = files["replay_skill"].parent / "scripts"
    scripts.mkdir()
    (scripts / "inspect_trace_window.py").write_text("", encoding="utf-8")
    (scripts / "distill_opponent_policy.py").write_text("", encoding="utf-8")
    bundle = ContextBundle.create(tmp_path / "bundle", files)
    shared = tmp_path / "shared-distillation.json"
    shared.write_text("{}\n", encoding="utf-8")

    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0004",
        branch_index=0,
        branch_count=4,
        parent_version_id="v000003",
        workspace=tmp_path / "candidate",
        replay_evidence=[{"opponent": "rank15", "trace": "trace.jsonl"}],
        previous_measurements={
            "curriculum_stagnation_count": 3,
            "opponent_distillation_path": str(shared),
        },
        experience_path=tmp_path / "experience.md",
        active_target="rank15",
    )

    assert str(shared) in prompt
    assert "共享 Ghost 蒸馏" in prompt
    assert "不得重复运行蒸馏脚本" in prompt


def test_prompt_uses_controller_supplied_distillation_from_research_debt(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    files = _static_files(tmp_path / "assets")
    scripts = files["replay_skill"].parent / "scripts"
    scripts.mkdir()
    (scripts / "inspect_trace_window.py").write_text("", encoding="utf-8")
    (scripts / "distill_opponent_policy.py").write_text("", encoding="utf-8")
    bundle = ContextBundle.create(tmp_path / "bundle", files)
    shared = tmp_path / "shared-distillation.json"
    shared.write_text("{}\n", encoding="utf-8")

    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0005",
        branch_index=0,
        branch_count=4,
        parent_version_id="v000000",
        workspace=tmp_path / "candidate",
        replay_evidence=[{"opponent": "rank15", "trace": "trace.jsonl"}],
        previous_measurements={
            "curriculum_stagnation_count": 1,
            "opponent_distillation_path": str(shared),
        },
        experience_path=tmp_path / "experience.md",
        active_target="rank15",
    )

    assert str(shared) in prompt
    assert "共享 Ghost 蒸馏" in prompt
    assert "不得重复运行蒸馏脚本" in prompt


def test_planner_prompt_requires_early_durable_branch_briefs(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )
    prompt = IterationContext(bundle).build_planner_prompt(
        act_id="act-planner",
        iteration_id="iter-000002",
        parent_version_id="v000000",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        replay_evidence=[{"summary": "summary.md"}],
        previous_measurements={},
        active_target="rank15",
    )

    assert "第 2 次工具调用结束前" in prompt
    assert "禁止读取 ai.py" in prompt
    assert "候选 act 负责核对代码" in prompt
    assert "不要逐个读取四个 summary" in prompt


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
    assert "gamestate_to_statedict" in prompt
    assert "pacman_pos" in prompt
    assert "只针对当前目标" in prompt
    assert "科研隔离边界" in prompt
    assert "其他 run" in prompt
    assert "其他候选目录" in prompt
    assert "审计工具调用路径" in prompt
    assert "最多 10 次工具调用" in prompt
    assert "不得打印完整 replay" in prompt
    assert "inspect_trace_window.py" in prompt
    assert "禁止用 cat、sed、head、tail、rg 或自行脚本读取 trace" in prompt
    assert "测量完整通过后合并" in prompt
    assert "summary" in prompt


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


def test_k4_role_prompts_use_digest_research_state_and_exact_branch_brief(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _digest_files(tmp_path / "assets"),
    )
    context = IterationContext(bundle)
    digest = tmp_path / "game_digest.json"
    research = tmp_path / "research_state.json"
    reducer_input = tmp_path / "reducer_input.json"
    repair_input = tmp_path / "repair_input.json"
    for path in (digest, research, reducer_input, repair_input):
        path.write_text("{}\n", encoding="utf-8")
    evidence = [{"opponent": "rank15", "summary": "summary.json"}]

    planner = context.build_planner_prompt(
        act_id="act-planner",
        iteration_id="iter-000001",
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        replay_evidence=evidence,
        previous_measurements={"score": 0.0},
        active_target="rank15",
    )
    candidate = context.build_candidate_prompt(
        act_id="act-b02",
        branch_index=2,
        branch_count=4,
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        replay_evidence=evidence,
        previous_measurements={"score": 0.0},
        experience_path=tmp_path / "experience" / "SKILL.md",
        branch_brief={
            "branch_index": 2,
            "diagnosis": "round 12 entered a trap",
            "mechanism": "time-expanded escape search",
            "activation_condition": "level 3 and next cell has one safe exit",
            "preservation_contract": "ordinary portal and safety selection stays unchanged",
            "expected_change": "survive the junction",
            "falsifier": "capture time does not improve",
        },
        active_target="rank15",
    )
    repair = context.build_repair_prompt(
        act_id="act-repair",
        iteration_id="iter-000001",
        branch_index=2,
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        repair_input_path=repair_input,
        experience_path=tmp_path / "experience" / "SKILL.md",
    )
    reducer = context.build_reducer_prompt(
        act_id="act-reducer",
        iteration_id="iter-000001",
        selected_version_id="v000004",
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        reducer_input_path=reducer_input,
    )

    assert str(digest) in planner
    assert str(research) in planner
    assert "branch_briefs.json" in planner
    assert "恰好 4" in planner
    assert "不得打开 replay 或 trace" in planner
    assert "完整 replay" in planner
    assert "activation_condition" in planner
    assert "preservation_contract" in planner
    assert "time-expanded escape search" in candidate
    assert "候选 3/4" in candidate
    assert "完整重读" in candidate
    assert "触发条件外" in candidate
    assert "保持父代" in candidate
    assert "不得修改全局 scorer" in candidate
    assert "第 6 次工具调用结束前" in candidate
    assert "首次可编译修改" in candidate
    assert "不得顺序打印完整 ai.py" in candidate
    assert str(repair_input) in repair
    assert "错误诊断" in repair
    assert "过宽" in repair
    assert "不得切换到其他 branch" in repair
    assert "最多 2 个" in repair
    assert "第 6 次工具调用结束前" in repair
    assert "首次可编译修复" in repair
    assert "批量读取" in repair
    assert str(reducer_input) in reducer
    assert "research_state_update.json" in reducer
    assert "不得修改" in reducer
    assert "不得先声明或访问 run 根目录" in reducer


def test_scope_contract_ablation_logs_scope_without_enforcing_it(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _digest_files(tmp_path / "assets"),
    )
    context = IterationContext(bundle)
    digest = tmp_path / "game_digest.json"
    research = tmp_path / "research_state.json"
    digest.write_text("{}\n", encoding="utf-8")
    research.write_text("{}\n", encoding="utf-8")

    prompt = context.build_candidate_prompt(
        act_id="act-b00",
        branch_index=0,
        branch_count=4,
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        replay_evidence=[{"opponent": "rank15", "summary": "summary.json"}],
        previous_measurements={"score": 0.0},
        experience_path=tmp_path / "experience" / "SKILL.md",
        branch_brief={
            "branch_index": 0,
            "diagnosis": "level 3 round 12 capture",
            "mechanism": "junction escape",
            "activation_condition": "one safe exit",
            "preservation_contract": "ordinary routing stays unchanged",
            "expected_change": "survive",
            "falsifier": "capture time does not improve",
        },
        active_target="rank15",
        scope_contract_required=False,
    )

    assert "one safe exit" in prompt
    assert "diagnostic-only" in prompt
    assert "不得修改全局 scorer" not in prompt


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
    assert "gamestate_to_statedict" in prompt
    assert "pacman_pos" in prompt
    assert "不返回 STAY fallback" in prompt


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

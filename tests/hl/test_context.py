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
  field_access:
    score: state.score
  replay_mapping:
    "frame.score": state.score
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
atomization:
  empty_bundle: STAY
  hold_output: 0
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
    assert value["policy_interface"]["field_access"] == {
        "score": "state.score"
    }
    assert value["policy_interface"]["replay_mapping"] == {
        "frame.score": "state.score"
    }
    assert value["atomization"] == {
        "empty_bundle": "STAY",
        "hold_output": 0,
    }


def test_game_digest_accepts_game_defined_atomic_operations(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, compile_game_digest

    files = _digest_files(tmp_path / "assets")
    files["rules"].write_text("# Ant colony rules\n", encoding="utf-8")
    files["replay_skill"].write_text(
        "---\nname: colony-replay\ndescription: Diagnose colony JSONL.\n---\n",
        encoding="utf-8",
    )
    files["decision_space"].write_text(
        """
policy_interface:
  input_type: antwar_sdk.PublicState
  object_fields: [round, camps, towers, ants, cooldowns]
  normalized_state_mapping: {}
roles:
  P0:
    role_id: 0
    output_shape: list[AtomicOperation]
    actions:
      - {name: HOLD, arguments: []}
      - {name: BUILD_TOWER, arguments: [cell]}
  P1:
    role_id: 1
    output_shape: list[AtomicOperation]
    actions:
      - {name: HOLD, arguments: []}
      - {name: UPGRADE_TOWER, arguments: [tower_id, target_type]}
""".lstrip(),
        encoding="utf-8",
    )
    bundle = ContextBundle.create(tmp_path / "bundle", files)

    value = json.loads(
        compile_game_digest(bundle, tmp_path / "digest.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(value["roles"]) == ("P0", "P1")
    assert value["roles"]["P0"]["actions"][1] == {
        "arguments": ["cell"],
        "name": "BUILD_TOWER",
    }
    assert "rollman" not in json.dumps(value).lower()


def test_profile_prompt_uses_game_vocabulary_without_rollman_leak(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext
    from agentbench_frame.hl.game_profile import PromptProfile

    bundle = ContextBundle.create(
        tmp_path / "bundle", _static_files(tmp_path / "assets")
    )
    context = IterationContext(
        bundle,
        prompt_profile=PromptProfile(
            candidate_label="colony policy",
            opponent_label="human colony",
            roles=("P0", "P1"),
            policy_input="antwar_sdk.PublicState",
            output_contract="list[AtomicOperation]",
            planner_diversity=("economy", "defense", "timing", "counterplay"),
            prohibited_information=("opponent source", "seed lookup"),
        ),
    )

    prompt = context.build_planner_prompt(
        act_id="act-000001-planner",
        iteration_id="iter-000001",
        parent_version_id="v0",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        replay_evidence=[],
        previous_measurements={},
    )

    assert "colony policy" in prompt
    assert "P0" in prompt and "P1" in prompt
    assert "antwar_sdk.PublicState" in prompt
    assert "list[AtomicOperation]" in prompt
    assert "At least one branch must distill" in prompt
    assert "public state → opponent atomic operation" in prompt
    for forbidden in ("Rollman", "Ghost", "pacman_pos", "rank15", "rank16"):
        assert forbidden not in prompt


def test_profile_planner_requires_exact_state_for_every_branch(tmp_path):
    """Keep the prompt and reachable-action validator on one contract."""
    prompt = _profile_context(tmp_path).build_planner_prompt(
        act_id="act-planner",
        iteration_id="iter-000003",
        parent_version_id="v0",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        replay_evidence=[],
        previous_measurements={},
    )

    assert (
        "Every branch must cite at least one exact state_id from "
        "parent_occupancy.state_examples" in prompt
    )
    assert (
        "An observed per-role range is supplementary and never replaces "
        "that exact state citation" in prompt
    )
    assert "inside `mechanism` or `activation_condition`" in prompt
    assert "citations that appear only in `diagnosis` are ignored" in prompt


def _profile_context(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext
    from agentbench_frame.hl.game_profile import PromptProfile

    bundle = ContextBundle.create(
        tmp_path / "bundle", _static_files(tmp_path / "assets")
    )
    return IterationContext(
        bundle,
        prompt_profile=PromptProfile(
            candidate_label="colony policy",
            opponent_label="human colony",
            roles=("P0", "P1"),
            policy_input="antwar_sdk.PublicState",
            output_contract="ordered list[AtomicOperation]",
            planner_diversity=("economy", "defense", "timing", "counterplay"),
            prohibited_information=("opponent source", "seed lookup"),
            policy_entry_symbol="AI.choose_operations",
        ),
    )


def test_profile_candidate_packet_requires_immediate_edit_without_duplicate_reads(
    tmp_path,
):
    context = _profile_context(tmp_path)
    packet = tmp_path / "candidate_input-b00.json"
    packet.write_text("{}\n", encoding="utf-8")

    prompt = context.build_candidate_prompt(
        act_id="act-b00",
        branch_index=0,
        branch_count=4,
        parent_version_id="v0",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        replay_evidence=[],
        previous_measurements={},
        experience_path=tmp_path / "experience" / "SKILL.md",
        branch_brief={
            "branch_index": 0,
            "diagnosis": "round 8 camp damage",
            "mechanism": "visible counterattack",
            "activation_condition": "camp hp decreased",
            "preservation_contract": "retain parent otherwise",
            "expected_change": "larger margin",
            "falsifier": "no margin gain",
            "code_symbols": ["AI.choose_operations", "AI.counterattack"],
        },
        candidate_input_path=packet,
    )

    assert "First tool call" in prompt
    assert "Second tool call" in prompt
    assert "Experience Skill is embedded" in prompt
    assert "do not read Experience Skill separately" in prompt
    assert "Do not run `sed`, `cat ai.py`, `rg`, `find`, or `ls`" in prompt
    assert "candidate_code_slices" in prompt
    assert "same file-change tool call" in prompt
    assert "must print `smoke_contract`" in prompt
    assert "exact `smoke_contract.command`" in prompt
    assert "host-side result serialization" in prompt
    assert "do not inspect, shim, or patch the fixture" in prompt
    assert "game_digest.policy_interface.field_access" in prompt
    assert "game_digest.atomization" in prompt
    assert "candidate_code_index entry signature" in prompt


def test_profile_repair_requires_direct_entry_edit_and_frozen_field_access(tmp_path):
    prompt = _profile_context(tmp_path).build_repair_prompt(
        act_id="act-repair",
        iteration_id="iter-000003",
        branch_index=0,
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        repair_input_path=tmp_path / "repair.json",
        experience_path=tmp_path / "experience" / "SKILL.md",
    )

    assert "game_digest.policy_interface.field_access" in prompt
    assert "game_digest.atomization" in prompt
    assert "scope.activation_condition" in prompt
    assert "candidate.activation.details.state_examples" in prompt
    assert "Do not monkey-patch, rebind, or wrap the public entry" in prompt


def test_profile_reducer_reads_one_enriched_packet_without_discovery(tmp_path):
    reducer = tmp_path / "reducer.json"
    prompt = _profile_context(tmp_path).build_reducer_prompt(
        act_id="act-reducer",
        iteration_id="iter-000003",
        selected_version_id="v0",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "digest.json",
        research_state_path=tmp_path / "research.json",
        reducer_input_path=reducer,
    )

    assert f"Single bounded input: {reducer.resolve()}" in prompt
    assert str(
        (tmp_path / "candidate/.agentbench/research_state_update.json").resolve()
    ) in prompt
    assert "`workspace/.agentbench/research_state_update.json`" not in prompt
    assert "It embeds game_digest and research_state" in prompt
    assert "Do not run `wc`, `rg`, `find`, `ls`, or exploratory `jq keys`" in prompt


@pytest.mark.parametrize("kind", ["bootstrap", "candidate", "repair", "reducer"])
def test_all_profile_prompts_are_game_neutral(tmp_path, kind):
    context = _profile_context(tmp_path)
    common = {
        "act_id": f"act-{kind}",
        "workspace": tmp_path / "candidate",
    }
    if kind == "bootstrap":
        prompt = context.build_bootstrap_prompt(
            **common,
            experience_path=tmp_path / "experience" / "SKILL.md",
        )
    elif kind == "candidate":
        prompt = context.build_candidate_prompt(
            **common,
            branch_index=0,
            branch_count=4,
            parent_version_id="v0",
            game_digest_path=tmp_path / "digest.json",
            research_state_path=tmp_path / "research.json",
            replay_evidence=[
                {
                    "opponent": "human-01",
                    "candidate_role": "P0",
                    "summary": str(tmp_path / "summary.json"),
                }
            ],
            previous_measurements={"live_win_rate": 0.25},
            experience_path=tmp_path / "experience" / "SKILL.md",
            branch_brief={
                "branch_index": 0,
                "diagnosis": "round 8: camp lost health",
                "mechanism": "counterattack after visible damage",
                "activation_condition": "camp hp decreased",
                "preservation_contract": "otherwise retain parent ordering",
                "expected_change": "larger camp margin",
                "falsifier": "no margin gain",
                "code_symbols": ["AI.choose_operations", "AI.counterattack"],
            },
        )
    elif kind == "repair":
        prompt = context.build_repair_prompt(
            **common,
            iteration_id="iter-1",
            branch_index=0,
            game_digest_path=tmp_path / "digest.json",
            research_state_path=tmp_path / "research.json",
            repair_input_path=tmp_path / "repair.json",
            experience_path=tmp_path / "experience" / "SKILL.md",
        )
    else:
        prompt = context.build_reducer_prompt(
            **common,
            iteration_id="iter-1",
            selected_version_id="v1",
            game_digest_path=tmp_path / "digest.json",
            research_state_path=tmp_path / "research.json",
            reducer_input_path=tmp_path / "reducer.json",
        )

    assert "colony policy" in prompt
    assert "human colony" in prompt
    assert "P0" in prompt and "P1" in prompt
    assert "AI.choose_operations" in prompt
    assert "grid search" in prompt
    if kind == "reducer":
        assert "changed_action_count=0" in prompt
        assert "集成失败" in prompt
        assert "至少一个原子动作" in prompt
    if kind == "repair":
        assert "candidate_code_slices" in prompt
        assert "Do not read the digest, research state, Experience Skill" in prompt
        assert "next tool call must edit" in prompt
        assert "smoke_contract.command" in prompt
        assert "never invert its value function" in prompt
        assert "arbitrary divergence solely to satisfy `changed_action_count`" in prompt
        assert "report the mechanism as redundant" in prompt
    for forbidden in ("Rollman", "Ghost", "pacman", "rank15", "rank16"):
        assert forbidden not in prompt


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


def test_planner_distillation_assigns_offensive_and_predictive_branch_roles(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )
    shared = tmp_path / "shared-distillation.json"
    shared.write_text("{}\n", encoding="utf-8")
    context = IterationContext(bundle)
    common = {
        "act_id": "act-planner",
        "iteration_id": "iter-000012",
        "parent_version_id": "v000055",
        "workspace": tmp_path / "candidate",
        "game_digest_path": tmp_path / "digest.json",
        "research_state_path": tmp_path / "research.json",
        "replay_evidence": [{"summary": "summary.md"}],
        "active_target": "rank15",
    }

    prompt = context.build_planner_prompt(
        **common,
        previous_measurements={"opponent_distillation_path": str(shared)},
    )
    ordinary_prompt = context.build_planner_prompt(
        **common,
        previous_measurements={},
    )

    assert "branch 0：rank15 对手得分来源抑制" in prompt
    assert "branch 1：rank16 进攻得分或完成关卡" in prompt
    assert "branch 2：跨回放 Ghost 蒸馏与 best response" in prompt
    assert "branch 3：泛化与策略整合" in prompt
    assert "至少两支必须以推进、得分、完成关卡或压制对手得分为主目标" in prompt
    assert "不能直接复制 Ghost 动作" in prompt
    assert "不得重复运行蒸馏脚本" in prompt
    assert "branch 0：rank15 对手得分来源抑制" in ordinary_prompt
    assert "branch 1：rank16 进攻得分或完成关卡" in ordinary_prompt


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


def test_planner_prompt_emits_exact_read_allowlist_and_forbids_discovery(tmp_path):
    """Catch planners adding globs after the framework supplied exact inputs."""
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )
    digest = tmp_path / "digest.json"
    research = tmp_path / "research.json"
    summary = tmp_path / "matches" / "rank15" / "summary.md"
    shared = tmp_path / "shared-distillation.json"
    planner_input = tmp_path / "planner_input.json"
    planner_input.write_text("{}\n", encoding="utf-8")
    prompt = IterationContext(bundle).build_planner_prompt(
        act_id="act-planner",
        iteration_id="iter-000002",
        parent_version_id="v000000",
        workspace=tmp_path / "candidate",
        game_digest_path=digest,
        research_state_path=research,
        replay_evidence=[{"summary": str(summary)}],
        previous_measurements={"opponent_distillation_path": str(shared)},
        active_target="rank15",
        planner_input_path=planner_input,
    )

    expected_allowlist = json.dumps(
        [str(planner_input.resolve())],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert f"planner 精确只读白名单：{expected_allowlist}" in prompt
    assert "禁止使用 glob、通配符、find、目录列举或路径发现" in prompt
    assert "白名单没有共享蒸馏文件时，视为该输入不存在" in prompt
    assert "planner packet 已内嵌全部 summary_text" in prompt
    assert "只运行一次 `cat planner_input.json` 等价的单文件读取" in prompt
    assert str(shared.resolve()) not in prompt


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


def test_curriculum_prompt_accepts_evidence_from_locked_hard_opponent(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _static_files(tmp_path / "assets"),
    )

    prompt = IterationContext(bundle).build_prompt(
        act_id="act-0003",
        branch_index=0,
        branch_count=4,
        parent_version_id="v000001",
        workspace=tmp_path / "candidate",
        replay_evidence=[
            {
                "opponent": "rank15",
                "seed": 101,
                "replay": "/matches/rank15/replay.jsonl",
            },
            {
                "opponent": "rank16",
                "seed": 101,
                "replay": "/matches/rank16/replay.jsonl",
            },
        ],
        previous_measurements={"benchmark_score": 0.0},
        experience_path=tmp_path / "experience.md",
        active_target="rank15",
        locked_opponents=("rank16",),
    )

    assert "rank15" in prompt
    assert "rank16" in prompt


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
    assert "code_symbols" in planner
    assert "2–8" in planner
    assert "必须包含 `ai_func`" in planner
    assert "preservation_contract" in planner
    assert "time-expanded escape search" in candidate
    assert "候选 3/4" in candidate
    assert "完整重读" in candidate
    assert "触发条件外" in candidate
    assert "保持父代" in candidate
    assert "不得修改全局 scorer" in candidate
    assert "第 5 次工具调用结束前" in candidate
    assert "首次可编译修改" in candidate
    assert "不得顺序打印完整 ai.py" in candidate
    assert "命令必须直接引用白名单中的完整文件路径" in candidate
    assert "不得把 run 根目录或父目录保存为变量" in candidate
    assert str(repair_input) in repair
    assert "错误诊断" in repair
    assert "过宽" in repair
    assert "不得切换到其他 branch" in repair
    assert "最多 2 个" in repair
    assert "第 6 次工具调用结束前" in repair
    assert "首次可编译修复" in repair
    assert "批量读取" in repair
    assert "命令必须直接引用白名单中的完整文件路径" in repair
    assert "不得把 run 根目录或父目录保存为变量" in repair
    assert str(reducer_input) in reducer
    assert "research_state_update.json" in reducer
    assert "positive_margin_deltas" in reducer
    assert "可观察状态谓词" in reducer
    assert "不得按 seed" in reducer
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
            "code_symbols": ["ai_func", "_junction_escape"],
        },
        active_target="rank15",
        scope_contract_required=False,
    )

    assert "one safe exit" in prompt
    assert "diagnostic-only" in prompt
    assert "不得修改全局 scorer" not in prompt


def test_candidate_prompt_uses_one_prebuilt_context_packet(tmp_path):
    from agentbench_frame.hl.context import ContextBundle, IterationContext

    bundle = ContextBundle.create(
        tmp_path / "bundle",
        _digest_files(tmp_path / "assets"),
    )
    packet = tmp_path / "candidate_input-b00.json"
    packet.write_text("{}\n", encoding="utf-8")
    prompt = IterationContext(bundle).build_candidate_prompt(
        act_id="act-b00",
        branch_index=0,
        branch_count=4,
        parent_version_id="v000041",
        workspace=tmp_path / "candidate",
        game_digest_path=tmp_path / "game_digest.json",
        research_state_path=tmp_path / "research_state.json",
        replay_evidence=[
            {
                "opponent": "rank15",
                "summary": "/matches/seed-101/summary.md",
                "trace": "/matches/seed-101/trace.jsonl",
            }
        ],
        previous_measurements={"score": 0.25},
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
        candidate_input_path=packet,
    )

    assert str(packet) in prompt
    assert "第一次调用只读取 candidate input packet" in prompt
    assert "candidate-context-contract: rollman-v2" in prompt
    assert "candidate_code_slices" in prompt
    assert "smoke_contract" in prompt
    assert "exact `smoke_contract.command`" in prompt
    assert "marked `truncated`" in prompt
    assert "第 5 次工具调用" in prompt
    assert "不得顺序打印完整 ai.py" in prompt
    assert "必须命中新机制的 activation_condition" in prompt
    assert "公开入口 `ai_func` 返回新增分支的 `memory_id`" in prompt
    assert "直接调用内部 helper 不算" in prompt
    assert "replay_evidence" in prompt
    assert f"cat {packet}" in prompt
    assert "逐个读取 summary" not in prompt


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

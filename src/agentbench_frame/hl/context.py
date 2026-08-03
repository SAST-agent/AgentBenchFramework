"""Hashed static context, incremental prompts, and recoverable checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from agentbench_frame.hl.game_profile import PromptProfile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ContextBundle:
    root: Path
    manifest_path: Path
    bundle_hash: str
    files: Mapping[str, Path]

    @classmethod
    def create(
        cls,
        root: str | Path,
        sources: Mapping[str, str | Path],
    ) -> "ContextBundle":
        target_root = Path(root)
        files_root = target_root / "files"
        files_root.mkdir(parents=True, exist_ok=True)
        copied: dict[str, Path] = {}
        manifest_files: dict[str, dict[str, Any]] = {}
        for name, source_value in sorted(sources.items()):
            source = Path(source_value)
            if not source.exists():
                raise FileNotFoundError(source)
            if source.is_dir():
                destination_root = files_root / name
                resources: list[str] = []
                resource_hashes: dict[str, str] = {}
                for source_file in sorted(source.rglob("*")):
                    if (
                        not source_file.is_file()
                        or "__pycache__" in source_file.parts
                        or source_file.suffix == ".pyc"
                    ):
                        continue
                    relative = source_file.relative_to(source)
                    destination_file = destination_root / relative
                    destination_file.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    shutil.copy2(source_file, destination_file)
                    resource = relative.as_posix()
                    resources.append(resource)
                    resource_hashes[resource] = _sha256(destination_file)
                destination = destination_root / "SKILL.md"
                if not destination.is_file():
                    raise FileNotFoundError(
                        f"skill package has no SKILL.md: {source}"
                    )
                package_hash = hashlib.sha256(
                    json.dumps(
                        resource_hashes,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                copied[name] = destination.resolve()
                manifest_files[name] = {
                    "path": str(destination.resolve()),
                    "sha256": package_hash,
                    "resources": resources,
                }
                continue
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = files_root / name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            copied[name] = destination.resolve()
            manifest_files[name] = {
                "path": str(destination.resolve()),
                "sha256": _sha256(destination),
            }
        canonical = json.dumps(
            manifest_files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        bundle_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": "1.0",
            "bundle_hash": bundle_hash,
            "files": manifest_files,
        }
        manifest_path = target_root / "context-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return cls(
            root=target_root.resolve(),
            manifest_path=manifest_path.resolve(),
            bundle_hash=bundle_hash,
            files=copied,
        )


def compile_game_digest(
    bundle: ContextBundle,
    destination: str | Path,
) -> Path:
    """Compile a deterministic index without inventing tactical semantics."""

    decision_value = yaml.safe_load(
        bundle.files["decision_space"].read_text(encoding="utf-8")
    )
    if not isinstance(decision_value, Mapping):
        raise ValueError("decision space must be a mapping")
    policy_interface = decision_value.get("policy_interface")
    if not isinstance(policy_interface, Mapping):
        raise ValueError("decision space must define policy_interface")
    input_type = policy_interface.get("input_type")
    if not isinstance(input_type, str) or not input_type.strip():
        raise ValueError("policy_interface.input_type must be a non-empty string")
    object_fields = policy_interface.get("object_fields")
    if not isinstance(object_fields, list) or not all(
        isinstance(field, str) and field for field in object_fields
    ):
        raise ValueError("policy_interface.object_fields must be strings")
    normalized_mapping = policy_interface.get("normalized_state_mapping")
    if not isinstance(normalized_mapping, Mapping):
        raise ValueError("policy_interface must define normalized_state_mapping")
    roles = decision_value.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("decision space must define roles")
    if not roles:
        raise ValueError("decision space roles cannot be empty")
    normalized_roles: dict[str, dict[str, Any]] = {}
    for role_name, raw_role in roles.items():
        if not isinstance(role_name, str) or not role_name.strip():
            raise ValueError("decision space role names must be non-empty strings")
        if not isinstance(raw_role, Mapping):
            raise ValueError(f"decision space role {role_name} must be a mapping")
        role_id = raw_role.get("role_id")
        output_shape = raw_role.get("output_shape")
        if isinstance(role_id, bool) or not isinstance(role_id, int):
            raise ValueError(f"decision space role {role_name} requires integer role_id")
        if not isinstance(output_shape, str) or not output_shape.strip():
            raise ValueError(f"decision space role {role_name} requires output_shape")
        normalized_role = dict(raw_role)
        actions = raw_role.get("actions")
        component_support = raw_role.get("component_support")
        if actions is not None:
            if not isinstance(actions, list) or not actions or not all(
                isinstance(item, Mapping) for item in actions
            ):
                raise ValueError(
                    f"decision space role {role_name} actions must be structured mappings"
                )
            normalized_role["actions"] = [dict(item) for item in actions]
        elif not isinstance(component_support, list) or not component_support:
            raise ValueError(
                f"decision space role {role_name} requires actions or component_support"
            )
        normalized_roles[role_name] = normalized_role

    rules_text = bundle.files["rules"].read_text(encoding="utf-8")
    headings = [
        match.group(1).strip()
        for line in rules_text.splitlines()
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
    ]
    skill_text = bundle.files["replay_skill"].read_text(encoding="utf-8")
    skill_metadata: Mapping[str, Any] = {}
    if skill_text.startswith("---\n"):
        _, front_matter, _ = skill_text.split("---", 2)
        parsed_metadata = yaml.safe_load(front_matter)
        if isinstance(parsed_metadata, Mapping):
            skill_metadata = parsed_metadata

    value = {
        "schema_version": "1.0",
        "context_bundle_hash": bundle.bundle_hash,
        "context_manifest": str(bundle.manifest_path),
        "policy_interface": {
            "input_type": input_type,
            "object_fields": list(object_fields),
            "canonical_normalization": str(
                policy_interface.get("canonical_normalization") or ""
            ),
            "normalized_state_mapping": dict(normalized_mapping),
        },
        "roles": normalized_roles,
        "rule_sections": headings,
        "replay_skill": {
            "name": str(skill_metadata.get("name") or ""),
            "description": str(skill_metadata.get("description") or ""),
            "path": str(bundle.files["replay_skill"]),
        },
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


class IterationContext:
    """Build small act prompts that point at versioned files and artifacts."""

    def __init__(
        self,
        bundle: ContextBundle,
        *,
        prompt_profile: PromptProfile | None = None,
    ) -> None:
        self.bundle = bundle
        self.prompt_profile = prompt_profile

    def _build_profile_planner_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        parent_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        replay_evidence: list[Mapping[str, Any]],
        previous_measurements: Mapping[str, Any],
        active_target: Optional[str],
        scope_contract_required: bool,
        planner_input_path: str | Path | None,
    ) -> str:
        profile = self.prompt_profile
        assert profile is not None
        roles = ", ".join(profile.roles)
        diversity = "\n".join(
            f"- {item}" for item in profile.planner_diversity
        )
        prohibited = "\n".join(
            f"- {item}" for item in profile.prohibited_information
        )
        packet = (
            str(Path(planner_input_path).resolve())
            if planner_input_path is not None
            else "not provided; use the bounded paths below"
        )
        evidence = json.dumps(
            replay_evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        measurements = json.dumps(
            previous_measurements,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"""# Generic HL hypothesis planner {act_id}

proposal cycle: {iteration_id}
common parent: {parent_version_id}
candidate: {profile.candidate_label}
opponent: {profile.opponent_label}
candidate roles: {roles}
active target: {active_target or "none"}
policy input: {profile.policy_input}
output contract: {profile.output_contract}

Bounded inputs:
- planner packet: {packet}
- compact game digest: {Path(game_digest_path).resolve()}
- context manifest: {self.bundle.manifest_path.resolve()}
- research state: {Path(research_state_path).resolve()}
- candidate workspace: {Path(workspace).resolve()}
- replay evidence: {evidence}
- previous measurements: {measurements}

Read the packet once when present. It embeds the digest, manifest index, research state, bounded replay summaries, measurements and candidate code index. Consult an authoritative context file only when a precise rule or API question remains. Never print a complete replay, trace, board stream, or policy source.

Produce exactly four sibling hypotheses from the same parent. Each must contain a replay-grounded causal diagnosis, an observable activation condition, a mechanism, a preservation contract, an expected measurable change, a falsifier, and exact code symbols. The branches must differ in mechanism, not merely thresholds, weights, or parameter values. Do not perform grid search.

Required diversity axes:
{diversity}

Prohibited information and shortcuts:
{prohibited}

The atomic decision space in the frozen digest is authoritative for behavior measurement and KL. Do not invent tactical labels or latent hypothesis spaces. Source size and additional evidence-backed branches are not penalties. At least two branches must attempt proactive scoring, progress, resource acquisition, or direct suppression of the opponent rather than making all branches conservative.

At least one branch must distill a reusable public state → opponent atomic operation pattern from the supplied replay evidence, then propose an observable counter-response or an interpretable imitation of that response. Distillation may use only public state and accepted atomic operations; it must not depend on opponent identity, seed, fixed replay coordinates, hidden intent, or opponent source.

Write `workspace/.agentbench/branch_briefs.json` by the second tool call, validate its strict JSON shape once, and stop. Each of the four objects must use branch_index 0..3 and the fields diagnosis, mechanism, activation_condition, preservation_contract, expected_change, falsifier, and code_symbols. `code_symbols` must contain 2–8 unique names from candidate_code_index and include the public policy entry point. Scope contract: {"required" if scope_contract_required else "diagnostic-only"}.
"""

    def _profile_contract(self) -> str:
        profile = self.prompt_profile
        assert profile is not None
        prohibited = "\n".join(
            f"- {item}" for item in profile.prohibited_information
        )
        return f"""candidate: {profile.candidate_label}
opponent: {profile.opponent_label}
candidate roles: {", ".join(profile.roles)}
policy input: {profile.policy_input}
policy entry: {profile.policy_entry_symbol}
policy source: {profile.candidate_source_relative}
output contract: {profile.output_contract}

Forbidden information and shortcuts:
{prohibited}
"""

    def _build_profile_bootstrap_prompt(
        self,
        *,
        act_id: str,
        workspace: str | Path,
        experience_path: str | Path,
        bootstrap_input_path: str | Path | None,
    ) -> str:
        packet = (
            str(Path(bootstrap_input_path).resolve())
            if bootstrap_input_path is not None
            else "not provided; read the context manifest"
        )
        return f"""# Generic HL bootstrap {act_id}

Create the scientific origin policy from the frozen game specification. It must be interpretable, runnable, reproducible, and materially stronger than the placeholder.

{self._profile_contract()}
Authoritative inputs:
- bootstrap packet: {packet}
- context manifest: {self.bundle.manifest_path.resolve()}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}

Read the bootstrap packet exactly once. It embeds the rules, literal atomic decision space, SDK interface, Replay Skill, context manifest, and candidate entry source. Do not reopen those static files. Inspect only the exact workspace support or SDK symbol needed to resolve a concrete implementation ambiguity, then implement the policy through the public entry point. There is no replay evidence in this act; do not fabricate feedback or experience.

Checkpoint-first budget: after the packet, use at most five targeted workspace/SDK reads. By the eighth tool call, write the first compilable implementation to the policy source. Prefer a coherent simple economy/defense/offense controller over exhaustive engine inspection; live matches will supply the evidence for later refinement.

The atomic operations in the decision-space file are the only behavioral vocabulary. Do not invent tactical labels or a latent hypothesis space. Interpretable code may use conditionals, search, planning, state machines, finite memory, scoring functions, or their composition. Source growth and additional evidence-backed branches are not penalties.

Do not run grid search, enumerate arbitrary thresholds, memorize seeds, fixed replay coordinates, opponent identities, or read opponent source. Compile the candidate and run the workspace smoke command exactly once. Stop after the first successful compile and smoke validation. Do not edit the Experience Skill in the bootstrap act.

The framework audits paths. Read only the manifest-indexed context, candidate workspace, and Experience Skill. Do not enumerate their parent directories or inspect other runs, versions, framework source, human submissions, or evaluation-only policies.
"""

    def _build_profile_candidate_prompt(
        self,
        *,
        act_id: str,
        branch_index: int,
        branch_count: int,
        parent_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        experience_path: str | Path,
        branch_brief: Mapping[str, Any],
        scope_contract_required: bool,
        candidate_input_path: str | Path | None,
    ) -> str:
        profile = self.prompt_profile
        assert profile is not None
        packet = (
            str(Path(candidate_input_path).resolve())
            if candidate_input_path is not None
            else "not provided; use only the explicit bounded inputs below"
        )
        scope = (
            "Implement activation_condition as an explicit visible-state gate and preserve the parent action-selection path outside it."
            if scope_contract_required
            else "Record activation and preservation conditions, but the scope gate is diagnostic-only in this ablation."
        )
        brief = json.dumps(
            dict(branch_brief),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        trace_tool = (
            self.bundle.files["replay_skill"].parent
            / "scripts"
            / "inspect_trace_window.py"
        )
        return f"""# Generic HL candidate {act_id}

proposal branch: {branch_index + 1}/{branch_count}
common parent: {parent_version_id}

{self._profile_contract()}
candidate-context-contract: generic-v1

Bounded inputs:
- candidate packet: {packet}
- compact game digest: embedded in candidate packet
- research state: embedded in candidate packet
- Experience Skill: embedded in candidate packet
- candidate workspace: {Path(workspace).resolve()}
- bounded trace inspector: {trace_tool.resolve()}

First tool call: read the candidate packet exactly once and must print `smoke_contract` together with the selected branch and code slices. It contains the compact digest, accumulated condition-scoped experience, bounded replay summaries, previous measurements, the branch brief, selected code slices, and the exact `smoke_contract.command`. Experience Skill is embedded; do not read Experience Skill separately. Complete `candidate_code_slices` are authoritative for the first edit. Do not run `sed`, `cat ai.py`, `rg`, `find`, or `ls` before the first edit, and do not create directories that the framework already created. A slice explicitly marked truncated permits one exact-range source read; the following tool call must edit. Never print a complete replay, trace, board stream, or source file.

The only branch brief is: {brief}
Implement this mechanism rather than switching branches. Ground the diagnosis in the supplied replay summary and, only when necessary, inspect at most two small trace windows with the supplied Replay Skill tool. {scope}

Use literal protocol atomic operations from the frozen decision space. A deterministic selected operation still defines one observed atomic decision for KL; do not invent probabilities, tactical labels, or a human-authored hypothesis space. Do not run grid search, arbitrary parameter enumeration, seed/coordinate/identity lookup, or inspect {profile.opponent_label} source. Code growth and explicit condition handling are allowed when grounded in evidence.

Second tool call: make the first compilable edit to `{profile.candidate_source_relative}` and write `workspace/.agentbench/experience_update.json` in the same file-change tool call, with exactly four string arrays: stable_knowledge, failed_hypotheses, replay_evidence, active_questions. Each lesson must state an observable condition, attempted action/mechanism, and observed outcome; classify unsupported claims as questions. Then compile and execute the exact `smoke_contract.command` from the packet. Do not discover, invent, or replace the smoke entrypoint. Resolve a remaining SDK ambiguity only if that exact smoke command fails; do not browse preemptively. If the policy child check reports complete and only host-side result serialization fails, report that infrastructure failure and stop; do not inspect, shim, or patch the fixture. Stop after success; do not run git commands or a second refactor.

Read only the packet, its explicitly authorized trace files, exact manifest files needed for ambiguity resolution, Experience Skill, and candidate workspace. Other runs, versions, submissions, framework internals, and evaluation-only policies are forbidden.
"""

    def _build_profile_repair_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        branch_index: int,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        repair_input_path: str | Path,
        experience_path: str | Path,
    ) -> str:
        profile = self.prompt_profile
        assert profile is not None
        return f"""# Generic HL scoped repair {act_id}

proposal cycle: {iteration_id}
branch index: {branch_index}

{self._profile_contract()}
Bounded inputs:
- compact game digest: {Path(game_digest_path).resolve()}
- research state: {Path(research_state_path).resolve()}
- repair packet: {Path(repair_input_path).resolve()}
- Experience Skill: {Path(experience_path).resolve()}
- candidate workspace: {Path(workspace).resolve()}

Read the bounded repair packet first. Compare parent and candidate on the same opponent, role, and seed. Classify failure as a wrong causal diagnosis, an over-broad activation condition, or an integration error. Repair only this mechanism and preserve the parent path outside the brief's activation condition. The initial sibling remains immutable, so a failed repair cannot erase it.

Use only literal protocol atomic operations. Do not run grid search, threshold enumeration, seed/coordinate/opponent-identity memorization, or inspect {profile.opponent_label} source. At most two trace windows may be inspected through the Replay Skill. Compile `{profile.candidate_source_relative}`, run one smoke validation, and write the four condition-scoped Experience arrays to `workspace/.agentbench/experience_update.json`. Stop after success.
"""

    def _build_profile_reducer_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        selected_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        reducer_input_path: str | Path,
    ) -> str:
        profile = self.prompt_profile
        assert profile is not None
        return f"""# Generic HL comparative reducer {act_id}

proposal cycle: {iteration_id}
selected search parent: {selected_version_id}

{self._profile_contract()}
Read-only inputs:
- compact game digest: {Path(game_digest_path).resolve()}
- research state: {Path(research_state_path).resolve()}
- factual sibling packet: {Path(reducer_input_path).resolve()}
- selected candidate workspace: {Path(workspace).resolve()}

Compare all sibling mechanisms using framework-recorded outcomes for the same opponent, role, and seed. Match facts outrank model inference. Preserve conditional gains and failures, including exact dense-margin deltas and activation evidence; do not turn speculation into stable knowledge. Code size and local conditionals are not failure criteria.

Do not modify policy code, version pointers, curriculum, or certification facts. Do not run grid search or inspect {profile.opponent_label} source. Write exactly one `workspace/.agentbench/research_state_update.json` object with four arrays: stable_knowledge, failed_hypotheses, open_questions, recent_comparisons. The first three contain non-empty strings; comparisons are concise objects. Read, write, validate once, and stop.
"""

    def build_bootstrap_prompt(
        self,
        *,
        act_id: str,
        workspace: str | Path,
        experience_path: str | Path,
        bootstrap_input_path: str | Path | None = None,
    ) -> str:
        if self.prompt_profile is not None:
            return self._build_profile_bootstrap_prompt(
                act_id=act_id,
                workspace=workspace,
                experience_path=experience_path,
                bootstrap_input_path=bootstrap_input_path,
            )
        return f"""# HL bootstrap {act_id} — 生成科研 origin

目标：根据冻结游戏规则，从规则出发设计并实现一版可解释、可运行、可复现的初始算法。该版本将作为后续 HL 迭代的 origin。

实时策略输入是冻结 SDK 的 `core.gamedata.GameState` 对象。必须读取规则第 9 节，优先调用 `game_state.gamestate_to_statedict()`；不要把对象属性误写成回放字段。规范化结果中的 board、坐标和技能状态是 NumPy 数组，禁止 `array or []`、`if array` 等隐式布尔判断。smoke test 必须使用真实 `numpy.ndarray` 覆盖 `pacman_pos`、`ghosts_pos`、`pacman_score`、`ghosts_score`，并确认首个合法状态不返回 STAY fallback。

只读上下文：
- context manifest: {self.bundle.manifest_path}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}

本阶段没有比赛回放。不要虚构回放证据，也不要假装从反馈中得出结论。

Act 预算：
- 最多 10 次工具调用；优先批量读取规则与接口，禁止反复查看同一文件。
- 写完策略后只运行一次编译和一次使用真实 NumPy 数组的对象 smoke test；两者成功后立即结束。

科研隔离边界：
- 只允许读取上述 context manifest 及其 files、candidate workspace 和 Experience Skill。
- 禁止读取或搜索其他 run、其他候选目录、Framework 源码、人类程序、对手构建目录及用户目录中的其他文件。
- 禁止列举 candidate workspace 或指定 context 目录的父目录，禁止使用 `..` 绕过边界。
- Framework 会审计工具调用路径和原始记录；越界 act 会被标记失败，不评测、不晋级。

执行约束：
1. 完整阅读 context manifest 指向的规则、原子决策空间和 Replay Skill，再阅读 workspace 中的候选接口与脚手架代码。
2. 基于可见 GameState 和合法原子动作，设计一套机制连贯的初始策略；允许路径规划、搜索、状态机、有限记忆及其他可解释代码。
3. 初始算法必须显著优于占位策略，并在代码结构或注释中清楚表达决策依据。
4. 禁止无依据的参数枚举、grid search、seed/固定坐标记忆和人工战术标签。
5. 不读取、搜索或推断人类对手源码。
6. 允许策略代码增长和增加新的情形分支；不以源代码长度或 if/else 数量作为惩罚。
7. 完成框架指定的静态检查和 smoke test。
8. 不要直接修改 Experience Skill；本阶段只建立初始算法，后续再从合法比赛回放更新经验。
9. 第一次编译与真实类型 smoke 成功后立即结束；禁止重复执行测试、git status/diff、额外润色或第二轮重构，真实比赛会负责验证。
"""

    def build_prompt(
        self,
        *,
        act_id: str,
        branch_index: int,
        branch_count: int,
        parent_version_id: str,
        workspace: str | Path,
        replay_evidence: list[Mapping[str, Any]],
        previous_measurements: Mapping[str, Any],
        experience_path: str | Path,
        active_target: Optional[str] = None,
        locked_opponents: Sequence[str] = (),
    ) -> str:
        if not 0 <= branch_index < branch_count:
            raise ValueError("branch_index must be inside branch_count")
        if active_target is not None:
            allowed_opponents = {active_target, *locked_opponents}
            wrong_opponents = sorted(
                {
                    str(item.get("opponent"))
                    for item in replay_evidence
                    if item.get("opponent") not in allowed_opponents
                }
            )
            if wrong_opponents:
                raise ValueError(
                    "replay evidence must match the active target: "
                    f"{wrong_opponents}"
                )
        evidence = json.dumps(
            replay_evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        measurements = json.dumps(
            previous_measurements,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        trace_window_tool = (
            self.bundle.files["replay_skill"].parent
            / "scripts"
            / "inspect_trace_window.py"
        )
        distillation_tool = (
            self.bundle.files["replay_skill"].parent
            / "scripts"
            / "distill_opponent_policy.py"
        )
        stagnation_count = int(
            previous_measurements.get("curriculum_stagnation_count") or 0
        )
        shared_distillation = previous_measurements.get(
            "opponent_distillation_path"
        )
        distillation = ""
        if shared_distillation:
            trigger_label = (
                f"连续无提升 {stagnation_count} 轮"
                if stagnation_count >= 3
                else "继承研究债务达到阈值"
            )
            distillation = f"""
停滞干预（{trigger_label}）：
- 读取共享 Ghost 蒸馏：`{Path(str(shared_distillation)).resolve()}`；不得重复运行蒸馏脚本。
- 蒸馏只含相对几何和原子 Ghost 动作统计；用 fine table + coarse backoff 构造可解释预测器。
- 我方角色是 Rollman，不能复制 Ghost 动作；应预测 Ghost 下一步路径后选择 Rollman 动作。
- 不得把 seed、绝对坐标、对手身份或 replay ID 写入策略；Rollman 的 KL 决策空间保持不变。
"""
        elif stagnation_count >= 3:
            distillation = f"""
停滞干预（连续无提升 {stagnation_count} 轮）：
- 检验“可预测的 Ghost 行为能否支持 best response”，同时允许保留有回放证据支持的局部规则。
- 对 evidence 中全部 trace 一次性运行 `{distillation_tool} TRACE1 TRACE2 TRACE3`。
- 蒸馏输出只含相对几何和原子 Ghost 动作统计；用 fine table + coarse backoff 构造可解释预测器。
- 我方角色是 Rollman，不能复制 Ghost 动作；应预测 Ghost 下一步路径后选择 Rollman 动作。
- 不得把 seed、绝对坐标、对手身份或 replay ID 写入策略；Rollman 的 KL 决策空间保持不变。
"""
        curriculum = ""
        if active_target is not None:
            locked = json.dumps(
                list(locked_opponents),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            curriculum = f"""
课程阶段：
- 当前学习目标：{active_target}
- 已锁定通过的对手：{locked}
- 本轮诊断和策略改动只针对当前目标；全池认证负责检查已锁定对手是否退化。
- 允许以可观测状态为依据的 if/else、状态机、路径规划、图搜索和有限记忆。
- 禁止针对 seed、固定回放坐标或对手身份硬编码，禁止人工战术标签。
"""
        return f"""# HL iteration {act_id} — 候选 {branch_index + 1}/{branch_count}

目标：在冻结评测协议下提升游戏 agent，保持程序可解释、可运行、可复现。

实时策略输入是冻结 SDK 的 `core.gamedata.GameState` 对象；规范化入口是 `game_state.gamestate_to_statedict()`。对象属性使用 `pacman_pos`、`ghosts_pos`、`pacman_score`、`ghosts_score`，而回放/规范字典使用 `pacman_coord`、`ghosts_coord`、`score`。不得混淆两层字段。规范字典中的 board、坐标和技能状态是 NumPy 数组，禁止对数组使用隐式布尔判断（例如 `array or []` 或 `if array`）。

只读上下文：
- context manifest: {self.bundle.manifest_path}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}
- parent version: {parent_version_id}
{curriculum}

上一轮测量：{measurements}
必须核查的回放证据：{evidence}
{distillation}

科研隔离边界：
- 只允许读取上述 context manifest 及其 files、candidate workspace、Experience Skill，以及“必须核查的回放证据”明确列出的 replay/trace。
- 禁止读取或搜索其他 run、其他候选目录、Framework 源码、人类程序、对手构建目录及用户目录中的其他文件。
- 禁止列举 candidate workspace、指定 context 或指定回放目录的父目录，禁止使用 `..` 绕过边界。
- Framework 会审计工具调用路径和原始记录；越界 act 会被标记失败，不评测、不晋级。

Act 预算：
- 最多 10 次工具调用；优先批量读取，禁止用许多小命令反复查看同一材料。
- 必须先读取 evidence 中的 `summary`；不得打印完整 replay、完整 trace、完整棋盘或全量事件流。
- 只允许对最多 2 个可证伪假设做定点探针；trace 必须用 `{trace_window_tool} TRACE --level L --round R --radius 1` 读取。
- 除停滞干预指定的 `{distillation_tool}` 外，禁止用 cat、sed、head、tail、rg 或自行脚本读取 trace。两个工具单次输出均不超过 64 KiB，单条定点 trace 最多请求 20 个中心回合。
- 完成一次证据诊断后立即实现最小机制改动并验证；禁止在同一 act 内形成参数搜索循环。

执行约束：
1. 先阅读 context manifest 指向的规则、决策空间和 Replay Skill，再阅读 workspace 中的代码。
2. 从给定回放中引用至少一个具体 level/round/事件，提出一个可证伪的因果诊断。
3. 实现一个机制连贯、可泛化的改进；允许搜索、路径规划、状态机、记忆和其他可解释代码。
4. 禁止无依据的参数枚举或 grid search。只在回放证据直接指向决策边界时修改数值。
5. 若一轮有多个候选，本候选必须与同轮其他候选机制上不同，不能只是换阈值。
6. 允许策略代码增长和堆叠有证据支持的情形分支；不以代码长度或 if/else 数量作为惩罚。
7. 不读取、搜索或推断人类对手源码。只能从合法比赛回放学习。
8. 只在 candidate workspace 内完成 `python -m py_compile ai.py` 和候选侧 smoke test，不搜索 Framework 命令。不要直接修改 Experience Skill；将四个字符串数组 stable_knowledge、failed_hypotheses、replay_evidence、active_questions 写入 workspace/.agentbench/experience_update.json，由 Framework 在候选测量完整通过后合并。
9. 第一次编译、对象输入 smoke 和 experience JSON 成功后立即结束；禁止再运行 git status/diff、重复读取、额外边缘润色或第二轮重构，真实比赛会负责验证。
"""

    def build_planner_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        parent_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        replay_evidence: list[Mapping[str, Any]],
        previous_measurements: Mapping[str, Any],
        active_target: Optional[str] = None,
        scope_contract_required: bool = True,
        planner_input_path: str | Path | None = None,
    ) -> str:
        if self.prompt_profile is not None:
            return self._build_profile_planner_prompt(
                act_id=act_id,
                iteration_id=iteration_id,
                parent_version_id=parent_version_id,
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                replay_evidence=replay_evidence,
                previous_measurements=previous_measurements,
                active_target=active_target,
                scope_contract_required=scope_contract_required,
                planner_input_path=planner_input_path,
            )
        evidence = json.dumps(
            replay_evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        measurements = json.dumps(
            previous_measurements,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stagnation_count = int(
            previous_measurements.get("curriculum_stagnation_count") or 0
        )
        shared_distillation = previous_measurements.get(
            "opponent_distillation_path"
        )
        if planner_input_path is not None:
            resolved_planner_input = Path(planner_input_path).resolve()
            planner_read_paths = [str(resolved_planner_input)]
            read_inputs = f"- bounded planner packet: {resolved_planner_input}"
            packet_boundary = """
planner packet 已内嵌全部 summary_text、game digest、context manifest、research state、previous measurements 与可选 Ghost 蒸馏。只运行一次 `cat planner_input.json` 等价的单文件读取；不得再读取 packet 内提到或暗示的任何路径。
"""
            first_read_instruction = (
                "1. 第一次且唯一一次工具调用读取 planner_input.json；"
                "不要拆成多个并行 cat。"
            )
        else:
            planner_read_paths = [
                str(Path(game_digest_path).resolve()),
                str(self.bundle.manifest_path.resolve()),
                str(Path(research_state_path).resolve()),
            ]
            planner_read_paths.extend(
                str(Path(str(item["summary"])).resolve())
                for item in replay_evidence
                if item.get("summary")
            )
            if shared_distillation:
                planner_read_paths.append(
                    str(Path(str(shared_distillation)).resolve())
                )
            read_inputs = f"""- compact game digest: {Path(game_digest_path).resolve()}
- authoritative context manifest: {self.bundle.manifest_path}
- explicit research state: {Path(research_state_path).resolve()}
- previous measurements: {measurements}
- bounded replay evidence: {evidence}"""
            packet_boundary = ""
            first_read_instruction = (
                "1. 第一次工具调用批量读取 game digest、research state、"
                "全部 evidence summary，以及存在时的共享 Ghost 蒸馏；"
                "不要逐个读取四个 summary。"
            )
        planner_read_allowlist = json.dumps(
            list(dict.fromkeys(planner_read_paths)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        distillation = ""
        if shared_distillation and planner_input_path is None:
            distillation = f"""
共享 Ghost 蒸馏：{Path(str(shared_distillation)).resolve()}
先读取该坐标无关统计；不得重复运行蒸馏脚本。Rollman 只能预测 Ghost 行为后选择自身动作，不能直接复制 Ghost 动作。
"""
        elif shared_distillation:
            distillation = """
共享 Ghost 蒸馏已内嵌在 planner packet；不得读取外部蒸馏路径或重复运行蒸馏脚本。Rollman 只能预测 Ghost 行为后选择自身动作，不能直接复制 Ghost 动作。
"""
        branch_roles = """
四分支探索职责：
- branch 0：rank15 对手得分来源抑制。定位 rank15 最大的可观测 Ghost 得分来源，用有限作用域机制降低连续捕获。
- branch 1：rank16 进攻得分或完成关卡。提高 Rollman 采集、得分或 FINISH_LEVEL 效率，同时保持生存底线。
- branch 2：跨回放 Ghost 蒸馏与 best response。用 rank15/rank16 的相对几何原子动作统计预测 Ghost，再选择可解释 Rollman 响应。
- branch 3：泛化与策略整合。检查既有机制在哪些状态有效或退化，修复作用域或整合兼容经验，不得重复前三支。

至少两支必须以推进、得分、完成关卡或压制对手得分为主目标，不能让四支都以 veto、retreat、wait 或 avoid-contact 为主要机制。Rollman 的 KL 原子决策空间保持不变。四支不得只对同一回放做阈值变化。
"""
        return f"""# Rollman HL hypothesis planner {act_id}

proposal cycle: {iteration_id}
common parent: {parent_version_id}
active target: {active_target or "none"}

只读输入：
{read_inputs}
{distillation}
{branch_roles}

planner 精确只读白名单：{planner_read_allowlist}
第一次工具调用只能逐项直接引用上述完整文件路径并批量读取。禁止使用 glob、通配符、find、目录列举或路径发现，也不得扫描 run/context/workspace。白名单没有共享蒸馏文件时，视为该输入不存在，不得自行搜索替代文件。
{packet_boundary}

这是压缩假设规划，不是代码审查。planner packet 中的 `candidate_code_index` 是 Framework 解析的模块级函数名、signature 与行号；只能从这个索引选择符号。禁止读取 ai.py、列举 workspace 或运行符号搜索。候选 act 负责核对代码并实现机制。

先落盘：第 2 次工具调用结束前，必须已经写出一份结构合法、含恰好四项的 `workspace/.agentbench/branch_briefs.json`。第一次读取完成后立即完成有限推理并写文件；不得在写文件前继续浏览、长时间扩展分析或调用第三个工具。

读取顺序：
{first_read_instruction}
2. 第二次工具调用直接写入 branch_briefs.json；四项都必须引用摘要中的具体证据，并避开 research state 已否定的机制。
3. 写出后只允许一次 JSON 结构校验，然后立即结束。禁止补读代码、规则、replay 或 trace。

不要求每轮完整重读静态长文。不得读取人类对手源码、其他 run、其他候选版本或当前候选代码。

Planner 压缩边界：
- 只能读取 evidence 的 summary；不得打开 replay 或 trace，不得对它们运行脚本或自行解析。
- 不得打印完整 replay、完整 trace、完整 observation、完整棋盘或全量事件流。
- diagnosis 中的 level/round/事件必须来自 summary；精确窗口由后续候选 act 使用受限工具核查。
- 对照 research state，四个 mechanism 必须是未被既有失败证据否定的实质新机制；由候选 act 在实现前检查当前 ai.py，若机制已存在则在同一证据目标下实现缺失的控制环节。

最多 3 次工具调用。写出并校验 branch_briefs.json 后立即结束，不运行 git status/diff，不继续扩展诊断。

基于同一份证据，提出恰好 4 个机制上不同、可证伪的 Rollman 改进方向。禁止把同一机制的阈值、权重或参数变化伪装成四种方案；禁止 grid search。允许 if/else、路径规划、搜索、状态机、有限记忆和策略代码增长。

将严格 JSON 数组写入 workspace/.agentbench/branch_briefs.json。每项必须且只能包含：branch_index（0..3）、diagnosis、mechanism、activation_condition、preservation_contract、expected_change、falsifier、code_symbols。`code_symbols` 必须是 `candidate_code_index` 中 2–8 个互不重复的精确函数名，必须包含 `ai_func`，不得臆造 helper。diagnosis 必须引用具体回放 level/round/事件。activation_condition 必须是可观测状态谓词；preservation_contract 必须指出触发条件外保留的父代决策路径。作用域契约状态：{"required" if scope_contract_required else "diagnostic-only"}。不要修改候选策略代码。
"""

    def build_candidate_prompt(
        self,
        *,
        act_id: str,
        branch_index: int,
        branch_count: int,
        parent_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        replay_evidence: list[Mapping[str, Any]],
        previous_measurements: Mapping[str, Any],
        experience_path: str | Path,
        branch_brief: Mapping[str, Any],
        active_target: Optional[str] = None,
        locked_opponents: Sequence[str] = (),
        scope_contract_required: bool = True,
        candidate_input_path: str | Path | None = None,
    ) -> str:
        if int(branch_brief.get("branch_index", -1)) != branch_index:
            raise ValueError("branch brief index does not match candidate branch")
        if self.prompt_profile is not None:
            return self._build_profile_candidate_prompt(
                act_id=act_id,
                branch_index=branch_index,
                branch_count=branch_count,
                parent_version_id=parent_version_id,
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                experience_path=experience_path,
                branch_brief=branch_brief,
                scope_contract_required=scope_contract_required,
                candidate_input_path=candidate_input_path,
            )
        prompt_measurements = dict(previous_measurements)
        if candidate_input_path is not None:
            prompt_measurements.pop("opponent_distillation_path", None)
        base = self.build_prompt(
            act_id=act_id,
            branch_index=branch_index,
            branch_count=branch_count,
            parent_version_id=parent_version_id,
            workspace=workspace,
            replay_evidence=replay_evidence,
            previous_measurements=prompt_measurements,
            experience_path=experience_path,
            active_target=active_target,
            locked_opponents=locked_opponents,
        )
        base = base.replace(
            "- context manifest:",
            f"- compact game digest: {Path(game_digest_path).resolve()}\n"
            f"- explicit research state: {Path(research_state_path).resolve()}\n"
            "- authoritative context manifest:",
            1,
        ).replace(
            "1. 先阅读 context manifest 指向的规则、决策空间和 Replay Skill，再阅读 workspace 中的代码。",
            "1. 先读取 compact game digest、research state、指定 evidence summary 和 workspace 代码；只有改动依赖精确规则语义时才按 manifest 定点核对权威章节，不要求每轮完整重读静态长文。",
        )
        if candidate_input_path is not None:
            packet = Path(candidate_input_path).resolve()
            evidence = json.dumps(
                replay_evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            measurements = json.dumps(
                prompt_measurements,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            base = base.replace(
                f"- compact game digest: {Path(game_digest_path).resolve()}\n"
                f"- explicit research state: {Path(research_state_path).resolve()}\n"
                "- authoritative context manifest:",
                f"- candidate input packet: {packet}\n"
                "- authoritative context manifest:",
                1,
            ).replace(
                f"- Experience Skill: {Path(experience_path).resolve()}",
                "- Experience Skill: embedded in candidate input packet",
                1,
            ).replace(
                f"上一轮测量：{measurements}\n必须核查的回放证据：{evidence}",
                "上一轮测量与必须核查的回放证据：embedded in candidate input packet",
                1,
            ).replace(
                "- 必须先读取 evidence 中的 `summary`；不得打印完整 replay、完整 trace、完整棋盘或全量事件流。",
                "- candidate input packet 已内嵌全部有界 summary；不得再次逐项读取 summary 文件，不得打印完整 replay、完整 trace、完整棋盘或全量事件流。",
                1,
            ).replace(
                "1. 先读取 compact game digest、research state、指定 evidence summary 和 workspace 代码；只有改动依赖精确规则语义时才按 manifest 定点核对权威章节，不要求每轮完整重读静态长文。",
                "1. 第一次调用只读取 candidate input packet；随后读取 workspace 相关代码。只有改动依赖精确规则语义时才按 manifest 定点核对权威章节，不要求每轮完整重读静态长文。",
                1,
            )
        brief = json.dumps(
            dict(branch_brief),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scope_contract = (
            """
作用域契约（required）：
- activation_condition 必须实现为基于可观测状态的显式门控。
- 触发条件外保持父代 action-selection 路径；preservation_contract 指定的逻辑不得改变。
- 除非 activation_condition 为真，不得修改全局 scorer、预测器、权重或候选排序。
"""
            if scope_contract_required
            else """
作用域契约消融（diagnostic-only）：
- 记录 activation_condition 与 preservation_contract 供分析，但本候选不强制保持父代路径。
"""
        )
        packet_command = (
            f"cat {Path(candidate_input_path).resolve()}"
            if candidate_input_path is not None
            else "read the candidate input packet"
        )
        return base + f"""

candidate-context-contract: rollman-v2

本候选的唯一结构化 branch brief：{brief}
必须实现并检验这个机制；不得改做其他分支，也不得只调整无证据参数。
{scope_contract}

候选 checkpoint-first 顺序：
- 第一次调用必须直接执行 `{packet_command}`，完整读取一次后不得写脚本筛选或发现其中路径。顶层键为 `branch_brief`、`game_digest`、`research_state`、`experience_skill`、`replay_evidence`、`previous_measurements`、`opponent_distillation`、`candidate_code_index`、`candidate_code_slices`、`smoke_contract`；禁止再次分别读取已内嵌内容。
- `candidate_code_slices` 已含 planner 选择的精确实现区间。只有某一 selected slice 明确 marked `truncated` 时，才允许再读一次该函数的精确行号范围；否则禁止再次读取 ai.py，且任何情况下不得顺序打印完整 ai.py。
- `candidate_code_index` 与 slices 已给出模块级函数行号和 signature。新增代码调用前必须核对每个既有 helper 的 signature；不得凭函数名猜参数。
- 最多读取两个 packet 授权的定点 trace 窗口；第 5 次工具调用结束前必须已完成 `ai.py` 的首次可编译修改并写入 experience_update.json。
- 首次修改落盘后，只允许编译、一次 smoke，以及为修复验证失败所必需的一次更正。必须写 `smoke_contract.scenario_path` 的 JSON，然后执行 packet 中 exact `smoke_contract.command`；不得另写 Python fixture。smoke 必须命中新机制的 activation_condition，并断言公开入口 `ai_func` 返回新增分支的 `memory_id`，同时证明 preservation state 未进入新分支；直接调用内部 helper 不算，不能只验证父代 fallback；不得把实现留到长推理末尾。
- 命令必须直接引用白名单中的完整文件路径；不得把 run 根目录或父目录保存为变量后再拼接，也不得列举这些目录。
"""

    def build_repair_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        branch_index: int,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        repair_input_path: str | Path,
        experience_path: str | Path,
    ) -> str:
        if self.prompt_profile is not None:
            return self._build_profile_repair_prompt(
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=branch_index,
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                repair_input_path=repair_input_path,
                experience_path=experience_path,
            )
        trace_window_tool = (
            self.bundle.files["replay_skill"].parent
            / "scripts"
            / "inspect_trace_window.py"
        )
        return f"""# Rollman scoped repair {act_id}

proposal cycle: {iteration_id}
branch index: {branch_index}

只读输入：
- compact game digest: {Path(game_digest_path).resolve()}
- current research state: {Path(research_state_path).resolve()}
- bounded repair packet: {Path(repair_input_path).resolve()}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}

先读取 repair packet 指定的父代与候选同种子比分、回放 summary、branch brief 和运行错误。把失败归类为：错误诊断、activation_condition 过宽、或机制集成错误。只能修复同一个 branch；不得切换到其他 branch，不得参数枚举、grid search、seed/坐标/对手身份记忆。

修复约束：
1. 触发条件外保持父代 action-selection 路径和 preservation_contract。
2. 初始候选由 Framework 作为不可变版本保留；修复失败不会覆盖它。
3. 最多 2 个可证伪假设；trace 只能用 `{trace_window_tool} TRACE --level L --round R --radius 1` 做定点读取。
4. 只运行一次 `python -m py_compile ai.py` 和一次真实 NumPy 对象 smoke test。
5. 将四个字符串数组 stable_knowledge、failed_hypotheses、replay_evidence、active_questions 写入 workspace/.agentbench/experience_update.json。
6. 验证成功后立即结束；不得继续润色、git status/diff 或第二轮重构。

Repair checkpoint-first 顺序：
- repair packet 已内嵌有界 `summary_text`；第一次调用只批量读取这个 packet、game digest、research state 和 Experience Skill 四个精确路径，不得扫描目录或自行发现 summary。第二次只读 branch 相关代码区间，不得顺序打印完整 ai.py。
- 第三、四次最多完成两个定点 trace 窗口；第 6 次工具调用结束前必须已完成 `ai.py` 的首次可编译修复并写入 experience_update.json。
- 首次修复落盘后，只允许编译、一次对象 smoke，以及为修复验证失败所必需的一次更正；不得把实现留到长推理末尾。
- 命令必须直接引用白名单中的完整文件路径；不得把 run 根目录或父目录保存为变量后再拼接，也不得列举这些目录。

科研隔离边界：只能读取以上路径及 repair packet 明确列出的 summary/replay/trace；不得先声明或访问 run 根目录、其他版本、其他候选、人类源码或用户目录中的其他文件。
"""

    def build_reducer_prompt(
        self,
        *,
        act_id: str,
        iteration_id: str,
        selected_version_id: str,
        workspace: str | Path,
        game_digest_path: str | Path,
        research_state_path: str | Path,
        reducer_input_path: str | Path,
    ) -> str:
        if self.prompt_profile is not None:
            return self._build_profile_reducer_prompt(
                act_id=act_id,
                iteration_id=iteration_id,
                selected_version_id=selected_version_id,
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                reducer_input_path=reducer_input_path,
            )
        return f"""# Rollman HL comparative reducer {act_id}

proposal cycle: {iteration_id}
selected search parent: {selected_version_id}

只读输入：
- compact game digest: {Path(game_digest_path).resolve()}
- current research state: {Path(research_state_path).resolve()}
- four-candidate factual packet: {Path(reducer_input_path).resolve()}
- selected candidate workspace: {Path(workspace).resolve()}

比较四个机制的实际比赛反馈。Framework 记录的分数、回放和测量高于模型推断；不得把没有证据的解释写成稳定知识。策略膨胀和局部 if/else 不构成失败理由。不得修改候选代码、版本指针、对手课程或认证结论。

reducer input 中的 `positive_margin_deltas` 是 Framework 按同一对手、同一 seed 计算的候选相对父版本分差增益。必须在 recent_comparisons 中保留其中最强的条件性改进及精确 margin_delta，并在 open_questions 中提出区分“改进场景”和“退化场景”的可观察状态谓词。seed 只能作为复现实验证据，不得按 seed、固定坐标或回放帧号设计策略。

将严格 JSON 对象写入 workspace/.agentbench/research_state_update.json，必须且只能包含四个数组字段：stable_knowledge、failed_hypotheses、open_questions、recent_comparisons。前三项只能包含非空字符串；recent_comparisons 每项为简短对象。不得修改 proposal_cycle、search_parent、official_champion、active_target、locked_opponents 或 exploration_debt，这些字段由 Framework 根据事实维护。
最多 6 次工具调用；读取 reducer input 与 research state、写出并校验 JSON 后立即结束，不运行 git status/diff 或额外分析。
必须直接使用上面列出的精确文件路径；不得先声明或访问 run 根目录、这些文件的父目录，也不得通过父目录拼接路径。
"""


def write_checkpoint(
    path: str | Path,
    *,
    act_id: str,
    parent_version_id: str,
    bundle: ContextBundle,
    experience_path: str | Path,
    replay_ids: list[str],
    prompt: str,
    thread_id: Optional[str],
    provider_fingerprint: str,
) -> dict[str, Any]:
    experience = Path(experience_path)
    if not experience.is_file():
        raise FileNotFoundError(experience)
    record = {
        "schema_version": "1.0",
        "act_id": act_id,
        "parent_version_id": parent_version_id,
        "context_bundle_hash": bundle.bundle_hash,
        "context_manifest": str(bundle.manifest_path),
        "experience_path": str(experience.resolve()),
        "experience_hash": _sha256(experience),
        "replay_ids": list(replay_ids),
        "prompt": prompt,
        "thread_id": thread_id,
        "provider_fingerprint": provider_fingerprint,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return record

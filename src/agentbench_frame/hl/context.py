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
    if policy_interface.get("input_type") != "core.gamedata.GameState":
        raise ValueError("Rollman policy input must be core.gamedata.GameState")
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
    rollman = roles.get("rollman")
    ghosts = roles.get("ghosts")
    if not isinstance(rollman, Mapping) or not isinstance(ghosts, Mapping):
        raise ValueError("decision space must define rollman and ghosts")
    actions = rollman.get("actions")
    if not isinstance(actions, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("id"), int)
        for item in actions
    ):
        raise ValueError("rollman actions must be structured mappings")
    normalized_actions = [dict(item) for item in actions]
    if [item["id"] for item in normalized_actions] != [0, 1, 2, 3, 4]:
        raise ValueError("Rollman primitive action support must be exactly 0..4")

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
            "input_type": str(policy_interface["input_type"]),
            "object_fields": list(object_fields),
            "canonical_normalization": str(
                policy_interface.get("canonical_normalization") or ""
            ),
            "normalized_state_mapping": dict(normalized_mapping),
        },
        "roles": {
            "rollman": {
                "role_id": int(rollman["role_id"]),
                "output_shape": str(rollman["output_shape"]),
                "actions": normalized_actions,
            },
            "ghosts": {
                "role_id": int(ghosts["role_id"]),
                "output_shape": str(ghosts["output_shape"]),
                "component_support": list(ghosts["component_support"]),
            },
        },
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

    def __init__(self, bundle: ContextBundle) -> None:
        self.bundle = bundle

    def build_bootstrap_prompt(
        self,
        *,
        act_id: str,
        workspace: str | Path,
        experience_path: str | Path,
    ) -> str:
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
            wrong_opponents = sorted(
                {
                    str(item.get("opponent"))
                    for item in replay_evidence
                    if item.get("opponent") != active_target
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
        if stagnation_count >= 3:
            if shared_distillation:
                distillation = f"""
停滞干预（连续无提升 {stagnation_count} 轮）：
- 读取共享 Ghost 蒸馏：`{Path(str(shared_distillation)).resolve()}`；不得重复运行蒸馏脚本。
- 蒸馏只含相对几何和原子 Ghost 动作统计；用 fine table + coarse backoff 构造可解释预测器。
- 我方角色是 Rollman，不能复制 Ghost 动作；应预测 Ghost 下一步路径后选择 Rollman 动作。
- 不得把 seed、绝对坐标、对手身份或 replay ID 写入策略；Rollman 的 KL 决策空间保持不变。
"""
            else:
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
    ) -> str:
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
        distillation = ""
        if stagnation_count >= 3 and shared_distillation:
            distillation = f"""
共享 Ghost 蒸馏：{Path(str(shared_distillation)).resolve()}
先读取该坐标无关统计，再提出四个使用方式不同的 best-response 机制；不得重复运行蒸馏脚本。Rollman 只能预测 Ghost 行为后选择自身动作，不能复制 Ghost 动作。
"""
        return f"""# Rollman HL hypothesis planner {act_id}

proposal cycle: {iteration_id}
common parent: {parent_version_id}
active target: {active_target or "none"}

只读输入：
- compact game digest: {Path(game_digest_path).resolve()}
- authoritative context manifest: {self.bundle.manifest_path}
- explicit research state: {Path(research_state_path).resolve()}
- candidate workspace: {Path(workspace).resolve()}
- previous measurements: {measurements}
- bounded replay evidence: {evidence}
{distillation}

先读取 game digest、research state、所有 evidence summary 和当前 workspace/ai.py。只有诊断依赖精确规则语义时，才按 manifest 定点读取权威规则对应章节；不要求每轮完整重读静态长文。不得读取人类对手源码、其他 run 或其他候选版本。

Planner 压缩边界：
- 只能读取 evidence 的 summary；不得打开 replay 或 trace，不得对它们运行脚本或自行解析。
- 不得打印完整 replay、完整 trace、完整 observation、完整棋盘或全量事件流。
- diagnosis 中的 level/round/事件必须来自 summary；精确窗口由后续候选 act 使用受限工具核查。
- 对照 research state 与当前 ai.py，四个 mechanism 必须是尚未实现、未被既有失败证据否定的实质新机制。

最多 10 次工具调用；四个 summary 应批量读取。写出并校验 branch_briefs.json 后立即结束，不运行 git status/diff，不继续扩展诊断。

基于同一份证据，提出恰好 4 个机制上不同、可证伪的 Rollman 改进方向。禁止把同一机制的阈值、权重或参数变化伪装成四种方案；禁止 grid search。允许 if/else、路径规划、搜索、状态机、有限记忆和策略代码增长。

将严格 JSON 数组写入 workspace/.agentbench/branch_briefs.json。每项必须且只能包含：branch_index（0..3）、diagnosis、mechanism、expected_change、falsifier。diagnosis 必须引用具体回放 level/round/事件。不要修改候选策略代码。
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
    ) -> str:
        if int(branch_brief.get("branch_index", -1)) != branch_index:
            raise ValueError("branch brief index does not match candidate branch")
        base = self.build_prompt(
            act_id=act_id,
            branch_index=branch_index,
            branch_count=branch_count,
            parent_version_id=parent_version_id,
            workspace=workspace,
            replay_evidence=replay_evidence,
            previous_measurements=previous_measurements,
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
        brief = json.dumps(
            dict(branch_brief),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return base + f"""

本候选的唯一结构化 branch brief：{brief}
必须实现并检验这个机制；不得改做其他分支，也不得只调整无证据参数。
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
        return f"""# Rollman HL comparative reducer {act_id}

proposal cycle: {iteration_id}
selected search parent: {selected_version_id}

只读输入：
- compact game digest: {Path(game_digest_path).resolve()}
- current research state: {Path(research_state_path).resolve()}
- four-candidate factual packet: {Path(reducer_input_path).resolve()}
- selected candidate workspace: {Path(workspace).resolve()}

比较四个机制的实际比赛反馈。Framework 记录的分数、回放和测量高于模型推断；不得把没有证据的解释写成稳定知识。策略膨胀和局部 if/else 不构成失败理由。不得修改候选代码、版本指针、对手课程或认证结论。

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

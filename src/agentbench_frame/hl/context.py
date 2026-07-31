"""Hashed static context, incremental prompts, and recoverable checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


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

只读上下文：
- context manifest: {self.bundle.manifest_path}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}

本阶段没有比赛回放。不要虚构回放证据，也不要假装从反馈中得出结论。

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
6. 压缩重复规则，避免堆叠散乱 if/else；保持清晰的策略层次和回滚边界。
7. 完成框架指定的静态检查和 smoke test。
8. 不要直接修改 Experience Skill；本阶段只建立初始算法，后续再从合法比赛回放更新经验。
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

只读上下文：
- context manifest: {self.bundle.manifest_path}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}
- parent version: {parent_version_id}
{curriculum}

上一轮测量：{measurements}
必须核查的回放证据：{evidence}

科研隔离边界：
- 只允许读取上述 context manifest 及其 files、candidate workspace、Experience Skill，以及“必须核查的回放证据”明确列出的 replay/trace。
- 禁止读取或搜索其他 run、其他候选目录、Framework 源码、人类程序、对手构建目录及用户目录中的其他文件。
- 禁止列举 candidate workspace、指定 context 或指定回放目录的父目录，禁止使用 `..` 绕过边界。
- Framework 会审计工具调用路径和原始记录；越界 act 会被标记失败，不评测、不晋级。

Act 预算：
- 最多 14 次工具调用；优先批量读取，禁止用许多小命令反复查看同一材料。
- 必须先读取 evidence 中的 `summary`；不得打印完整 replay、完整 trace、完整棋盘或全量事件流。
- 只允许对最多 2 个可证伪假设做定点探针，每个命令输出不超过 6000 tokens，每条 trace 最多展开 20 个相关回合。
- 完成一次证据诊断后立即实现最小机制改动并验证；禁止在同一 act 内形成参数搜索循环。

执行约束：
1. 先阅读 context manifest 指向的规则、决策空间和 Replay Skill，再阅读 workspace 中的代码。
2. 从给定回放中引用至少一个具体 level/round/事件，提出一个可证伪的因果诊断。
3. 实现一个机制连贯、可泛化的改进；允许搜索、路径规划、状态机、记忆和其他可解释代码。
4. 禁止无依据的参数枚举或 grid search。只在回放证据直接指向决策边界时修改数值。
5. 若一轮有多个候选，本候选必须与同轮其他候选机制上不同，不能只是换阈值。
6. 压缩或整合被替代的策略，避免持续堆叠分支；保留清晰回滚边界。
7. 不读取、搜索或推断人类对手源码。只能从合法比赛回放学习。
8. 只在 candidate workspace 内完成 `python -m py_compile ai.py` 和候选侧 smoke test，不搜索 Framework 命令。不要直接修改 Experience Skill；将四个字符串数组 stable_knowledge、failed_hypotheses、replay_evidence、active_questions 写入 workspace/.agentbench/experience_update.json，由 Framework 在候选入选后合并。
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

"""Hashed static context, incremental prompts, and recoverable checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


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
        manifest_files: dict[str, dict[str, str]] = {}
        for name, source_value in sorted(sources.items()):
            source = Path(source_value)
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
    ) -> str:
        if not 0 <= branch_index < branch_count:
            raise ValueError("branch_index must be inside branch_count")
        evidence = json.dumps(
            replay_evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        measurements = json.dumps(
            previous_measurements,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"""# HL iteration {act_id} — 候选 {branch_index + 1}/{branch_count}

目标：在冻结评测协议下提升游戏 agent，保持程序可解释、可运行、可复现。

只读上下文：
- context manifest: {self.bundle.manifest_path}
- candidate workspace: {Path(workspace).resolve()}
- Experience Skill: {Path(experience_path).resolve()}
- parent version: {parent_version_id}

上一轮测量：{measurements}
必须核查的回放证据：{evidence}

执行约束：
1. 先阅读 context manifest 指向的规则、决策空间和 Replay Skill，再阅读 workspace 中的代码。
2. 从给定回放中引用至少一个具体 level/round/事件，提出一个可证伪的因果诊断。
3. 实现一个机制连贯、可泛化的改进；允许搜索、路径规划、状态机、记忆和其他可解释代码。
4. 禁止无依据的参数枚举或 grid search。只在回放证据直接指向决策边界时修改数值。
5. 若一轮有多个候选，本候选必须与同轮其他候选机制上不同，不能只是换阈值。
6. 压缩或整合被替代的策略，避免持续堆叠分支；保留清晰回滚边界。
7. 不读取、搜索或推断人类对手源码。只能从合法比赛回放学习。
8. 完成框架指定的静态检查和 smoke test，并更新 Experience Skill 的稳定经验、失败反例与未决问题。
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

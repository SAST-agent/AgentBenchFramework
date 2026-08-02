"""Miracle Loop 的 Results-compatible Run 存储与预算账本。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .loop_config import BudgetConfig, LoopConfig


class BudgetExceeded(RuntimeError):
    def __init__(self, dimension: str):
        super().__init__(f"budget exceeded: {dimension}")
        self.dimension = dimension


class BudgetLedger:
    def __init__(self, limits: BudgetConfig):
        self.limits = limits
        self.started = time.monotonic()
        self.rollouts = 0
        self.episode_reads = 0
        self.decision_reads = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_seconds = 0.0
        self.battle_seconds = 0.0

    def _ensure(self, dimension: str, value: int | float, limit: int | float) -> None:
        if value > limit:
            raise BudgetExceeded(dimension)

    def check(self) -> None:
        self._ensure("wall_seconds", time.monotonic() - self.started, self.limits.max_wall_seconds)

    def charge_rollout(self, count: int = 1) -> None:
        value = self.rollouts + count
        self._ensure("rollouts", value, self.limits.max_rollouts)
        self.rollouts = value
        self.check()

    def charge_read(self, episodes: int, decisions: int) -> None:
        ep_value = self.episode_reads + episodes
        decision_value = self.decision_reads + decisions
        self._ensure("episode_reads", ep_value, self.limits.max_episode_reads)
        self._ensure("decision_reads", decision_value, self.limits.max_decision_reads)
        self.episode_reads = ep_value
        self.decision_reads = decision_value
        self.check()

    def charge_usage(self, usage: dict) -> None:
        total = self.total_tokens + int(usage.get("total_tokens", 0) or 0)
        self._ensure("total_tokens", total, self.limits.max_total_tokens)
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens = total
        self.check()

    def charge_context(self, total_tokens: int, limit: int) -> None:
        self._ensure("context_tokens", int(total_tokens), int(limit))
        self.check()

    def charge_api_time(self, seconds: float) -> None:
        self.api_seconds += max(0.0, float(seconds))
        self.check()

    def charge_battle_time(self, seconds: float) -> None:
        self.battle_seconds += max(0.0, float(seconds))
        self.check()

    def snapshot(self) -> dict:
        return {
            "rollouts": self.rollouts,
            "episode_reads": self.episode_reads,
            "decision_reads": self.decision_reads,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_seconds": round(self.api_seconds, 6),
            "battle_seconds": round(self.battle_seconds, 6),
            "wall_seconds": round(time.monotonic() - self.started, 6),
        }


class MiracleRunStore:
    def __init__(self, run_dir: Path, run_id: str, config: LoopConfig, created: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self.config = config
        self.created = created
        self.started_at = time.time()
        self.events_path = run_dir / "events.jsonl"

    @classmethod
    def create(
        cls,
        config: LoopConfig,
        data_dir: Path | str | None = None,
        run_id: str | None = None,
    ) -> "MiracleRunStore":
        root = Path(data_dir) if data_dir is not None else Path(
            os.environ.get("AGENTBENCH_DATA", "agentbench_data")
        )
        run_id = run_id or cls._make_run_id()
        run_dir = root / "runs" / "24_miracle" / config.agent / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        store = cls(run_dir, run_id, config, created)
        store._copy_skills()
        store._write_run_toml()
        return store

    @staticmethod
    def _make_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        digest = hashlib.md5(os.urandom(8)).hexdigest()[:8]
        return f"{stamp}_{digest}"

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def _copy_skills(self) -> None:
        root = Path(__file__).resolve().parents[3]
        target = self.run_dir / "skills"
        target.mkdir()
        for name in ("miracle-harness", "miracle-replay-reader"):
            shutil.copyfile(
                root / "skills" / name / "SKILL.md",
                target / f"{name}.SKILL.md",
            )

    def _write_run_toml(self, summary: dict | None = None) -> None:
        cfg = self.config.public_dict()
        lines = [
            "[run]",
            f'run_id = "{self.run_id}"',
            'game = "24_miracle"',
            f'agent = "{self.config.agent}"',
            'type = "rule_iter"',
            f'created = "{self.created}"',
            f'git_commit = "{self._git_commit()}"',
            f"started_at = {self.started_at}",
            "",
            "[config]",
            f'opponent = "{self.config.opponent}"',
            f'model = "{self.config.llm.model}"',
            f"max_iterations = {self.config.budget.max_iterations}",
            f"max_rollouts = {self.config.budget.max_rollouts}",
        ]
        if summary is not None:
            lines[8:8] = [
                f"finished_at = {time.time()}",
                f"total_steps = {int(summary.get('total_steps', 0))}",
                f"total_episodes = {int(summary.get('total_episodes', 0))}",
            ]
        self._write_text_atomic(self.run_dir / "run.toml", "\n".join(lines) + "\n")
        self.write_json_atomic(self.run_dir / "config.json", cfg)

    def iteration_dir(self, index: int) -> Path:
        path = self.run_dir / "iterations" / f"iteration-{index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_event(self, event: str, **fields) -> None:
        row = {"event": event, "timestamp": time.time(), **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

    def write_json_atomic(self, path: Path | str, value) -> None:
        self._write_text_atomic(
            Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _write_text_atomic(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def finish(self, summary: dict) -> None:
        wall_seconds = float(summary.get("wall_seconds", time.time() - self.started_at))
        complete = {
            "run_id": self.run_id,
            "game": "24_miracle",
            "agent": self.config.agent,
            "run_type": "rule_iter",
            "created": self.created,
            "git_commit": self._git_commit(),
            "wall_hours": round(wall_seconds / 3600, 6),
            **summary,
        }
        self.write_json_atomic(self.run_dir / "summary.json", complete)
        self._write_run_toml(complete)
        self.write_event("run_finished", status=complete.get("status", "complete"))

"""
Run class — manages one experimental run with transparent observability.

A Run is the central coordination point for tracking. It owns:
- A JSONL writer for step-level events
- A ResourceSampler for system metrics
- Metadata (run.toml) and summary (summary.json)
"""

import json
import os
import time
import hashlib
from typing import Any, Dict, Optional

from agentbench_frame.tracking.records import RunMeta
from agentbench_frame.tracking.writer import JSONLWriter
from agentbench_frame.tracking.sampler import ResourceSampler
from agentbench_frame.tracking.wrappers import TrackedEnv, TimedAgent


class Run:
    """Manages one experimental run.

    Usage:
        run = Run.start(game="generals", agent="aggressive")
        env = run.wrap_env(GeneralsEnv())
        agent = run.wrap_agent(RuleBasedAgent("aggressive"))
        run.start_sampler(interval_s=5)
        # ... run experiment ...
        run.finish()
    """

    @classmethod
    def start(cls, game: str, agent: str,
              data_dir: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None) -> "Run":
        """Create and initialise a Run.

        Args:
            game: Human-readable game name (e.g. "generals")
            agent: Human-readable agent name (e.g. "ppo-v1")
            data_dir: Root data directory (default: "./runs")
            config: Arbitrary config dict written to run.toml
        """
        data_dir = data_dir or "./runs"
        run_id = cls._make_run_id(game, agent)
        run_dir = os.path.join(data_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        meta = RunMeta(
            run_id=run_id,
            game=game,
            agent=agent,
            agent_type="",
            runner="",
            started_at=time.time(),
            config=config or {},
        )

        writer = JSONLWriter(os.path.join(run_dir, "events.jsonl"))

        return cls(run_id=run_id, run_dir=run_dir, meta=meta,
                   writer=writer, config=config or {})

    @staticmethod
    def _make_run_id(game: str, agent: str) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        raw = f"{game}_{agent}_{ts}_{os.getpid()}"
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{game}_{ts}_{h}"

    def __init__(self, run_id: str, run_dir: str, meta: RunMeta,
                 writer: JSONLWriter, config: Dict[str, Any]):
        self.run_id = run_id
        self.run_dir = run_dir
        self.meta = meta
        self.writer = writer
        self.config = config
        self._sampler: Optional[ResourceSampler] = None
        self._tracked_env: Optional[TrackedEnv] = None
        self._timed_agent: Optional[TimedAgent] = None

        # Episode-level aggregation
        self._episodes: int = 0
        self._total_steps: int = 0
        self._total_reward: float = 0.0
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._episode_winners: list = []

    def wrap_env(self, env) -> TrackedEnv:
        """Wrap an environment for step-level tracking."""
        self._tracked_env = TrackedEnv(env, on_step=self.write)
        return self._tracked_env

    def wrap_agent(self, agent) -> TimedAgent:
        """Wrap an agent for decision timing."""
        self._timed_agent = TimedAgent(agent, on_act=self.write)
        return self._timed_agent

    def start_sampler(self, interval_s: float = 5.0):
        """Start background resource sampling."""
        self._sampler = ResourceSampler(
            callback=lambda record: self.write(**record),
            interval_s=interval_s,
        )
        self._sampler.start()

    def write(self, event_type: Optional[str] = None, **kwargs):
        """Write an arbitrary event to the events log."""
        if event_type:
            kwargs["event"] = event_type
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = time.time()
        self.writer.write(kwargs)

    def log_episode(self, reward: float, steps: int, winner: int,
                    info: Optional[Dict[str, Any]] = None):
        """Record episode-level summary."""
        self._episodes += 1
        self._total_steps += steps
        self._total_reward += reward
        self._episode_rewards.append(reward)
        self._episode_lengths.append(steps)
        self._episode_winners.append(winner)

        self.write("episode",
                   episode=self._episodes,
                   total_steps=steps,
                   total_reward=reward,
                   winner=winner,
                   wall_time_s=0.0,
                   info=info or {})

    def finish(self) -> Dict[str, Any]:
        """Finish the run: flush writer, stop sampler, write summary and metadata."""
        if self._sampler:
            self._sampler.stop()

        self.writer.flush()

        self.meta.finished_at = time.time()
        self.meta.total_steps = self._total_steps
        self.meta.total_episodes = self._episodes

        # summary.json
        summary = self._build_summary()
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # run.toml
        tom_path = os.path.join(self.run_dir, "run.toml")
        self._write_toml(tom_path)

        self.writer.close()
        return summary

    def _build_summary(self) -> Dict[str, Any]:
        n = max(1, self._episodes)
        wins = sum(1 for w in self._episode_winners if w == 0)
        return {
            "run_id": self.run_id,
            "game": self.meta.game,
            "agent": self.meta.agent,
            "started_at": self.meta.started_at,
            "finished_at": self.meta.finished_at,
            "duration_s": (self.meta.finished_at or time.time()) - self.meta.started_at,
            "total_episodes": self._episodes,
            "total_steps": self._total_steps,
            "total_reward": self._total_reward,
            "avg_reward_per_episode": self._total_reward / n,
            "avg_steps_per_episode": self._total_steps / n,
            "win_rate": wins / n,
            "episode_rewards": self._episode_rewards,
            "config": self.config,
        }

    def _write_toml(self, path: str):
        """Write minimal TOML metadata."""
        lines = []
        lines.append("[run]")
        lines.append(f'run_id = "{self.run_id}"')
        lines.append(f'game = "{self.meta.game}"')
        lines.append(f'agent = "{self.meta.agent}"')
        lines.append(f'agent_type = "{self.meta.agent_type}"')
        lines.append(f'runner = "{self.meta.runner}"')
        lines.append(f"started_at = {self.meta.started_at}")
        if self.meta.finished_at:
            lines.append(f"finished_at = {self.meta.finished_at}")
        lines.append(f"total_steps = {self.meta.total_steps}")
        lines.append(f"total_episodes = {self.meta.total_episodes}")
        if self.config:
            lines.append("")
            lines.append("[config]")
            for k, v in self.config.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, (int, float, bool)):
                    lines.append(f"{k} = {v}".lower() if isinstance(v, bool) else f"{k} = {v}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

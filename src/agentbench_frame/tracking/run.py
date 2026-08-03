"""
Run class — manages one experimental run with transparent observability.

Data contract for CI visualization:
    runs/{game}/{agent}/{run_id}/
    ├── run.toml         # type, created, git_commit, config
    ├── summary.json     # best_elo, final_elo, wall_hours, total_steps, win_rate, ...
    └── events.jsonl     # step-level records (optional, for debugging)
"""

import json, os, time, hashlib, subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentbench_frame.tracking.records import RunMeta
from agentbench_frame.tracking.writer import JSONLWriter
from agentbench_frame.tracking.sampler import ResourceSampler
from agentbench_frame.tracking.wrappers import TrackedEnv, TimedAgent


def _data_root() -> str:
    """Resolve data root: $AGENTBENCH_DATA or ./agentbench_data"""
    return os.environ.get("AGENTBENCH_DATA", os.path.join(os.getcwd(), "agentbench_data"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


class Run:
    """Manages one experimental run. Writes CI-compatible run.toml + summary.json.

    Usage:
        run = Run.start(game="28_generals", agent="ppo_v3", run_type="rl")
        env = run.wrap_env(GeneralsEnv())
        agent = run.wrap_agent(RuleBasedAgent("ppo_v3"))
        run.start_sampler()
        # ... training loop ...
        run.log_elo(1520)       # optional: record best Elo
        run.log_h2h({"archer": {"rider": 0.65}})  # optional: H2H
        run.finish()
    """

    @classmethod
    def start(cls, game: str, agent: str,
              run_type: str = "eval",
              data_dir: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None) -> "Run":
        """Create a Run.

        Args:
            game: Game identifier, e.g. "28_generals", "25_lostspace"
            agent: Agent name, e.g. "ppo_v3", "rule_expansionist"
            run_type: "rl" | "rule_iter" | "eval"
            data_dir: Override data root (default: $AGENTBENCH_DATA or ./agentbench_data)
            config: Arbitrary config dict written to run.toml [config] section
        """
        data_dir = data_dir or _data_root()
        run_id = cls._make_run_id()
        # CI-expected structure: runs/{game}/{agent}/{run_id}/
        run_path = os.path.join(data_dir, "runs", game, agent, run_id)
        os.makedirs(run_path, exist_ok=True)

        meta = RunMeta(
            run_id=run_id, game=game, agent=agent,
            run_type=run_type,
            created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            git_commit=_git_commit(),
            started_at=time.time(),
            config=config or {},
        )

        writer = JSONLWriter(os.path.join(run_path, "events.jsonl"))
        return cls(run_id=run_id, run_dir=run_path, meta=meta, writer=writer, config=config or {})

    @staticmethod
    def _make_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        h = hashlib.md5(os.urandom(8)).hexdigest()[:8]
        return f"{ts}_{h}"

    def __init__(self, run_id: str, run_dir: str, meta: RunMeta,
                 writer: JSONLWriter, config: Dict[str, Any]):
        self.run_id = run_id; self.run_dir = run_dir
        self.meta = meta; self.writer = writer; self.config = config
        self._sampler: Optional[ResourceSampler] = None
        self._tracked_env: Optional[TrackedEnv] = None
        self._timed_agent: Optional[TimedAgent] = None
        self._episodes = 0; self._total_steps = 0; self._total_reward = 0.0
        self._episode_rewards: list = []; self._episode_lengths: list = []
        self._episode_winners: list = []
        # CI-report fields
        self._best_elo: Optional[float] = None
        self._final_elo: Optional[float] = None
        self._elo_history: List[Dict] = []
        self._h2h: Dict[str, Dict[str, float]] = {}
        self._resource_samples: List[Dict] = []

    # ---- wrappers ----

    def wrap_env(self, env) -> TrackedEnv:
        self._tracked_env = TrackedEnv(env, on_step=self.write)
        return self._tracked_env

    def wrap_agent(self, agent) -> TimedAgent:
        self._timed_agent = TimedAgent(agent, on_act=self.write)
        return self._timed_agent

    def start_sampler(self, interval_s: float = 5.0):
        self._sampler = ResourceSampler(
            callback=lambda r: (self._resource_samples.append(r), self.write(**r)),
            interval_s=interval_s)
        self._sampler.start()

    # ---- logging ----

    def write(self, event_type: Optional[str] = None, **kwargs):
        if event_type: kwargs["event"] = event_type
        kwargs.setdefault("timestamp", time.time())
        self.writer.write(kwargs)

    def log_episode(self, reward: float, steps: int, winner: int,
                    info: Optional[Dict[str, Any]] = None):
        self._episodes += 1; self._total_steps += steps; self._total_reward += reward
        self._episode_rewards.append(reward); self._episode_lengths.append(steps)
        self._episode_winners.append(winner)
        self.write("episode", episode=self._episodes, total_steps=steps,
                   total_reward=reward, winner=winner, info=info or {})

    def log_elo(self, elo: float, step: Optional[int] = None):
        """Record current Elo rating. Call periodically during training."""
        self._elo_history.append({"step": step or self._total_steps, "elo": elo})
        if self._best_elo is None or elo > self._best_elo:
            self._best_elo = elo
        self._final_elo = elo

    def log_h2h(self, h2h: Dict[str, Dict[str, float]]):
        """Record head-to-head win rates. Merges with previous calls."""
        for a, opponents in h2h.items():
            if a not in self._h2h: self._h2h[a] = {}
            self._h2h[a].update(opponents)

    # ---- finish ----

    def finish(self) -> Dict[str, Any]:
        if self._sampler: self._sampler.stop()
        self.writer.flush()
        self.meta.finished_at = time.time()
        summary = self._build_summary()
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._write_toml(os.path.join(self.run_dir, "run.toml"))
        self.writer.close()
        return summary

    def _build_summary(self) -> Dict[str, Any]:
        n = max(1, self._episodes)
        wins = sum(1 for w in self._episode_winners if w == 0)
        wall_s = (self.meta.finished_at or time.time()) - self.meta.started_at
        return {
            "run_id": self.run_id, "game": self.meta.game, "agent": self.meta.agent,
            "run_type": self.meta.run_type,
            "created": self.meta.created,
            "git_commit": self.meta.git_commit,
            "started_at": self.meta.started_at,
            "finished_at": self.meta.finished_at,
            "wall_hours": round(wall_s / 3600, 2),
            "wall_seconds": round(wall_s, 2),
            "duration_s": round(wall_s, 2),
            "total_episodes": self._episodes,
            "total_steps": self._total_steps,
            "total_reward": self._total_reward,
            "avg_reward_per_episode": round(self._total_reward / n, 4),
            "win_rate": wins / n,
            "best_elo": self._best_elo,
            "final_elo": self._final_elo,
            "elo_history": self._elo_history,
            "h2h": self._h2h,
            "resource_summary": self._resource_summary(),
            "config": self.config,
        }

    def _resource_summary(self) -> Dict:
        if not self._resource_samples: return {}
        rss = [s.get("rss_mb", 0) for s in self._resource_samples if s.get("rss_mb")]
        cpu = [s.get("cpu_pct", 0) for s in self._resource_samples if s.get("cpu_pct")]
        return {
            "avg_rss_mb": round(sum(rss) / len(rss), 1) if rss else None,
            "max_rss_mb": max(rss) if rss else None,
            "avg_cpu_pct": round(sum(cpu) / len(cpu), 1) if cpu else None,
        }

    def _write_toml(self, path: str):
        lines = ["[run]",
                 f'run_id = "{self.run_id}"',
                 f'game = "{self.meta.game}"',
                 f'agent = "{self.meta.agent}"',
                 f'type = "{self.meta.run_type}"',
                 f'created = "{self.meta.created}"',
                 f'git_commit = "{self.meta.git_commit}"',
                 f"started_at = {self.meta.started_at}"]
        if self.meta.finished_at:
            lines.append(f"finished_at = {self.meta.finished_at}")
        lines.append(f"total_steps = {self._total_steps}")
        lines.append(f"total_episodes = {self._episodes}")
        if self.config:
            lines.append(""); lines.append("[config]")
            for k, v in self.config.items():
                if isinstance(v, str): lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool): lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, (int, float)): lines.append(f"{k} = {v}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

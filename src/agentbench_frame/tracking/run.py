"""
Run class — manages one experimental run with transparent observability.

Data contract for CI visualization:
    runs/{game}/{agent}/{run_id}/
    ├── run.toml         # type, created, git_commit, config
    ├── summary.json     # best_elo, final_elo, wall_hours, total_steps, win_rate, ...
    └── events.jsonl     # step-level records (optional, for debugging)
"""

import json, os, time, hashlib, subprocess, math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agentbench_frame.tracking.records import RunMeta
from agentbench_frame.tracking.writer import JSONLWriter
from agentbench_frame.tracking.sampler import ResourceSampler
from agentbench_frame.tracking.wrappers import TrackedEnv, TimedAgent
from agentbench_frame.tracking.budget import BudgetLedger
from agentbench_frame.tracking.iteration import ActRecord, VersionedActRecorder
from agentbench_frame.eval.information_gain import occupancy_shift as derive_occupancy_shift
from agentbench_frame.tracking.quality import inspect_event_file


SCHEMA_VERSION = "1.0"


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
        self._episode_agent_players: list = []
        self._episode_valid: list = []
        self._budget = BudgetLedger()
        self._act_recorder = VersionedActRecorder(emit=self.write)
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
        """Write one forward-compatible event.

        ``TrackedEnv`` historically passed dataclass records positionally, so
        this method accepts strings, dictionaries, and dataclass instances.
        The old ``event`` key remains as a compatibility alias for
        ``event_type``.
        """
        if event_type is not None and not isinstance(event_type, str):
            if is_dataclass(event_type):
                record = asdict(event_type)
            elif isinstance(event_type, dict):
                record = dict(event_type)
            else:
                record = {"raw": str(event_type)}
            event_type = record.get("event_type") or record.get("event")
            record.update(kwargs)
            kwargs = record

        event_name = event_type or kwargs.get("event_type") or kwargs.get("event") or "log"
        kwargs["schema_version"] = kwargs.get("schema_version", SCHEMA_VERSION)
        kwargs["event_id"] = kwargs.get("event_id", f"evt_{uuid4().hex}")
        kwargs["event_type"] = event_name
        kwargs["event"] = event_name
        kwargs["run_id"] = kwargs.get("run_id", self.run_id)
        kwargs["created_at"] = kwargs.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        kwargs.setdefault("timestamp", time.time())
        self.writer.write(kwargs)

    def log_episode(self, reward: float, steps: int, winner: int,
                    info: Optional[Dict[str, Any]] = None,
                    agent_player_id: int = 0, valid: bool = True):
        self._episodes += 1; self._total_steps += steps; self._total_reward += reward
        self._episode_rewards.append(reward); self._episode_lengths.append(steps)
        self._episode_winners.append(winner)
        self._episode_agent_players.append(agent_player_id)
        self._episode_valid.append(valid)
        phase = self.config.get(
            "budget_phase", "evaluation" if self.meta.run_type == "eval" else "learning"
        )
        self._budget.add(phase, episodes=1, env_steps=steps)
        self.write("episode", episode=self._episodes, total_steps=steps,
                   total_reward=reward, winner=winner,
                   agent_player_id=agent_player_id, valid=valid,
                   info=info or {})

    def log_budget(self, phase: str, **kwargs) -> Dict[str, Optional[float]]:
        """Add a learning/evaluation budget observation and persist it."""
        self._budget.add(phase, **kwargs)
        snapshot = self._budget.snapshot()
        self.write("budget", phase=phase, observation=kwargs, **snapshot)
        return snapshot

    def budget_snapshot(self) -> Dict[str, Optional[float]]:
        return self._budget.snapshot()

    def begin_act(self, provider: str, version_before: Optional[str] = None,
                  act_id: Optional[str] = None) -> ActRecord:
        return self._act_recorder.begin_act(provider, version_before, act_id)

    def finish_act(self, act_id: str, status: str, **kwargs) -> ActRecord:
        return self._act_recorder.finish_act(act_id, status, **kwargs)

    def record_act_evaluation(self, act_id: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Record an evaluation and attach the current cumulative budget axes.

        The coordinates are additive convenience fields derived from the same
        ledger snapshot; raw act/evaluation data remains in the event stream.
        """
        payload = dict(evaluation)
        phase = self.config.get(
            "budget_phase", "evaluation" if self.meta.run_type == "eval" else "learning"
        )
        snapshot = self._budget.snapshot()
        prefix = f"{phase}_"
        payload.setdefault("coding_agent_act", snapshot.get(prefix + "coding_agent_acts"))
        payload.setdefault("episode", snapshot.get(prefix + "episodes"))
        payload.setdefault("env_step", snapshot.get(prefix + "env_steps"))
        payload.setdefault("token", snapshot.get(prefix + "total_tokens"))
        payload.setdefault("time_s", snapshot.get(prefix + "time_s"))
        return self._act_recorder.record_evaluation(act_id, payload)

    def create_coding_agent_controller(self, provider, snapshotter=None,
                                       budget_phase: str = "learning"):
        """Create a provider-neutral act controller bound to this run."""
        from agentbench_frame.tracking.controller import CodingAgentController
        return CodingAgentController(
            provider,
            recorder=self._act_recorder,
            snapshotter=snapshotter,
            budget=self._budget,
            budget_phase=budget_phase,
            raw_output_dir=os.path.join(self.run_dir, "provider_output"),
        )

    def log_policy_kl_trace(
        self,
        episode: int,
        version_before: str,
        version_after: str,
        trace: List[float],
        context_refs: Optional[List[str]] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Persist the raw local-policy KL trace for one target-agent episode."""
        values = [float(value) for value in trace]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("policy KL trace values must be finite and non-negative")
        if context_refs is not None and len(context_refs) != len(values):
            raise ValueError("context_refs must align one-to-one with the KL trace")
        self.write(
            "policy_kl_trace",
            episode=episode,
            version_before=version_before,
            version_after=version_after,
            trace=values,
            decision_steps=len(values),
            context_refs=context_refs,
            epsilon=epsilon,
        )

    def log_occupancy(
        self,
        episode: int,
        version_before: str,
        version_after: str,
        state_ids: List[str],
        context: Optional[Dict[str, Any]] = None,
        reference_state_ids: Optional[List[str]] = None,
        smoothing: float = 1e-12,
    ) -> None:
        """Persist raw samples and, when supplied, a separate occupancy shift.

        ``state_ids`` and ``reference_state_ids`` are intentionally retained;
        the derived KL is not combined with policy KL or called information
        gain.  The reference is the measurement-domain sample chosen by the
        caller for this episode/version comparison.
        """
        if not state_ids:
            raise ValueError("occupancy state_ids cannot be empty")
        shift = None
        if reference_state_ids is not None:
            if not reference_state_ids:
                raise ValueError("reference_state_ids cannot be empty")
            shift = derive_occupancy_shift(state_ids, reference_state_ids, smoothing=smoothing)
        self.write(
            "occupancy",
            episode=episode,
            version_before=version_before,
            version_after=version_after,
            state_ids=list(state_ids),
            state_count=len(state_ids),
            reference_state_ids=(list(reference_state_ids)
                                 if reference_state_ids is not None else None),
            reference_state_count=(len(reference_state_ids)
                                   if reference_state_ids is not None else None),
            occupancy_shift=shift,
            smoothing=smoothing if reference_state_ids is not None else None,
            context=context or {},
        )

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

    def finish(self, extra_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stop tracking and persist the final run summary.

        ``extra_summary`` lets runners add workflow-specific metrics before
        the JSON file is written, so the returned and persisted summaries stay
        consistent.
        """
        if self._sampler: self._sampler.stop()
        self.writer.flush()
        self.meta.finished_at = time.time()
        summary = self._build_summary()
        summary["budget"] = self._budget.snapshot()
        if extra_summary:
            summary.update(extra_summary)
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._write_toml(os.path.join(self.run_dir, "run.toml"))
        self.writer.close()
        return summary

    def _build_summary(self) -> Dict[str, Any]:
        n = max(1, self._episodes)
        valid = [
            (winner, player)
            for winner, player, is_valid in zip(
                self._episode_winners, self._episode_agent_players, self._episode_valid
            )
            if is_valid
        ]
        wins = sum(1 for winner, player in valid if winner == player)
        draws = sum(1 for winner, _player in valid if winner == -1)
        losses = len(valid) - wins - draws
        benchmark_score = (
            (wins + 0.5 * draws) / len(valid) if valid else None
        )
        wall_s = (self.meta.finished_at or time.time()) - self.meta.started_at
        return {
            "run_id": self.run_id, "game": self.meta.game, "agent": self.meta.agent,
            "run_type": self.meta.run_type,
            "created": self.meta.created,
            "git_commit": self.meta.git_commit,
            "started_at": self.meta.started_at,
            "finished_at": self.meta.finished_at,
            "wall_hours": round(wall_s / 3600, 2),
            "total_episodes": self._episodes,
            "total_steps": self._total_steps,
            "total_reward": self._total_reward,
            "win_rate": wins / n,
            "benchmark_score": benchmark_score,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "best_elo": self._best_elo,
            "final_elo": self._final_elo,
            "elo_history": self._elo_history,
            "h2h": self._h2h,
            "resource_summary": self._resource_summary(),
            "event_quality": inspect_event_file(
                os.path.join(self.run_dir, "events.jsonl")
            ).to_dict(),
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

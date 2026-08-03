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
from agentbench_frame.eval.information_gain import (
    FORMAL_POLICY_KL_DIRECTION,
    FORMAL_POLICY_KL_LOG_BASE,
    FORMAL_POLICY_KL_SUM_UNIT,
    LEGACY_POLICY_KL_TRACE_PROFILE,
    occupancy_shift as derive_occupancy_shift,
)
from agentbench_frame.eval.trajectory_kl import trajectory_kl_result_from_payload
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
              config: Optional[Dict[str, Any]] = None,
              run_id: Optional[str] = None,
              append: bool = True,
              created: Optional[str] = None,
              git_commit: Optional[str] = None,
              started_at: Optional[float] = None) -> "Run":
        """Create a Run.

        Args:
            game: Game identifier, e.g. "28_generals", "25_lostspace"
            agent: Agent name, e.g. "ppo_v3", "rule_expansionist"
            run_type: "rl" | "rule_iter" | "eval"
            data_dir: Override data root (default: $AGENTBENCH_DATA or ./agentbench_data)
            config: Arbitrary config dict written to run.toml [config] section
            run_id: Optional caller-supplied run_id (review #1 backcompat). When
                provided (e.g. ``20260722-001929_52bd14`` for matrix resume or
                offline migration), the run directory reuses that id verbatim
                — no second run directory is ever created. When omitted, the
                framework auto-generates one via :meth:`_make_run_id`.
            append: If True (default), the events.jsonl writer opens in
                append mode so multiple ``Run`` objects on the same run_id
                preserve history. If False, the events.jsonl is truncated at
                start — used by idempotent rebuild paths (review #1 matrix
                ``write_run_compatible_output``) so re-invocations do not
                create duplicate event lines.
            created: Optional ISO-8601 timestamp string (e.g.
                ``2026-07-21T16:19:30Z``) for offline migration. When
                provided, the run's ``created`` metadata (and the persisted
                run.toml + summary.json ``created`` field) reflect this
                authoritative evidence timestamp instead of the framework's
                "now". When omitted, the current wall clock is used
                (default behaviour). Must be a valid ISO-8601 string.
            git_commit: Optional git commit short-SHA, overriding the
                framework's auto-detected ``git rev-parse --short HEAD``.
                Used in offline migration where the migrated run should carry
                the original commit identity.
        """
        data_dir = data_dir or _data_root()
        if run_id is None or not isinstance(run_id, str) or not run_id:
            run_id = cls._make_run_id()
        if created is None or not isinstance(created, str) or not created:
            created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if git_commit is None or not isinstance(git_commit, str):
            git_commit = _git_commit()
        # CI-expected structure: runs/{game}/{agent}/{run_id}/
        run_path = os.path.join(data_dir, "runs", game, agent, run_id)
        os.makedirs(run_path, exist_ok=True)

        meta = RunMeta(
            run_id=run_id, game=game, agent=agent,
            run_type=run_type,
            created=created,
            git_commit=git_commit,
            started_at=(started_at if started_at is not None else time.time()),
            config=config or {},
        )

        writer = JSONLWriter(os.path.join(run_path, "events.jsonl"), append=append)
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
        if any(type(value) not in {int, float} for value in trace):
            raise TypeError("policy KL trace values must be numbers, not bool")
        values = [float(value) for value in trace]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("policy KL trace values must be finite and non-negative")
        if context_refs is not None and len(context_refs) != len(values):
            raise ValueError("context_refs must align one-to-one with the KL trace")
        if epsilon is not None:
            if type(epsilon) not in {int, float}:
                raise TypeError("epsilon must be an int or float, not bool")
            epsilon = float(epsilon)
            if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
                raise ValueError("epsilon must be in [0, 1]")
        errors = []
        try:
            trajectory_kl_episode = math.fsum(values) if values else None
        except OverflowError:
            trajectory_kl_episode = None
            errors.append("trajectory KL sum is not finite")
        if (
            trajectory_kl_episode is not None
            and not math.isfinite(trajectory_kl_episode)
        ):
            trajectory_kl_episode = None
            errors.append("trajectory KL sum is not finite")
        complete = bool(values) and trajectory_kl_episode is not None
        estimand = (
            "legacy_unspecified"
            if epsilon is None
            else "epsilon_regularized_local_kl_sum_under_unspecified_occupancy"
        )
        self.write(
            "policy_kl_trace",
            episode=episode,
            version_before=version_before,
            version_after=version_after,
            trace=values,
            decision_steps=len(values),
            context_refs=context_refs,
            epsilon=epsilon,
            measurement_profile=LEGACY_POLICY_KL_TRACE_PROFILE,
            measurement_status="complete" if complete else "incomplete",
            trajectory_kl_episode=trajectory_kl_episode,
            mean_local_policy_kl=(
                trajectory_kl_episode / len(values)
                if trajectory_kl_episode is not None
                else None
            ),
            information_gain=None,
            information_gain_status="unverified",
            local_policy_kl_sum=trajectory_kl_episode,
            aggregation=None,
            information_gain_unit=None,
            local_policy_kl_sum_unit=FORMAL_POLICY_KL_SUM_UNIT,
            direction=FORMAL_POLICY_KL_DIRECTION,
            log_base=FORMAL_POLICY_KL_LOG_BASE,
            rollout_source="unspecified",
            estimand=estimand,
            errors=errors,
        )

    def log_trajectory_kl_result(self, result: Any) -> None:
        """Persist a rich trajectory-KL result under the legacy event type."""

        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            raise TypeError("trajectory KL result must be a mapping or expose to_dict()")

        verified = trajectory_kl_result_from_payload(payload)
        payload = verified.to_dict()
        payload.pop("status", None)
        payload.setdefault(
            "context_refs",
            [decision.context_ref for decision in verified.decisions],
        )
        self.write("policy_kl_trace", **payload)

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

    def recompute_totals_from_events(self, *,
                                     game_event_type: str = "game") -> Dict[str, Any]:
        """Recompute ``_episodes`` / ``_total_steps`` / win-loss-draw counts
        from the live ``game`` events written to disk by callers that bypass
        :meth:`log_episode` (e.g. matrix re-emit, offline migration). Returns
        the recalculated counters as a dict.

        Why this is needed (review #1 §5): matrix records ARE per-game events
        emitted via ``run.write("game", ...)``; framework counters
        (``self._episodes`` / ``self._total_steps``) are only incremented by
        :meth:`log_episode`. Without this recomputation, ``_write_toml`` /
        ``_build_summary`` would report zero totals even though the events
        file on disk has the real per-game records. We refuse to fabricate
        episode events; matrix events ARE the events.

        Semantics:

          - reads only events of ``event_type == game_event_type`` (default
            ``"game"``) — game records that carry ``valid``,
            ``normalized_result`` (win/loss/draw/error), ``steps``;
          - increments ``_episodes`` only for valid game records (matches
            :meth:`log_episode`'s ``valid=True`` semantics + matches the
            Miracle aggregate's ``valid_games`` count);
          - increments ``_total_steps`` by per-game steps for valid games;
          - DOES NOT mutate ``_episode_rewards`` / ``_episode_winners`` /
            ``_episode_agent_players`` — these bookkeeping lists are only
            populated by :meth:`log_episode`, and matrix migration has no
            meaningful reward / per-episode winner representation. summary
            ``win_rate`` / ``wins`` / ``losses`` / ``draws`` are still set
            by ``extra_summary`` passed to :meth:`finish` for matrix paths.

        Returns a dict with the recomputed counters so callers (matrix
        migration) can sanity-check totals are consistent with the hard
        anchors BEFORE the run is committed/promoted.
        """
        # Flush buffered writes before reading.
        self.writer.flush()
        path = os.path.join(self.run_dir, "events.jsonl")
        recomputed = dict(episodes=0, total_steps=0, wins=0, losses=0,
                          draws=0, valid_games=0)
        if not os.path.exists(path):
            return recomputed
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(rec, dict):
                    continue
                if rec.get("event_type", rec.get("event")) != game_event_type:
                    continue
                valid = bool(rec.get("valid"))
                if not valid:
                    continue
                recomputed["valid_games"] += 1
                recomputed["episodes"] += 1
                recomputed["total_steps"] += int(rec.get("steps") or 0)
                nr = rec.get("normalized_result")
                if nr == "win":
                    recomputed["wins"] += 1
                elif nr == "loss":
                    recomputed["losses"] += 1
                elif nr == "draw":
                    recomputed["draws"] += 1
        # Mutate framework counters in place so _build_summary / _write_toml
        # see the recomputed totals.
        self._episodes = recomputed["episodes"]
        self._total_steps = recomputed["total_steps"]
        return recomputed

    def finish(self, extra_summary: Optional[Dict[str, Any]] = None,
               finished_at: Optional[float] = None) -> Dict[str, Any]:
        """Stop tracking and persist the final run summary.

        ``extra_summary`` lets runners add workflow-specific metrics before
        the JSON file is written, so the returned and persisted summaries stay
        consistent.

        ``finished_at`` lets offline migration preserve the original execution
        wall-clock time instead of stamping the migration's own time.
        """
        if self._sampler: self._sampler.stop()
        self.writer.flush()
        self.meta.finished_at = (finished_at if finished_at is not None else time.time())
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
                 f'run_id = "{self._toml_escape(self.run_id)}"',
                 f'game = "{self._toml_escape(self.meta.game)}"',
                 f'agent = "{self._toml_escape(self.meta.agent)}"',
                 f'type = "{self._toml_escape(self.meta.run_type)}"',
                 f'created = "{self._toml_escape(self.meta.created)}"',
                 f'git_commit = "{self._toml_escape(self.meta.git_commit)}"',
                 f"started_at = {self.meta.started_at}"]
        if self.meta.finished_at:
            lines.append(f"finished_at = {self.meta.finished_at}")
        lines.append(f"total_steps = {self._total_steps}")
        lines.append(f"total_episodes = {self._episodes}")
        if self.config:
            lines.append(""); lines.append("[config]")
            for k, v in self.config.items():
                if isinstance(v, str): lines.append(f'{k} = "{self._toml_escape(v)}"')
                elif isinstance(v, bool): lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, (int, float)): lines.append(f"{k} = {v}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _toml_escape(s: str) -> str:
        """Escape a string for TOML double-quoted value: backslash first, then quote."""
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

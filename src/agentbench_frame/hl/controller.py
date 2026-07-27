"""
HLIterationController — the Bench Controller for Heuristic Learning.

Owns one coding-agent act lifecycle (doc §12.1, §13, §15):

    agent_act  →  runner.run  →  snapshot(version_after)  →  classify edit
        →  eval on frozen BenchmarkSpec
        →  probe BOTH versions over the frozen ReferenceStateSet ν   (Q2)
        →  local_policy_kl_trace + occupancy_shift
        →  emit version / eval / policy_kl / occupancy_shift / budget

Design choices locked during planning:
- Re-run both versions over ν (not reuse-old-trace): the controller probes
  v_{k-1} and v_k over the same frozen reference set every act, so policy KL
  is a pure measurement with no occupancy-drift confound (doc §9.3).
- Coding-agent invocation is pluggable (CodingAgentRunner): FakeRunner for
  tests, ClaudeCodeRunner for real HL iteration.
- Missing scores stay missing (doc §12): an incomplete eval records
  evaluation_status='incomplete' and win_rate=None, never 0 and never a loss.
- Budget split: learning/evaluation/total; unknown tokens/time → None (doc §13).
- Append-only events via HLEventWriter (doc §14).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.distribution import (
    enumerate_legal_actions,
    epsilon_smoothed_distribution,
    policy_kl,
    local_policy_kl_trace,
    LegalActionSet,
)
from agentbench_frame.hl.events import HLEventWriter
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.runner import CodingAgentRunner, AgentRunResult


ProbeFactory = Callable[..., ReferenceProbe]
EvaluatorFactory = Callable[..., Any]


class HLIterationController:
    def __init__(
        self,
        *,
        codebase: HLCodebase,
        runner,  # CodingAgentRunner
        spec: BenchmarkSpec,
        reference: ReferenceStateSet,
        run_id: str,
        events_path,
        epsilon: float = 0.1,
        evaluator_factory: Optional[EvaluatorFactory] = None,
        probe_factory: Optional[ProbeFactory] = None,
        stage_root: Optional[Path] = None,
    ):
        self.codebase = codebase
        self.runner = runner
        self.spec = spec
        self.reference = reference
        self.run_id = run_id
        self.epsilon = epsilon
        self._evaluator_factory = evaluator_factory
        self._probe_factory = probe_factory or ReferenceProbe
        self._stage_root = Path(stage_root) if stage_root else codebase.root.parent / "stage"
        self._events = HLEventWriter(run_id=run_id, path=events_path)
        self._act_counter = 0
        self._budget = {
            "learning_coding_agent_acts": 0,
            "evaluation_coding_agent_acts": 0,
            "total_coding_agent_acts": 0,
        }

    def act(self, *, version_before: Optional[VersionHandle]) -> VersionHandle:
        """Run one coding-agent act. Returns the version_after handle.

        Even if the workspace is unreadable after the run, an act event is
        written; version_after is None in that case (doc §12.1).
        """
        self._act_counter += 1
        act_id = f"{self.run_id}-act{self._act_counter:04d}"
        t0 = time.monotonic()

        # 1. emit agent_act
        self._events.write(
            "agent_act", act_id=act_id,
            version_before=version_before.version_id if version_before else None,
            edit_policy_mode="free",
            budget_before=self._budget,
        )

        # 2. run the coding agent against the workspace
        context = {"act_id": act_id, "spec_id": self.spec.spec_id}
        run_result = self.runner.run(workspace=self.codebase.root, context=context)
        runner_time = run_result.time_s

        # 3. snapshot version_after (crash-safe; None if unreadable)
        try:
            version_after = self.codebase.snapshot(
                parent_version_id=version_before.version_id if version_before else None,
                edit_type=run_result.edit_type,
            )
        except Exception:
            version_after = None

        # 4. classify edit_type is already on the handle; emit version event
        if version_after is not None:
            self._events.write(
                "version",
                version_id=version_after.version_id,
                content_hash=version_after.content_hash,
                parent_version_id=version_after.parent_version_id,
                edit_type=version_after.edit_type,
                files_touched=run_result.files_touched,
            )

        # 5. eval on the frozen BenchmarkSpec (only if we have a runnable version)
        eval_status = "incomplete"
        win_rate = None
        if version_after is not None and self._evaluator_factory is not None:
            try:
                ev_result = self._run_eval(version_after)
                summary = getattr(ev_result, "summary", None) or {}
                # honor the evaluator's own completeness verdict if present
                eval_status = summary.get("evaluation_status") or "complete"
                win_rate = summary.get("win_rate")
            except Exception:
                eval_status = "incomplete"
                win_rate = None
        self._events.write(
            "eval", act_id=act_id,
            spec_id=self.spec.spec_id,
            version_after=version_after.version_id if version_after else None,
            evaluation_status=eval_status,
            win_rate=win_rate,  # None stays None
        )

        # 6. probe BOTH versions over ν (Q2) → policy_kl + occupancy_shift
        kl_trace: List[float] = []
        occupancy_shift: Optional[float] = None
        if version_before is not None and version_after is not None:
            kl_trace, occupancy_shift = self._measure_policy_kl(
                version_before, version_after,
            )
            self._events.write(
                "policy_kl", act_id=act_id,
                version_before=version_before.version_id,
                version_after=version_after.version_id,
                local_policy_kl_trace=kl_trace,
                epsilon=self.epsilon,
            )
            self._events.write(
                "occupancy_shift", act_id=act_id,
                version_before=version_before.version_id,
                version_after=version_after.version_id,
                shift=occupancy_shift,
            )

        # 7. budget (learning scope; unknown tokens → None)
        self._budget["learning_coding_agent_acts"] += 1
        self._budget["total_coding_agent_acts"] += 1
        self._events.write(
            "budget", scope="learning", act_id=act_id,
            coding_agent_acts=self._budget["learning_coding_agent_acts"],
            prompt_tokens=run_result.prompt_tokens,      # None for fake
            completion_tokens=run_result.completion_tokens,
            total_tokens=run_result.total_tokens,
            runner_time_s=runner_time,
            act_time_s=time.monotonic() - t0,
        )
        self._events.flush()
        return version_after

    # ---- internals ----

    def _run_eval(self, version: VersionHandle):
        from agentbench_frame.hl.adapter import candidate_command
        cmd, cwd = candidate_command(version, store=self.codebase.store,
                                      dest=self._stage_root / version.version_id)
        ev = self._evaluator_factory(
            logic_command=_logic_command_stub(),
            candidate_name=f"hl-{version.version_id}",
            candidate_command=_cmd_to_str(cmd, cwd),
            opponents=[_opponent(o) for o in self.spec.opponents],
            filler_command=_cmd_to_str(cmd, cwd),
            pairs=self.spec.pairs, seats=self.spec.seats,
            timeout=self.spec.timeout,
        )
        return ev.evaluate()

    def _measure_policy_kl(
        self, version_before: VersionHandle, version_after: VersionHandle,
    ) -> (List[float], Optional[float]):
        # Build A(s) once per reference sample (shared by both versions).
        legal_sets: List[LegalActionSet] = [
            enumerate_legal_actions(
                s.legal_actions, status=s.status, inventory=s.inventory,
            )
            for s in self.reference.samples
        ]
        # Probe both versions over the same ν.
        emitted_old = self._probe_version(version_before)
        emitted_new = self._probe_version(version_after)

        # Align by sample index; only non-None pairs contribute.
        chosen_old: List = []
        chosen_new: List = []
        aligned_legal: List[LegalActionSet] = []
        for i, (a, b, las) in enumerate(zip(emitted_old, emitted_new, legal_sets)):
            if len(las) == 0:
                continue
            chosen_old.append(a.primitive if a else None)
            chosen_new.append(b.primitive if b else None)
            aligned_legal.append(las)

        trace = local_policy_kl_trace(
            chosen_new, chosen_old, aligned_legal, self.epsilon,
        )
        # occupancy_shift: simple state-id distribution divergence across the
        # decision points each version actually reached (probed non-None).
        shift = self._occupancy_shift(emitted_old, emitted_new, legal_sets)
        return trace, shift

    def _probe_version(self, version: VersionHandle) -> List[Optional[EmittedAction]]:
        from agentbench_frame.hl.adapter import candidate_command
        cmd, cwd = candidate_command(version, store=self.codebase.store,
                                      dest=self._stage_root / version.version_id)
        probe = self._probe_factory(cmd=cmd, cwd=cwd, timeout=self.spec.timeout)
        try:
            return probe.probe_set(list(self.reference.samples))
        finally:
            probe.close()

    @staticmethod
    def _occupancy_shift(emitted_old, emitted_new, legal_sets) -> Optional[float]:
        # Fraction of decision points where exactly one version emitted
        # (reached) vs the other didn't — a simple, interpretable shift.
        reached_old = sum(1 for e in emitted_old if e is not None)
        reached_new = sum(1 for e in emitted_new if e is not None)
        total = max(1, len(emitted_old))
        return abs(reached_old - reached_new) / total


def _logic_command_stub() -> str:
    # The controller's eval path uses an injected evaluator factory in tests;
    # in production this is the real lostspace logic command. Placeholder.
    return "echo"


def _opponent(name: str):
    from agentbench_frame.lostspace.evaluator import Opponent
    return Opponent(name=name, command="echo")


def _cmd_to_str(cmd, cwd) -> str:
    import shlex
    parts = [shlex.quote(c) for c in cmd]
    return " ".join(parts)

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
        context_builder: Optional[Callable[..., Dict[str, Any]]] = None,
        curriculum: bool = False,
        promote_rank: float = 2.0,
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
        self._context_builder = context_builder
        # Curriculum (B1): when on, evaluate one opponent tier at a time, weakest
        # first, promoting to the next tier only once avg_rank <= promote_rank.
        # Stops the loop from throwing a weak agent straight at the strongest
        # opponent (a guaranteed 0.0 that teaches nothing). _tier indexes
        # spec.opponents.
        self.curriculum = curriculum
        self.promote_rank = promote_rank
        self._tier = 0
        self._last_eval_opponents: Optional[tuple] = None
        self._events = HLEventWriter(run_id=run_id, path=events_path)
        self._act_counter = 0
        # Feedback carried from act N into act N+1's prompt: the eval outcome
        # + policy_kl/occupancy measured at the end of the previous act. None
        # before the first act with a prior result. Without this the loop
        # measures whether an edit changed behavior and then never tells the
        # coding agent — so it keeps making no-op edits.
        self._prev_feedback: Optional[Dict[str, Any]] = None
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
        # act_id is the per-act name <run_id>-00000<m> (one identifier per
        # iteration; see hl/naming.py). ``run_id`` is the round name.
        from agentbench_frame.hl.naming import act_name
        act_id = act_name(self.run_id, self._act_counter)
        t0 = time.monotonic()

        # 1. emit agent_act
        self._events.write(
            "agent_act", act_id=act_id,
            version_before=version_before.version_id if version_before else None,
            edit_policy_mode="free",
            budget_before=self._budget,
        )

        # 2. run the coding agent against the workspace
        context: Dict[str, Any] = {"act_id": act_id, "spec_id": self.spec.spec_id}
        if self._context_builder is not None:
            try:
                extra = self._context_builder(
                    version_before=version_before, act_id=act_id,
                    prev_feedback=self._prev_feedback,
                )
                if extra:
                    context.update(extra)
            except Exception:
                # context is best-effort; never let it break the act loop
                pass
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

        # 4. classify edit_type is already on the handle; emit version event.
        # failure_reason is the run classification: None on success, a stable
        # string (timeout/not-found/non-zero-exit) on failure. Written so a
        # silent timeout is never mistaken for a clean no-op (doc: failures
        # must be visible in the research stream).
        failure_reason = run_result.failure_reason or run_result.error
        if version_after is not None:
            self._events.write(
                "version",
                version_id=version_after.version_id,
                content_hash=version_after.content_hash,
                parent_version_id=version_after.parent_version_id,
                edit_type=version_after.edit_type,
                files_touched=run_result.files_touched,
                failure_reason=failure_reason,
                session_id=run_result.session_id,
                transcript_path=run_result.transcript_path,
            )
        else:
            # Workspace unreadable: still record the failure reason so the
            # act isn't a silent black box.
            self._events.write(
                "version",
                version_id=None,
                content_hash=None,
                parent_version_id=(version_before.version_id
                                   if version_before else None),
                edit_type="noop",
                files_touched=[],
                failure_reason=failure_reason or "workspace unreadable after run",
                session_id=run_result.session_id,
                transcript_path=run_result.transcript_path,
            )

        # 5. eval on the frozen BenchmarkSpec (only if we have a runnable version)
        eval_status = "incomplete"
        win_rate = None
        ev_result = None
        # Capture the opponent tier actually evaluated this act (before any
        # promotion) so the feedback section reports who the outcome was against.
        self._last_eval_opponents = self._active_opponent_names()
        if version_after is not None and self._evaluator_factory is not None:
            try:
                ev_result = self._run_eval(version_after)
                summary = getattr(ev_result, "summary", None) or {}
                # honor the evaluator's own completeness verdict if present
                eval_status = summary.get("evaluation_status") or "complete"
                win_rate = summary.get("win_rate")
                # B1: curriculum promotion — advance to the next (stronger)
                # opponent tier once avg_rank meets the threshold.
                if self.curriculum and eval_status == "complete":
                    agg = (summary.get("lostspace") or {}).get("aggregate") or {}
                    ar = agg.get("avg_rank")
                    if (ar is not None and ar <= self.promote_rank
                            and self._tier < len(self.spec.opponents) - 1):
                        self._tier += 1
            except Exception:
                eval_status = "incomplete"
                win_rate = None
                ev_result = None
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

        # 6.5 carry this act's measurements into the NEXT act's prompt so the
        # coding agent sees whether its edit moved behavior/outcome (not just
        # the events.jsonl research stream). Best-effort: missing fields stay
        # missing, never coerced.
        self._prev_feedback = self._assemble_feedback(
            ev_result=ev_result, kl_trace=kl_trace,
            occupancy_shift=occupancy_shift, run_result=run_result,
            version_after=version_after,
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
            failure_reason=failure_reason,
        )
        self._events.flush()
        return version_after

    # ---- internals ----

    def _run_eval(self, version: VersionHandle):
        """Run the frozen benchmark eval on ``version``.

        The evaluator_factory (supplied by the CLI) captures the real
        logic command, opponents, and filler — it builds a LostSpaceEvaluator
        given ``(version, spec, run_id)`` and returns its ``evaluate()`` result.
        ``opponent_names`` (None for the full pool, or a subset under curriculum)
        lets the controller gate evaluation to one tier at a time.
        """
        ev = self._evaluator_factory(
            version=version, spec=self.spec, run_id=self.run_id,
            opponent_names=self._last_eval_opponents,
        )
        return ev.evaluate()

    def _active_opponent_names(self) -> Optional[tuple]:
        """Names of opponents to evaluate this act. None => the factory uses
        the full pool (no curriculum, or no opponents configured)."""
        if not self.curriculum or not self.spec.opponents:
            return None
        return (self.spec.opponents[self._tier],)

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

    def _assemble_feedback(
        self, *, ev_result, kl_trace: List[float],
        occupancy_shift: Optional[float], run_result, version_after,
    ) -> Dict[str, Any]:
        """Pack the previous act's outcome + behavior-change measurement for the
        next act's prompt. All fields optional — missing stays missing."""
        summary = getattr(ev_result, "summary", None) if ev_result else None
        summary = summary or {}
        agg = (summary.get("lostspace") or {}).get("aggregate") or {}
        kl_mean = (sum(kl_trace) / len(kl_trace)) if kl_trace else None
        n_changed = sum(
            1 for k in kl_trace if k is not None and k > self.epsilon
        )
        return {
            "evaluation_status": summary.get("evaluation_status"),
            "win_rate": summary.get("win_rate"),
            "avg_rank": agg.get("avg_rank"),
            "avg_score": agg.get("avg_score"),
            "avg_turns": agg.get("avg_turns"),
            "kl_mean": kl_mean,
            "occupancy_shift": occupancy_shift,
            "n_changed": n_changed,
            "n_total": len(kl_trace),
            "edit_type": getattr(version_after, "edit_type", None),
            "files_touched": list(run_result.files_touched) if run_result else [],
            "active_opponents": self._last_eval_opponents,
        }

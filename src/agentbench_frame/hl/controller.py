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

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.distribution import (
    enumerate_legal_actions,
    epsilon_smoothed_distribution,
    normalize_emitted,
    tracked_pos_from_transcript,
    policy_kl,
    local_policy_kl_trace,
    PolicyKLPoint,
    ok_kl_values,
    LegalActionSet,
)
from agentbench_frame.hl.events import HLEventWriter, read_events
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction
from agentbench_frame.hl.action_freq import action_freq_kl
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.runner import CodingAgentRunner, AgentRunResult


ProbeFactory = Callable[..., ReferenceProbe]
EvaluatorFactory = Callable[..., Any]


def _anchored_normalize(emitted: Optional[EmittedAction],
                        fallback_pos, obs_pos) -> Optional[Any]:
    """Normalize one version's emitted primitive to the A(s) token form.

    The anchor is the position the CANDIDATE tracks — the probe-reported pos
    (``emitted.pos``, advanced by echoed move replies), else the id-frame spawn
    (``tracked_pos_from_transcript``), else the sample's obs pos. If the tracked
    position differs from the decision-point state's position, the candidate's
    coordinate emissions were computed from a stale frame and cannot be mapped
    faithfully against A(s) — leave them unnormalized (``pos=None``) so they
    register as honest ``out_of_support`` instead of a coincidental direction
    match (the fabricated ν pins states to spawn, so this only triggers on
    real-trace decision points the candidate didn't reach faithfully)."""
    if emitted is None:
        return None
    anchor = (getattr(emitted, "pos", None) or None) or fallback_pos or obs_pos
    if obs_pos and anchor and anchor != obs_pos:
        return normalize_emitted(emitted.primitive, pos=None)
    return normalize_emitted(emitted.primitive, pos=anchor)


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
        dynamic_nu: bool = False,
        nu_samples: int = 24,
        nu_anchor: int = 0,
        nu_max_round: Optional[int] = None,
        action_freq: bool = False,
        evaluator_factory: Optional[EvaluatorFactory] = None,
        probe_factory: Optional[ProbeFactory] = None,
        stage_root: Optional[Path] = None,
        context_builder: Optional[Callable[..., Dict[str, Any]]] = None,
        curriculum: bool = False,
        promote_rank: float = 2.0,
        experience=None,  # ExperienceStore | None (HL std 5)
        data_root: Optional[Path] = None,  # for rules_validation replay lookup
        game: str = "",
    ):
        self.codebase = codebase
        self.runner = runner
        self.spec = spec
        self.reference = reference
        self.run_id = run_id
        self.epsilon = epsilon
        # Rolling dynamic ν: re-record the reference set from the evaluated
        # version's real match traces each act, so policy KL stays live past
        # the first strategy flip (a frozen ν saturates — after edit 1 the new
        # first-actions land out-of-support and every following edit reads
        # KL=0). The seed reference still measures act 1 and supplies the
        # fixed anchor core (see ``_nu_anchor``).
        self.dynamic_nu = dynamic_nu
        self.nu_samples = max(1, int(nu_samples))
        self.nu_max_round = nu_max_round
        # Base spec_id for rolling ν generations ("<seed>-a<act>"); the
        # current self.reference.spec_id already carries the last generation
        # suffix, so deriving from it would compound ("seed-a1-a2-a3").
        self._nu_base_spec = reference.spec_id
        # Fixed anchor: spawn-pinned samples from the seed reference, kept at
        # the head of every rolling ν — a stable axis + guaranteed measurable
        # points even when a recording fails. Empty when nu_anchor==0.
        self._nu_anchor: Tuple[ReferenceSample, ...] = self._pick_anchor(
            reference, int(nu_anchor))
        # Rolling: the version evaluated LAST act's trace paths, fed into the
        # next ``_refresh_dynamic_nu`` so act k's KL compares v_{k-1} vs v_k
        # over states v_{k-1} actually reached (high n_ok, live signal).
        self._pending_traces: List[Path] = []
        # Last ν-refresh summary, surfaced in the next act's feedback so the
        # agent knows the 'reference first-actions' rows target real states.
        self._last_nu_refresh: Optional[Dict[str, Any]] = None
        # Action-frequency KL (--action-freq): per-version counts of the full
        # real-match action mix, KL'd between consecutive versions. This is the
        # channel that sees mid-game edits — the first-primitive probe is blind
        # to them once play()'s top branch stabilizes after one strategy flip.
        self.action_freq = action_freq
        self._version_freq: Dict[str, Counter] = {}
        self._pending_trace_seats: Dict[Path, int] = {}
        self._last_action_kl: Optional[float] = None
        self._last_action_top: Optional[List[Dict[str, Any]]] = None
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
        # Append-only stream (resume-safe): open in append mode so re-running a
        # --name never truncates a prior research stream, and resume the act
        # counter from existing agent_act events so act_ids keep numbering
        # instead of colliding.
        prior_acts = 0
        try:
            prior_acts = sum(
                1 for e in read_events(events_path)
                if e.get("event_type") == "agent_act"
            )
        except OSError:
            prior_acts = 0
        self._events = HLEventWriter(run_id=run_id, path=events_path, append=True)
        self._act_counter = prior_acts
        # Self-summarized experience (HL std 5): persisted lessons fed back
        # into each act's prompt. None disables the feature.
        self._experience = experience
        self._data_root = Path(data_root) if data_root else None
        self._game = game
        # Rolling per-act history for the code-growth nudge (HL std 4):
        # (loc of agent.py, edit_type) per act, oldest first.
        self._loc_history: List[int] = []
        self._edit_type_history: List[str] = []
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
                    act_index=self._act_counter,
                    loc_history=list(self._loc_history),
                    edit_type_history=list(self._edit_type_history),
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
                edit_type=self._resolve_edit_type(run_result, version_before),
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
                act_id=act_id,
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
                act_id=act_id,
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
        # Stash the evaluated version's real-match traces for the next act's
        # dynamic-ν refresh: act k+1's KL will compare v_k vs v_{k+1} over the
        # states v_k actually reached.
        if ev_result is not None:
            self._capture_eval_traces(ev_result)
        # 5.5 action-frequency stats (--action-freq): count the evaluated
        # version's full real-match action mix so step 6.4 can KL it against
        # the previous version's (the channel that sees mid-game edits).
        if (self.action_freq and version_after is not None
                and self._pending_traces):
            from agentbench_frame.hl.action_freq import count_actions
            freq: Counter = Counter()
            for tp in self._pending_traces:
                seat = self._pending_trace_seats.get(tp, 0)
                freq.update(count_actions(tp, seat=seat))
            self._version_freq[version_after.version_id] = freq
        self._events.write(
            "eval", act_id=act_id,
            spec_id=self.spec.spec_id,
            version_after=version_after.version_id if version_after else None,
            evaluation_status=eval_status,
            win_rate=win_rate,  # None stays None
        )

        # 6. probe BOTH versions over ν (Q2) → policy_kl + occupancy_shift
        kl_trace: List[PolicyKLPoint] = []
        occupancy_shift: Optional[float] = None
        ref_digest: List[Dict[str, Any]] = []
        if version_before is not None and version_after is not None:
            kl_trace, occupancy_shift, ref_digest = self._measure_policy_kl(
                version_before, version_after,
            )
            ok_kl = ok_kl_values(kl_trace)
            kl_mean = (sum(ok_kl) / len(ok_kl)) if ok_kl else None
            n_ok = len(ok_kl)
            n_missing = sum(1 for p in kl_trace if p.status != "ok")
            missing_reasons = [p.reason for p in kl_trace if p.reason]
            # Honest top-level reason when strict KL cannot be computed (no ok
            # samples) — never substitute another metric for the missing KL.
            kl_missing_reason = None
            if kl_mean is None and missing_reasons:
                counts = Counter(missing_reasons)
                kl_missing_reason = (
                    f"no ok samples ({n_missing}/{len(kl_trace)} missing): "
                    + ", ".join(f"{r} x{c}" for r, c in counts.most_common())
                )
            self._events.write(
                "policy_kl", act_id=act_id,
                nu_spec_id=self.reference.spec_id,
                version_before=version_before.version_id,
                version_after=version_after.version_id,
                local_policy_kl_trace=[p.to_dict() for p in kl_trace],
                per_sample_status=[p.status for p in kl_trace],
                n_ok=n_ok,
                n_missing=n_missing,
                missing_reasons=missing_reasons,
                kl_mean=kl_mean,
                ig=kl_mean,  # unified per-iteration IG = mean local KL over ν
                kl_missing_reason=kl_missing_reason,
                epsilon=self.epsilon,
            )
            self._events.write(
                "occupancy_shift", act_id=act_id,
                version_before=version_before.version_id,
                version_after=version_after.version_id,
                shift=occupancy_shift,
            )

            # 6.4 action-frequency KL (--action-freq): the evaluated version's
            # full real-match action mix vs the previous version's. This is the
            # channel that sees mid-game edits (first-primitive KL is blind to
            # them once the top branch stabilizes).
            self._last_action_kl = None
            self._last_action_top = None
            if self.action_freq:
                freq_old = self._version_freq.get(version_before.version_id)
                freq_new = self._version_freq.get(version_after.version_id)
                if freq_old is not None and freq_new is not None:
                    akl = action_freq_kl(freq_new, freq_old,
                                         epsilon=self.epsilon)
                    self._last_action_kl = akl
                    self._last_action_top = self._top_actions(
                        freq_new, freq_old, k=6)
                    self._events.write(
                        "action_freq", act_id=act_id,
                        version_before=version_before.version_id,
                        version_after=version_after.version_id,
                        kl=akl,
                        total_actions_new=sum(freq_new.values()),
                        total_actions_old=sum(freq_old.values()),
                        vocab_size=len(set(freq_new) | set(freq_old)),
                        top_actions=self._last_action_top,
                    )

        # 6.5a roll the reference set forward from THIS act's traces so the
        # next act compares v_k vs v_{k+1} over states v_k reached. The KL
        # just measured used the pre-refresh ν (deliberate); a failed refresh
        # keeps the previous ν unchanged (fallback).
        self._refresh_dynamic_nu()
        # 6.5 carry this act's measurements into the NEXT act's prompt so the
        # coding agent sees whether its edit moved behavior/outcome (not just
        # the events.jsonl research stream). Best-effort: missing fields stay
        # missing, never coerced.
        self._prev_feedback = self._assemble_feedback(
            ev_result=ev_result, kl_trace=kl_trace,
            occupancy_shift=occupancy_shift, run_result=run_result,
            version_after=version_after, ref_digest=ref_digest,
        )
        if self._last_nu_refresh:
            self._prev_feedback["nu_refreshed"] = dict(self._last_nu_refresh)
        if self._last_action_kl is not None:
            self._prev_feedback["action_kl"] = self._last_action_kl
            self._prev_feedback["action_top"] = self._last_action_top

        # 6.6 update the self-summarized experience store + rolling LOC history
        # (HL std 4/5). Best-effort: a failure here never breaks the act loop.
        try:
            self._record_experience(act_id, run_result, version_after)
        except Exception:
            pass

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

    def rules_validation_act(self, *, version: Optional[VersionHandle] = None
                             ) -> Optional[Dict[str, Any]]:
        """Run one REPLAY_SKILL validation act (doc Fix-D / Q2 gap 2).

        The coding agent parses one real replay round, writes the meaning of
        each field, and reports its ``score_dic`` claim; the harness then
        INDEPENDENTLY cross-checks the claim against the replay's actual
        ``r[-1]``. The agent's deliverable is its **final message** (no file
        write — the harness persists it as ``artifacts/RULES_VALIDATION.md``
        outside the workspace so it never enters a snapshot diff).

        Emits only a ``rules_validation`` event (no ``version`` / ``eval`` /
        ``policy_kl``). Returns the event payload, or None when no replay
        could be resolved or materialized (a ``validation_status="no_replay"``
        event is still written so the absence is visible in the stream).
        """
        from agentbench_frame.hl.context import _load_rules_doc
        from agentbench_frame.hl.naming import act_name

        act_id = act_name(self.run_id, 0)
        replay = self._resolve_validation_replay(version)
        if replay is None:
            self._events.write(
                "rules_validation", act_id=act_id, replay=None,
                validation_status="no_replay", fields_checked=None,
                mismatches=[], doc_path=None)
            self._events.flush()
            return None

        prompt = _VALIDATION_PROMPT.format(
            rules=_load_rules_doc(), replay=replay,
        )
        context: Dict[str, Any] = {
            "act_id": act_id,
            "prompt": prompt,
            # read_replay/list_replays tools read from these context keys
            "replays_dir": str(replay.parent),
            "replays": [replay.name],
            "transcript_dir": str(self.codebase.root.parent / "transcripts"),
        }
        run_result = self.runner.run(workspace=self.codebase.root, context=context)

        # Harness-side verification: the agent's stated score_dic vs the
        # replay's actual r[-1]. The agent's report is its final message; a
        # missing/truncated message is an honest "no_score_claim", never a pass.
        report = _parse_validation_report(run_result.final_text or "")
        actual = _read_score_dic(replay)
        if report.get("score_dic") is None:
            status = "no_score_claim"
        elif report["score_dic"] == actual:
            status = "pass"
        else:
            status = "fail"
        mismatches = report.get("mismatches") or []
        if status == "fail":
            mismatches.append(
                f"score_dic claimed={report['score_dic']} actual={actual}")
        doc_path = self._write_validation_doc(
            run_result.final_text or "", status, replay, report)

        payload = {
            "act_id": act_id,
            "replay": str(replay),
            "validation_status": status,
            "fields_checked": report.get("fields_checked"),
            "mismatches": mismatches,
            "doc_path": doc_path,
        }
        self._events.write("rules_validation", **payload)
        self._events.flush()
        return payload

    def _resolve_validation_replay(self, version: Optional[VersionHandle]
                                   ) -> Optional[Path]:
        """Locate a real replay for the validation act.

        Prefers an existing run's latest replay (``MatchHistoryView``); if none
        exists yet, materializes one by running a single eval of ``version``
        (writes replay artifacts via ``save_replays=True``) and re-looks.
        Returns None when the replay store is unavailable (no data_root/game
        configured) or an eval produced no replay.
        """
        if self._data_root is None or not self._game:
            return None
        replay = self._latest_replay()
        if replay is not None:
            return replay
        if self._evaluator_factory is None:
            return None
        # No replay yet: materialize one by evaluating the current workspace
        # (the seeded candidate at validation time) if no version is in hand.
        if version is None:
            try:
                version = self.codebase.snapshot(
                    parent_version_id=None, edit_type="initial")
            except Exception:
                return None
        try:
            self._run_eval(version)
        except Exception:
            return None
        return self._latest_replay()

    def _latest_replay(self) -> Optional[Path]:
        from agentbench_frame.hl.resources import MatchHistoryView

        view = MatchHistoryView(
            data_root=self._data_root, game=self._game, agent=self.run_id)
        rows = view.match_rows()
        for latest in reversed(rows):
            run_dir = self._data_root / "runs" / self._game / self.run_id \
                / latest.get("run_id", "")
            rel = latest.get("replay")
            if not rel:
                continue
            p = run_dir / rel
            if p.is_file():
                return p
        return None

    def _write_validation_doc(self, final_text: str, status: str,
                              replay: Path, report: Dict[str, Any]) -> Optional[str]:
        """Persist the agent's validation report outside the workspace.

        ``<round_root>/artifacts/RULES_VALIDATION.md`` — never inside the
        workspace, so it cannot leak into a snapshot diff. Best-effort; a write
        failure returns None (the event still records the verdict)."""
        try:
            artifacts = self.codebase.root.parent / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            doc = artifacts / "RULES_VALIDATION.md"
            lines = [
                f"# RULES_VALIDATION — {self.run_id}",
                f"- replay: {replay}",
                f"- verdict: {status}",
                f"- fields_checked: {report.get('fields_checked')}",
                f"- score_dic claimed: {report.get('score_dic')}",
                f"- mismatches: {report.get('mismatches')}",
                "",
                "## Agent report",
                final_text or "(no final message)",
                "",
            ]
            doc.write_text("\n".join(lines), encoding="utf-8")
            return str(doc)
        except OSError:
            return None

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

    def _resolve_edit_type(self, run_result, version_before) -> str:
        """Classify this act's edit_type.

        ``ClaudeCodeRunner`` returns ``edit_type=None`` ("unclassified") on
        success — the CLI gives us no semantic label, so we classify from the
        actual content change via ``codebase.diff``. ``FakeRunner`` (and any
        runner that knows what it did) returns a concrete label, which we trust
        unchanged.

        Honest classification from the diff:
        - no parent → ``initial`` (the snapshot() call also enforces this).
        - content unchanged from the parent (identical content_hash) →
          ``noop`` — the agent edited nothing. Previously this was mislabeled
          ``refactor`` because the runner hard-coded ``refactor``.
        - files added or removed → ``replace`` (the agent swapped structure).
        - only ``agent.py`` (and/or helper modules) modified → ``parametrize``
          (a real behavioral edit to existing code; the smallest honest
          non-noop label — we cannot semantically distinguish add_rule vs
          parametrize vs refactor from a byte-diff alone, and refuse to claim
          the stronger ``refactor`` = behavior-preserving without evidence).

        This is best-effort: the *behavioral* truth is measured separately by
        policy_kl. ``noop`` detection is the real fix — it stops masking
        "agent edited nothing" as a refactor in the research stream.
        """
        declared = getattr(run_result, "edit_type", None)
        if declared is not None:
            return declared
        if version_before is None:
            return "initial"
        try:
            after_hash = self._workspace_content_hash()
        except Exception:
            return "noop"
        if after_hash is None:
            return "noop"
        if version_before.content_hash == after_hash:
            return "noop"
        # content changed — classify by file-level diff against the parent's
        # stored snapshot. We diff the parent snapshot (in the store) against
        # the LIVE workspace (not a store entry), because the after-snapshot
        # has not been written yet at this point in the act lifecycle.
        try:
            from agentbench_frame.hl.codebase import _walk
            before_files = {rel: data for rel, data
                            in _walk(self.codebase.store / version_before.content_hash)}
            after_files = {rel: data for rel, data in _walk(self.codebase.root)}
            b_keys, a_keys = set(before_files), set(after_files)
            added = a_keys - b_keys
            removed = b_keys - a_keys
        except Exception:
            return "parametrize"
        if added or removed:
            return "replace"
        return "parametrize"

    def _workspace_content_hash(self) -> Optional[str]:
        """Content-hash of the live workspace tree (mirrors HLCodebase's
        hashing so it compares equal to a stored snapshot's hash). Returns
        None if the tree is unreadable."""
        from agentbench_frame.hl.codebase import _walk, _content_hash
        try:
            return _content_hash(_walk(self.codebase.root))
        except Exception:
            return None

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

        # Align by sample index; only non-None pairs contribute. Normalize each
        # emitted primitive to the A(s) token form (move/detect/attack/Transport
        # arrive with coordinates, A(s) codes direction indices / bare ids) so
        # real behavioral flips register as policy_kl instead of collapsing to
        # out_of_support.
        chosen_old: List = []
        chosen_new: List = []
        aligned_legal: List[LegalActionSet] = []
        for i, (a, b, las) in enumerate(zip(emitted_old, emitted_new, legal_sets)):
            if len(las) == 0:
                continue
            s_i = self.reference.samples[i]
            fallback = tracked_pos_from_transcript(s_i.transcript)
            obs_pos = (s_i.observation or {}).get("pos")
            chosen_old.append(_anchored_normalize(a, fallback, obs_pos))
            chosen_new.append(_anchored_normalize(b, fallback, obs_pos))
            aligned_legal.append(las)

        trace = local_policy_kl_trace(
            chosen_new, chosen_old, aligned_legal, self.epsilon,
        )
        # occupancy_shift: simple state-id distribution divergence across the
        # decision points each version actually reached (probed non-None).
        shift = self._occupancy_shift(emitted_old, emitted_new, legal_sets)
        # Per-ok-point first-action digest, fed into the NEXT act's prompt so
        # the coding agent sees exactly which measurable decision points its
        # edit did (not) move — the actionable target for a valid update.
        digest = self._first_action_digest(
            emitted_old, emitted_new, legal_sets, trace)
        return trace, shift, digest

    def _first_action_digest(self, emitted_old, emitted_new, legal_sets,
                             trace) -> List[Dict[str, Any]]:
        """Compact per-ok-point first-action digest for the next act's prompt.

        Each entry names one reference decision point where BOTH versions
        emitted an in-support primitive, the state it saw, and what each
        version sent. The coding agent reads this to know exactly which
        measurable decision its edit did (not) move — the actionable target
        for a valid policy update. Capped at 8 rows to keep the prompt small.
        """
        out: List[Dict[str, Any]] = []
        t = 0
        for i, (a_old, a_new, las) in enumerate(
                zip(emitted_old, emitted_new, legal_sets)):
            if len(las) == 0:
                continue
            pt = trace[t]
            t += 1
            if pt.status != "ok":
                continue
            s = self.reference.samples[i]
            obs = s.observation or {}
            inv = s.inventory or {}
            out.append({
                "idx": i,
                "hp": obs.get("hp"),
                "keys": len(obs.get("keys") or []),
                "kit": inv.get("Kit", 0),
                "interprops": (s.legal_actions or {}).get("interprops", []),
                "old": a_old,
                "new": a_new,
            })
            if len(out) >= 8:
                break
        return out

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

    # ---- rolling dynamic ν ----

    @staticmethod
    def _spawn_pinned(s: ReferenceSample) -> bool:
        """True iff a sample's observation pos is the seat-0 spawn [0,0].

        The fabricated ν pins states to spawn (so the candidate's tracked
        position from the 2-frame transcript equals the decision-point state),
        which is exactly what keeps ``normalize_emitted`` in-support. Anchor
        candidates are drawn from these — the reliably measurable points."""
        pos = (s.observation or {}).get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                return int(pos[0]) == 0 and int(pos[1]) == 0
            except (TypeError, ValueError):
                return False
        return False

    @classmethod
    def _pick_anchor(cls, reference: ReferenceStateSet, n: int
                     ) -> Tuple[ReferenceSample, ...]:
        """Up to ``n`` spawn-pinned seed samples, kept as the rolling ν's
        fixed core. Spawn-pinned points are the ones whose emissions reliably
        normalize against A(s), so the anchor is a stable, measurable axis even
        when a recording fails. Falls back to the first samples overall when no
        spawn-pinned ones exist."""
        if n <= 0 or not reference.samples:
            return ()
        pinned = [s for s in reference.samples if cls._spawn_pinned(s)]
        pool = pinned or list(reference.samples)
        return tuple(pool[:n])

    def _capture_eval_traces(self, ev_result) -> None:
        """Stash the evaluated version's per-match ``.trace.jsonl`` paths for
        the next act's dynamic-ν refresh / action-freq stats.

        Trace paths are relative to the eval's ``run_dir``
        (``evaluator.py:226``); resolve and keep only files that exist. Each
        trace's candidate seat (``candidate_seat``) is remembered so
        action-frequency counting targets the right player. Best-effort: no
        run_dir / no trace records / missing files → empty."""
        traces: List[Path] = []
        seats: Dict[Path, int] = {}
        run_dir = getattr(ev_result, "run_dir", None)
        if run_dir is not None:
            for m in (getattr(ev_result, "matches", None) or []):
                rel = m.get("trace") if isinstance(m, dict) else None
                if not rel:
                    continue
                p = Path(run_dir) / rel
                if p.is_file():
                    traces.append(p)
                    try:
                        seat = int(m.get("candidate_seat", 0))
                    except (TypeError, ValueError):
                        seat = 0
                    seats[p] = seat
        self._pending_traces = traces
        self._pending_trace_seats = seats

    def _refresh_dynamic_nu(self) -> None:
        """Rolling dynamic ν: re-record decision points from the previous act's
        real match traces, prepend the fixed anchor, and swap ``self.reference``
        (used by the NEXT act's KL measurement).

        Keeps policy KL live past the first strategy flip: a frozen ν saturates
        once the new first-actions stop being in-support, while a ν re-recorded
        from the evaluated version's own states stays aligned with where the
        policy actually acts. Act k+1's KL then compares v_k vs v_{k+1} over
        the states v_k reached — the older version is in-support by
        construction, so mid-game decision changes register KL every act.

        Best-effort + honest: too few usable samples (<4) or no traces at all
        keeps the previous ν (fallback), and every swap is announced via a
        ``reference_refresh`` event + the next act's feedback.
        """
        self._last_nu_refresh = None
        if not self.dynamic_nu or not self._pending_traces:
            return
        from agentbench_frame.hl.naming import act_name
        from agentbench_frame.hl.reference_recorder import (
            record_reference_states, subsample)

        combined: List[ReferenceSample] = []
        sources: List[str] = []
        for tp in self._pending_traces:
            try:
                rss = record_reference_states(
                    tp, spec_id=f"dyn-a{self._act_counter}",
                    opponent=str(tp.parent.name), seat=0,
                )
            except Exception:
                continue  # a broken trace must not kill the act loop
            if rss.samples:
                combined.extend(rss.samples)
                sources.append(str(tp))
        if len(combined) < 4:
            return  # fallback: keep the previous ν
        picked = subsample(combined, max_count=self.nu_samples,
                           max_round=self.nu_max_round)
        samples = list(self._nu_anchor) + list(picked)
        if not samples:
            return
        self.reference = ReferenceStateSet(
            spec_id=f"{self._nu_base_spec}-a{self._act_counter}",
            samples=tuple(samples),
        )
        self._last_nu_refresh = {
            "n": len(samples), "fresh": len(picked),
            "anchor": len(self._nu_anchor), "sources": sources,
        }
        self._events.write(
            "reference_refresh", act_id=act_name(self.run_id, self._act_counter),
            n=len(samples), fresh=len(picked), anchor=len(self._nu_anchor),
            sources=sources, spec_id=self.reference.spec_id,
        )

    @staticmethod
    def _top_actions(freq_new: Counter, freq_old: Counter, k: int = 6
                     ) -> List[Dict[str, Any]]:
        """The most-frequent canonical actions for the action_freq event +
        feedback digest. Ranked by the NEW version's count (ties by old), so
        the headline rows are the actions the edit actually shifted."""
        union = sorted(set(freq_new) | set(freq_old))

        def rank(t):
            return (-freq_new.get(t, 0), -freq_old.get(t, 0), str(t))

        top = sorted(union, key=rank)[:k]
        return [{
            "action": list(t) if isinstance(t, tuple) else t,
            "count_new": freq_new.get(t, 0),
            "count_old": freq_old.get(t, 0),
        } for t in top]

    def _assemble_feedback(
        self, *, ev_result, kl_trace: List[PolicyKLPoint],
        occupancy_shift: Optional[float], run_result, version_after,
        ref_digest: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Pack the previous act's outcome + behavior-change measurement for the
        next act's prompt. All fields optional — missing stays missing.

        ``kl_mean``/``n_changed``/``n_total`` derive from the ok-only trace
        values: a ``no_emission``/``out_of_support`` point never enters a mean
        or a changed-count denominator (Fix-A honesty).

        ``ref_points`` = ``_first_action_digest`` output: the per-measurable-
        point first actions so the next prompt names concrete targets."""
        summary = getattr(ev_result, "summary", None) if ev_result else None
        summary = summary or {}
        agg = (summary.get("lostspace") or {}).get("aggregate") or {}
        ok_kl = ok_kl_values(kl_trace)
        kl_mean = (sum(ok_kl) / len(ok_kl)) if ok_kl else None
        n_changed = sum(1 for k in ok_kl if k > self.epsilon)
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
            "ref_points": ref_digest or [],
        }

    def _record_experience(self, act_id: str, run_result, version_after) -> None:
        """Append this act's raw observation to the experience store and
        update the rolling LOC/edit-type history used by the code-growth
        nudge (HL std 4/5). Best-effort, called in a try/except."""
        et = getattr(version_after, "edit_type", None) or "noop"
        self._edit_type_history.append(et)
        self._loc_history.append(self._workspace_loc())
        if self._experience is None:
            return
        # seat0 digest is intentionally not recomputed here — the per-act prompt
        # already surfaces it via ContextBuilder; the store keeps the
        # outcome-level signal (win_rate / avg_rank / opponents / edit_type).
        self._experience.propose_update(
            act_id=act_id, feedback=self._prev_feedback or {},
            seat0_digest=None,
        )

    def _workspace_loc(self) -> int:
        """Line count of the current workspace ``agent.py`` (0 if unreadable).
        A cheap proxy for strategy surface size, used to detect piling."""
        try:
            p = self.codebase.root / "agent.py"
            return p.read_text(encoding="utf-8").count("\n")
        except Exception:
            return 0


#: The rules_validation act prompt (doc Fix-D). The agent's deliverable is its
#: FINAL MESSAGE in the exact report format below — the harness parses the
#: ``SCORE_DIC`` line and cross-checks it against the replay's real ``r[-1]``.
_VALIDATION_PROMPT = """\
# REPLAY_SKILL validation act — prove you read the rules and replay format correctly

This is NOT an edit act. Do NOT call `str_replace`. Do NOT modify any file.

Read the authoritative rules + replay skill doc below, then parse ONE round of
the real replay at `{replay}` (use the `read_replay` tool):
- For a chosen round and your seat 0, write the meaning of each field in the
  first action dict, and what you conclude the player did.
- State the `score_dic` you read from the replay's last element and the
  resulting ranking (sort player ids by score descending; 4 = 1st, 1 = 4th).

You MAY use `read_replay` to inspect the replay. Your deliverable is your
FINAL message, ending with exactly three lines:

FIELDS_CHECKED: <int>
SCORE_DIC: <the score_dic you read, as JSON like {{"0": 4, "1": 3, "2": 2, "3": 1}}>
MISMATCHES: <comma-separated list, or "none">

If you cannot complete the parse, say so and emit SCORE_DIC: none.

--- rules + replay skill doc ---
{rules}
"""


def _parse_validation_report(final_text: str) -> Dict[str, Any]:
    """Best-effort parse of the agent's ``FIELDS_CHECKED`` / ``SCORE_DIC`` /
    ``MISMATCHES`` report from its final message. Absent fields stay None —
    never guessed (a missing score claim is an honest ``no_score_claim``)."""
    import re

    out: Dict[str, Any] = {"fields_checked": None, "score_dic": None,
                           "mismatches": []}
    m = re.search(r"FIELDS_CHECKED\s*:\s*(\d+)", final_text)
    if m:
        out["fields_checked"] = int(m.group(1))
    m = re.search(r"SCORE_DIC\s*:\s*(\{[^}]*\})", final_text)
    if m:
        try:
            out["score_dic"] = json.loads(m.group(1))
        except (ValueError, TypeError):
            out["score_dic"] = None
    m = re.search(r"MISMATCHES\s*:\s*(.+)", final_text)
    if m:
        raw = m.group(1).strip()
        out["mismatches"] = ([] if raw.lower() in ("", "none")
                             else [s.strip() for s in raw.split(",") if s.strip()])
    return out


def _read_score_dic(replay_path) -> Optional[Dict[str, Any]]:
    """Read the actual ``score_dic`` (the replay array's last element)."""
    import json as _json
    try:
        data = _json.loads(Path(replay_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    return last if isinstance(last, dict) else None

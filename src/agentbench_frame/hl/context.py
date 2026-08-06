"""
ContextBuilder — assemble the prompt the coding agent sees each HL act.

The coding agent (Claude Code) runs with ``cwd`` at the codebase workspace,
so it can read/edit ``agent.py`` directly. But it has no idea *what* to
improve unless we hand it: the match record so far, where the replays live,
how to read them, and the game rules. That's this module's job.

It produces a single prompt string (plus a timeout). It writes **nothing**
into the workspace — any file dropped there would be swept into the next
content-hash snapshot and pollute the diff. Everything goes in the prompt.

The replay-reading recipe is lifted from
``AgentBenchResults/skills/lostspace-playback/SKILL.md`` so the agent can
self-serve the "why did I lose?" question without us embedding 20 KB of JSON.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.reference import BenchmarkSpec
from agentbench_frame.hl.resources import MatchHistoryView


# All static prompt copy (system prompt, blurbs, recipes, mission blocks,
# feedback fragments) lives in ``prompts.py`` — edit it there, not here.
# This module only assembles dynamic per-act sections from live match data.
from agentbench_frame.hl import prompts


class ContextBuilder:
    """Build the per-act coding-agent prompt from live match data."""

    def __init__(
        self,
        *,
        codebase: HLCodebase,
        data_root,
        game: str,
        agent_name: str,
        spec: BenchmarkSpec,
        playback_skill_path: Optional[Any] = None,
        timeout: float = 600.0,
        experience: Any = None,
        consolidate_every: int = 4,
        max_growth_pct: float = 40.0,
    ):
        self.codebase = codebase
        self.data_root = Path(data_root)
        self.game = game
        self.agent_name = agent_name
        self.spec = spec
        self.playback_skill_path = (
            Path(playback_skill_path) if playback_skill_path else None
        )
        self.timeout = timeout
        # Self-summarized experience (HL std 5) + consolidation cadence (std 4).
        # ``experience`` is an ExperienceStore (or None to disable). The
        # consolidate-every cadence swaps the per-act mission to a
        # consolidation pass every K-th act; ``max_growth_pct`` gates the
        # code-growth nudge in the feedback section.
        self.experience = experience
        self.consolidate_every = max(0, int(consolidate_every))
        self.max_growth_pct = float(max_growth_pct)

    def build(
        self,
        *,
        version_before: Optional[VersionHandle],
        act_id: str,
        prev_feedback: Optional[Dict[str, Any]] = None,
        act_index: int = 1,
        loc_history: Optional[List[int]] = None,
        edit_type_history: Optional[List[str]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        return {
            "prompt": self._prompt(
                version_before, act_id, prev_feedback,
                act_index=act_index,
                loc_history=loc_history,
                edit_type_history=edit_type_history,
            ),
            "timeout": self.timeout,
        }

    # ---- internals ----

    def _prompt(self, version_before: Optional[VersionHandle],
                act_id: str,
                prev_feedback: Optional[Dict[str, Any]] = None,
                *,
                act_index: int = 1,
                loc_history: Optional[List[int]] = None,
                edit_type_history: Optional[List[str]] = None) -> str:
        lines: list[str] = []
        is_consolidation = (
            self.consolidate_every > 0
            and act_index > 1
            and act_index % self.consolidate_every == 0
        )
        lines.append(
            f"# HL act {act_id} — improve the LostSpace agent\n"
            "You are editing a LostSpace agent in this workspace. "
            "Your goal is to raise its win rate against the benchmark "
            f"opponents ({', '.join(self.spec.opponents)}).\n"
            + prompts.RULES_OF_ENGAGEMENT
        )
        lines.append(f"## Game rules\n{prompts.load_rules_doc()}\n")
        lines.append(f"## Data schema\n{prompts.DATA_SCHEMA_BLURB}\n")

        history = self._history_section()
        if history:
            lines.append(f"## Match history so far\n{history}\n")

        replay = self._replay_section()
        if replay:
            lines.append(f"## Replays — dig in if you need the 'why'\n{replay}\n")
        else:
            # First act: no replays yet, but teach the format so the agent is
            # ready to read them on later acts.
            lines.append(
                "## How to read replays (later acts will have them)\n"
                + prompts.PLAYBACK_RECIPE + "\n"
            )

        if version_before is not None:
            lines.append(
                "## Your previous version\n"
                + prompts.PREVIOUS_VERSION.format(
                    version_id=version_before.version_id,
                    content_hash=version_before.content_hash,
                    edit_type=version_before.edit_type,
                ) + "\n"
            )
        else:
            lines.append("## First act\n" + prompts.FIRST_ACT + "\n")

        feedback = self._feedback_section(
            prev_feedback, loc_history=loc_history,
            edit_type_history=edit_type_history,
        )
        if feedback:
            lines.append(feedback)

        experience = self._experience_section()
        if experience:
            lines.append(experience)

        if is_consolidation:
            lines.append(self._consolidation_mission())
        else:
            lines.append(
                prompts.WHAT_TO_DO_NOW + "\n" + "\n"
                + prompts.REQUIRED_BEHAVIORAL_CHANGE + "\n" + "\n"
                + prompts.MEASUREMENT_LOCATION + "\n"
            )
        lines.append(prompts.STAY_ON_MISSION + "\n")
        return "\n".join(lines)

    def _feedback_section(self, fb: Optional[Dict[str, Any]], *,
                          loc_history: Optional[List[int]] = None,
                          edit_type_history: Optional[List[str]] = None) -> str:
        """Render the previous act's outcome + behavior-change measurement.

        This is the closed-loop signal: the controller measures win_rate and
        policy_kl every act, and this section hands them back so the agent
        learns whether its last edit helped, hurt, or did nothing. Without it
        the agent cannot tell a no-op edit from a productive one.
        """
        if not fb:
            return ""
        def fmt(v, spec="%g"):
            return "-" if v is None else format(v, spec.lstrip("%"))
        win_rate = fb.get("win_rate")
        wr_s = (f"{win_rate:.0%}" if isinstance(win_rate, (int, float))
                else "-")
        outcome = (
            f"win_rate={wr_s} avg_rank={fmt(fb.get('avg_rank'), '%.1f')} "
            f"avg_score={fmt(fb.get('avg_score'), '%.2f')} "
            f"avg_turns={fmt(fb.get('avg_turns'), '.0f')} "
            f"(eval={fb.get('evaluation_status') or '-'})"
        )
        kl_mean = fb.get("kl_mean")
        n_changed = fb.get("n_changed", 0)
        n_total = fb.get("n_total", 0)
        occ = fb.get("occupancy_shift")
        behavior = prompts.FEEDBACK_BEHAVIOR_LINE.format(
            kl_mean=fmt(kl_mean, '%.4g'),
            n_changed=n_changed, n_total=n_total,
            occupancy=fmt(occ, '%.3g'),
        )
        parts = [
            "## Feedback on your last edit",
            f"- outcome: {outcome}",
        ]
        active = fb.get("active_opponents")
        if active:
            parts.append(f"- evaluated against: {', '.join(active)}")
        parts.append(f"- behavior: {behavior}")
        # Rolling dynamic ν (--dynamic-nu): the reference set was re-recorded
        # from the previous act's real match this act, so the first-actions
        # digest below targets REAL states the agent reached — not the
        # hand-authored seed points.
        nu_ref = fb.get("nu_refreshed")
        if nu_ref:
            parts.append(
                f"- reference refreshed: {nu_ref.get('n')} decision points "
                f"({nu_ref.get('fresh')} recorded from your last match + "
                f"{nu_ref.get('anchor')} anchor). The rows below are states "
                "you actually reached in your last eval."
            )
        # Action-frequency KL (--action-freq): the full-match action-mix shift
        # vs last version — the channel that sees mid-game edits (move-target
        # weights, loot/escape timing), not just the first emitted primitive.
        action_kl = fb.get("action_kl")
        if action_kl is not None:
            rows = []
            for r in (fb.get("action_top") or []):
                act = r["action"]
                a = " ".join(str(x) for x in act) if isinstance(act, list) else act
                rows.append(f"  {a}: {r['count_old']} -> {r['count_new']}")
            parts.append(
                prompts.ACTION_FREQ_INTRO.format(action_kl=fmt(action_kl, '%.4g'))
                + "\n" + "\n".join(rows)
            )
        # Per-measurable-point first-action digest: name the exact decision
        # points the edit did (not) move, so the next edit has a concrete
        # target for a valid policy update (policy_kl > 0).
        ref_points = fb.get("ref_points") or []
        if ref_points:
            rows = []
            for r in ref_points:
                flag = "CHANGED" if r["old"] != r["new"] else "UNCHANGED"
                rows.append(
                    f"  R{r['idx']} hp={r['hp']} keys={r['keys']} "
                    f"kit={r['kit']} props={r['interprops']}: "
                    f"old={r['old']} new={r['new']} ({flag})")
            parts.append(
                "- reference first-actions (both versions in-support):\n"
                + "\n".join(rows)
                + "\n" + prompts.REF_POINTS_CTA
            )
        # No-op callout: the edit landed but changed zero reference decisions
        # and produced no occupancy shift. This is the exact failure mode that
        # stalled prior iterations (plausible edits in dead/dominated code).
        no_behavior_change = (
            n_total > 0 and n_changed == 0
            and (occ is None or occ == 0)
        )
        if no_behavior_change:
            parts.append(prompts.NO_MEASURABLE_EFFECT.format(n_total=n_total))
        nudge = self._growth_nudge(loc_history, edit_type_history)
        if nudge:
            parts.append(nudge)
        return "\n".join(parts) + "\n"

    def _growth_nudge(self, loc_history: Optional[List[int]],
                      edit_type_history: Optional[List[str]]) -> str:
        """Code-growth nudge (HL std 4): when the last few acts were
        rule-piling (``add_rule``/``parametrize``) and ``agent.py`` grew
        beyond ``max_growth_pct`` over that window, surface a 'consider a
        consolidation pass' nudge. Empty string when not triggered."""
        if not loc_history or len(loc_history) < 2:
            return ""
        # Look at the trailing window of piling edits.
        window_et = (edit_type_history or [])[-len(loc_history):]
        piling = [e for e in window_et if e in ("add_rule", "parametrize")]
        if len(piling) < 2:
            return ""
        first, last = loc_history[-len(piling)], loc_history[-1]
        if first <= 0:
            return ""
        growth_pct = (last - first) / first * 100.0
        if growth_pct < self.max_growth_pct:
            return ""
        return prompts.CODE_GROWTH.format(
            first=first, last=last, pct=growth_pct, n_piling=len(piling))

    def _experience_section(self) -> str:
        """Render accumulated lessons (HL std 5). The agent-authored
        ``EXPERIENCE.md`` if present, else a compact view of raw staging
        observations. Empty when experience is disabled or empty."""
        if self.experience is None:
            return ""
        rendered = self.experience.render()
        if not rendered:
            return ""
        return prompts.EXPERIENCE_SECTION.format(
            rendered=rendered, path=self.experience.path)

    def _consolidation_mission(self) -> str:
        """The periodic consolidation mission (HL std 4): swap the act's goal
        from 'add one improvement' to 'compress and consolidate'."""
        xp_edit = ""
        if self.experience is not None:
            xp_edit = (
                "\n2. Re-summarize the experience file "
                f"`{self.experience.path}`: fold the new raw observations into "
                "the existing lessons, deduplicate, retire disproven ideas "
                "into a 'Retired ideas' section, and keep it short. This is "
                "your self-summarized memory — compress it, don't just append."
            )
        return prompts.CONSOLIDATION_MISSION.format(
            experience_edit=xp_edit) + "\n"

    def _history_section(self) -> str:
        view = MatchHistoryView(
            data_root=self.data_root, game=self.game, agent=self.agent_name
        )
        by_opp = view.by_opponent()
        if not by_opp:
            return ""
        rows = ["opponent  W  L  err  win_rate  avg_rank  avg_score  avg_turns"]
        for name, a in by_opp.items():
            wr = a.get("win_rate")
            wr_s = f"{wr:.0%}" if wr is not None else "-"
            ar = a.get("avg_rank")
            ar_s = f"{ar:.1f}" if ar is not None else "-"
            asc = a.get("avg_score")
            asc_s = f"{asc:.1f}" if asc is not None else "-"
            atn = a.get("avg_turns")
            atn_s = f"{atn:.0f}" if atn is not None else "-"
            rows.append(
                f"{name}  {a['wins']}  {a['losses']}  {a['errors']}  "
                f"{wr_s}  {ar_s}  {asc_s}  {atn_s}"
            )
        return "\n".join(rows)

    def _replay_section(self) -> str:
        view = MatchHistoryView(
            data_root=self.data_root, game=self.game, agent=self.agent_name
        )
        rows = view.match_rows()
        if not rows:
            return ""
        latest = rows[-1]
        run_dir = (self.data_root / "runs" / self.game / self.agent_name
                   / latest.get("run_id", ""))
        matches_jsonl = run_dir / "matches.jsonl"
        artifacts = run_dir / "artifacts"
        out = [
            f"Latest run dir: {run_dir}",
            f"- matches.jsonl: {matches_jsonl}",
            f"- artifacts/   : {artifacts}  (one replay JSON per match)",
            f"- last match   : opponent={latest.get('opponent')} "
            f"result={latest.get('candidate_result')} "
            f"rank={latest.get('candidate_rank')} turns={latest.get('turns')}",
        ]
        # Seat-0 failure digest: load the last match's replay (if present) and
        # summarize *why* seat 0 lost — keys/escaped/died/rounds/last-action.
        # Best-effort: a missing/unparseable replay never breaks the prompt.
        digest = self._seat0_digest_line(run_dir, latest)
        if digest:
            out.append(f"- {digest}")
        out += ["", "How to read a replay:", prompts.PLAYBACK_RECIPE]
        return "\n".join(out)

    @staticmethod
    def _seat0_digest_line(run_dir: Path, latest: Dict[str, Any]) -> str:
        from agentbench_frame.hl.resources import ReplayView

        replay_rel = latest.get("replay")
        if not replay_rel:
            return ""
        replay_abs = run_dir / replay_rel
        if not replay_abs.exists():
            return ""
        try:
            d = ReplayView(replay_abs).seat0_digest()
        except Exception:
            return ""
        died = f"died={d['died']}"
        if d["died_round"] is not None:
            died += f"@round{d['died_round']}"
            if d.get("attacked_near_death"):
                died += "(attacked)"
        bits = [f"keys={d['keys']}", f"escaped={d['escaped']}", died,
                f"ai_errors={d['ai_errors']}", f"rounds={d['n_rounds']}"]
        if d.get("escape_starts") or d.get("escape_aborts"):
            bits.append(f"escape_started={d['escape_starts']} "
                        f"escape_aborted={d['escape_aborts']}")
        if d.get("keys_by_round"):
            bits.append(f"key_path={d['keys_by_round']}")
        if d["last_action"]:
            bits.append(f"last={d['last_action']}")
        # Loud callout: a game that ran long enough to collect keys but where
        # seat 0 collected NONE and did not escape is almost certainly a broken
        # win-condition path (the interprops-gating regression). Make it an
        # explicit instruction, not a buried count.
        if (d.get("keys", 0) == 0 and not d.get("escaped")
                and d.get("n_rounds", 0) >= 20):
            bits.append(
                "ACTION: 0 keys + no escape = key collection is broken. Do "
                "NOT gate interact('KeyMachine') on interprops membership "
                "(int/object-coded, never the string 'KeyMachine'). Call "
                "interact('KeyMachine') and branch on result['success']. See "
                "the Data schema section.")
        # Escape-flag reminder whenever keys were collected but no win: the
        # most common invisible failure is sending False (abort) while Alive,
        # which the server rejects (interactive_props.py:113). Start = True.
        if d.get("keys", 0) > 0 and not d.get("escaped"):
            bits.append("escape=start needs interact('EscapeCapsule', True) "
                        "(False only aborts while WaitForEscape)")
        return ("seat 0 last game: " + " ".join(bits)
                + "  (replays are event-summarized; counts are lower bounds)")

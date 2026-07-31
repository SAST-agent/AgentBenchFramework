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


# The inline playback recipe (condensed from the SKILL.md so the coding agent
# can read replays without the skill being loaded). Kept short — just enough
# to orient, with the absolute skill path for the full version.
_PLAYBACK_RECIPE = """\
Replays are JSON arrays: [birthplaces, round_1, ..., round_N, score_dic].
  - r[1:-1]   = rounds; each round is 4 turns (one per seat 0..3).
  - r[-1]     = score_dic {"0":4,"1":3,...} (4=1st ... 1=4th). You are seat 0.
  - Coords are shifted by -3: grid = replay + 3. Capsule is at grid [3,3,0].
  - Action `type` values: move/flink, attack, getkey, keymachine,
    escape_capsule (to_escape), escaped (won), place_trap, detect, died,
    regenerate, ai_error. `ai_error` on seats 1-3 is background noise
    (ranked algos rely on judger TLE) — only worry about seat 0.
Quick scan: python -c "import json;print(json.load(open('REPLAY'))[-1])"
Full reader: see AgentBenchResults/skills/lostspace-playback/SKILL.md"""

_GAME_RULES_BLURB = (
    "LostSpace: 4-player FFA on a 7x7x3 grid. Win by collecting 4 corner keys "
    "(one per KeyMachine) then escaping via the center capsule. Scoring: "
    "+1 key, +2 kill, -3 death. Ranking = escape order, then survivors by "
    "score. First to escape = rank 1."
)

_DATA_SCHEMA_BLURB = (
    "`self.view.nodes[i].interprops` is a list of INTEGER CODES / objects "
    "(1=EscapeCapsule, 2=KeyMachine); the agent client also appends the "
    "string 'Box'. It is NOT a list of strings like 'KeyMachine', so "
    "`if 'KeyMachine' in interprops` (or 'EscapeCapsule') is always False — "
    "never gate an action on it. Safe pattern (already used by the seed): "
    "call the action blind, then branch on the returned `['success']`, e.g. "
    "`if self.interact('KeyMachine')['success']: return` — the server returns "
    "success only when the action is valid. Win condition: interact with each "
    "of the 4 corner KeyMachines (collect 4 keys), then interact with the "
    "center EscapeCapsule. An agent that never calls interact('KeyMachine') "
    "can never win."
)


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
            "RULES OF ENGAGEMENT:\n"
            "- Edit only the agent source (`agent.py` and any helper modules "
            "in this directory). Do NOT touch `manifest.toml`.\n"
            "- The agent may be ANY interpretable Python — not just if-else. "
            "You are encouraged to use utility/scoring functions, weighted "
            "evaluation of candidate moves, bounded lookahead or shallow "
            "search, explicit planners, and parametrized decision tables — "
            "whichever is the smallest change that fixes the weakest matchup. "
            "The only constraint: the logic must stay HUMAN-READABLE (no "
            "opaque black-box blobs, no dumped learned weights without an "
            "interpretable wrapper).\n"
            "- Keep the Saiblo stdio protocol intact (read 4-byte "
            "length-prefixed JSON, send the same).\n"
        )
        lines.append(f"## Game rules\n{_GAME_RULES_BLURB}\n")
        lines.append(f"## Data schema\n{_DATA_SCHEMA_BLURB}\n")

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
                + _PLAYBACK_RECIPE + "\n"
            )

        if version_before is not None:
            lines.append(
                "## Your previous version\n"
                f"- version_id: {version_before.version_id}\n"
                f"- content_hash: {version_before.content_hash}\n"
                f"- edit_type: {version_before.edit_type}\n"
                "The workspace currently holds your last edit. Read the "
                "current `agent.py` to see where you left off.\n"
            )
        else:
            lines.append(
                "## First act\nThis is the initial version. Read `agent.py`, "
                "understand the current strategy, and make the first "
                "improvement.\n"
            )

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
                "## What to do now\n"
                "1. Read `agent.py`.\n"
                "2. Use the match history above to find the weakest matchup; "
                "read one of its replays if you need to see *why*.\n"
                "3. Make a targeted improvement — a new/adjusted rule, a "
                "utility/scoring function, or a bounded lookahead — "
                "whichever is the smallest change that fixes it.\n"
                "4. Leave the agent runnable (valid Python, protocol intact).\n"
                "\n"
                "## REQUIRED: a behavioral change (not a refactor)\n"
                "Your edit MUST change the action the agent takes in at least "
                "one reachable game situation — a different move direction, an "
                "attack instead of a move, an interact/escape when it would "
                "otherwise not, a reweighted candidate ranking that flips the "
                "argmax. A pure rename, reformat, helper-extraction, or comment "
                "with NO change to any emitted action is a FAILED act: the "
                "harness measures policy_kl over reference decision points and "
                "a zero-KL edit teaches nothing. If you believe the current "
                "policy is already optimal, say so explicitly and make no edit "
                "— but do not dress a no-op up as a refactor.\n"
            )
        lines.append(
            "## STAY ON MISSION — read this before acting\n"
            "Your job is to edit `agent.py`, not to debug the harness.\n"
            "- If the match history shows EVERY match as an `error` "
            "(win_rate is null / '-' across all opponents), that is a "
            "harness or environment problem — NOT a strategy problem. "
            "Do NOT try to fix the eval, the logic, or the judger. Do NOT "
            "inspect `agentbench_data/` internals.\n"
            "- You MAY read the ranked reference algorithms under "
            "`AgentBench/top_algorithms/corpus/25_lostspace_final_ladder/` "
            "for strategy and mechanics research (status enums, operation "
            "sequencing, scoring, tile types) — that is legitimate strategy "
            "research, not harness debugging. Do not copy them verbatim and "
            "do not edit anything outside `agent.py`.\n"
            "- Even with no usable eval signal, make ONE small, reasoned "
            "edit to `agent.py` based on reading the current strategy, then "
            "stop. If you genuinely believe the agent is already optimal, "
            "say so explicitly and make no edit — but do not spend your "
            "budget investigating the harness.\n"
            "- You have a strict per-act time budget. Do not exhaust it on "
            "exploration. Edit, then finish.\n"
        )
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
        behavior = (
            f"policy_kl={fmt(kl_mean, '%.4g')} — your edit changed the chosen "
            f"action on {n_changed}/{n_total} reference decision points; "
            f"occupancy_shift={fmt(occ, '%.3g')}"
        )
        parts = [
            "## Feedback on your last edit",
            f"- outcome: {outcome}",
        ]
        active = fb.get("active_opponents")
        if active:
            parts.append(f"- evaluated against: {', '.join(active)}")
        parts.append(f"- behavior: {behavior}")
        # No-op callout: the edit landed but changed zero reference decisions
        # and produced no occupancy shift. This is the exact failure mode that
        # stalled prior iterations (plausible edits in dead/dominated code).
        no_behavior_change = (
            n_total > 0 and n_changed == 0
            and (occ is None or occ == 0)
        )
        if no_behavior_change:
            parts.append(
                "- NO MEASURABLE EFFECT: your last edit changed nothing "
                "observable (0/" + f"{n_total} decisions changed, no occupancy "
                "shift). It likely lands in a code path the game never reaches, "
                "or is dominated by other logic. This act, edit the ACTIVE "
                "decision path — the branch actually taken when seat 0 is alive "
                "— or say explicitly that the agent is optimal and make no edit."
            )
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
        return (
            f"- CODE GROWTH: agent.py grew {first} -> {last} lines "
            f"(+{growth_pct:.0f}%) over the last {len(piling)} piling acts "
            f"with no consolidation. Consider a consolidation pass next "
            f"(merge overlapping branches, extract helpers, remove dead "
            f"rules) to keep the strategy from sprawling."
        )

    def _experience_section(self) -> str:
        """Render accumulated lessons (HL std 5). The agent-authored
        ``EXPERIENCE.md`` if present, else a compact view of raw staging
        observations. Empty when experience is disabled or empty."""
        if self.experience is None:
            return ""
        rendered = self.experience.render()
        if not rendered:
            return ""
        return (
            "## Lessons learned so far\n"
            f"{rendered}\n"
            f"Experience file (read & re-summarize on consolidation acts): "
            f"{self.experience.path}\n"
        )

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
        return (
            "## CONSOLIDATION ACT — compress, do NOT pile on\n"
            "This act is a consolidation pass, not a new-feature act. Do NOT "
            "add new behavior.\n"
            "1. In `agent.py`: merge overlapping/duplicate decision branches, "
            "extract shared logic into helper functions, remove dead or "
            "superseded code paths, and tighten parametrization. Keep the "
            "agent's chosen action on every reference decision point "
            "UNCHANGED (behavior-preserving refactor)." + xp_edit + "\n"
            "3. Leave the agent runnable (valid Python, protocol intact). "
            "If you find nothing to consolidate, say so explicitly and make "
            "no edit.\n"
        )

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
        out += ["", "How to read a replay:", _PLAYBACK_RECIPE]
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
        died = f"died={d['died']}" + (
            f"@round{d['died_round']}" if d["died_round"] is not None else "")
        bits = [f"keys={d['keys']}", f"escaped={d['escaped']}", died,
                f"ai_errors={d['ai_errors']}", f"rounds={d['n_rounds']}"]
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
        return ("seat 0 last game: " + " ".join(bits)
                + "  (replays are event-summarized; counts are lower bounds)")

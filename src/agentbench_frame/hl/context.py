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
from typing import Any, Dict, Optional

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

    def build(
        self,
        *,
        version_before: Optional[VersionHandle],
        act_id: str,
        **_: Any,
    ) -> Dict[str, Any]:
        return {"prompt": self._prompt(version_before, act_id),
                "timeout": self.timeout}

    # ---- internals ----

    def _prompt(self, version_before: Optional[VersionHandle],
                act_id: str) -> str:
        lines: list[str] = []
        lines.append(
            f"# HL act {act_id} — improve the LostSpace agent\n"
            "You are editing a heuristic LostSpace agent in this workspace. "
            "Your goal is to raise its win rate against the benchmark "
            f"opponents ({', '.join(self.spec.opponents)}).\n"
            "RULES OF ENGAGEMENT:\n"
            "- Edit only the agent source (`agent.py` and any helper modules "
            "in this directory). Do NOT touch `manifest.toml`.\n"
            "- Make ONE coherent improvement. Do not rewrite from scratch "
            "unless clearly broken.\n"
            "- Keep the Saiblo stdio protocol intact (read 4-byte "
            "length-prefixed JSON, send the same).\n"
        )
        lines.append(f"## Game rules\n{_GAME_RULES_BLURB}\n")

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

        lines.append(
            "## What to do now\n"
            "1. Read `agent.py`.\n"
            "2. Use the match history above to find the weakest matchup; "
            "read one of its replays if you need to see *why*.\n"
            "3. Make a targeted improvement. Prefer small, reasoned edits.\n"
            "4. Leave the agent runnable (valid Python, protocol intact).\n"
        )
        return "\n".join(lines)

    def _history_section(self) -> str:
        view = MatchHistoryView(
            data_root=self.data_root, game=self.game, agent=self.agent_name
        )
        by_opp = view.by_opponent()
        if not by_opp:
            return ""
        rows = ["opponent  W  L  err  win_rate  avg_rank"]
        for name, a in by_opp.items():
            wr = a.get("win_rate")
            wr_s = f"{wr:.0%}" if wr is not None else "-"
            ar = a.get("avg_rank")
            ar_s = f"{ar:.1f}" if ar is not None else "-"
            rows.append(
                f"{name}  {a['wins']}  {a['losses']}  {a['errors']}  "
                f"{wr_s}  {ar_s}"
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
            "",
            "How to read a replay:",
            _PLAYBACK_RECIPE,
        ]
        return "\n".join(out)

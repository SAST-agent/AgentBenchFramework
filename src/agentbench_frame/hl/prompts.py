"""All static prompt text for the HL loop, in one place.

Programs import these constants/templates instead of embedding prompt copy:
- ``cli.py`` sends ``SYSTEM_PROMPT`` as the model's system message every act.
- ``context.py`` assembles the per-act user prompt from ``DATA_SCHEMA_BLURB``,
  ``GAME_RULES_BLURB`` / ``load_rules_doc()``, ``PLAYBACK_RECIPE`` and the
  mission blocks (engagement / what-to-do / required-behavioral-change /
  measurement-location / stay-on-mission / consolidation / experience).
- ``controller.py`` uses ``VALIDATION_PROMPT`` for the rules-validation act.

Edit prompt COPY here, not in cli/context/controller. Dynamic per-act sections
(match history, feedback numbers, replay digests) are still assembled in
``context.py`` from live data — only the fixed text lives here.

NOTE: ``SYSTEM_PROMPT`` and ``DATA_SCHEMA_BLURB`` both state the interprops
data-safety rule (int/object-coded, never string-gate). They intentionally
carry the same fact in two layers (durable system role + per-act reminder);
dedupe them here if you change one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# System prompt (cli.py) — sent as the model's system message on every act.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a heuristic-learning coding agent iterating on a LostSpace "
    "game AI. Read the prompt's "
    "match history and the replay, diagnose the weakest matchup, "
    "and make an edit to agent.py via the edit tool "
    "Never touch manifest.toml. "
    "Never break the Saiblo stdio protocol (4-byte big-endian length "
    "prefix + UTF-8 JSON). Leave the agent runnable. "
    "DATA-SAFETY RULE: never gate an action you emit (interact/attack/"
    "move/use_tool) on a membership or equality check against data whose "
    "RUNTIME type you have not confirmed from the code that produces it. "
    "In LostSpace `self.view.nodes[i].interprops` is a list of INTEGER "
    "CODES / objects (1=EscapeCapsule, 2=KeyMachine; the client also "
    "appends the string 'Box'), NOT a list of strings. So a guard like "
    "`if 'KeyMachine' in interprops` is ALWAYS False and silently disables "
    "key collection. Prefer the codebase's existing blind-call-then-check "
    "pattern: call `self.interact('KeyMachine')` and branch on "
    "`result['success']` (the server returns success only when the action "
    "is actually valid)."
)

# ---------------------------------------------------------------------------
# Game rules + data schema (context.py).
# ---------------------------------------------------------------------------

# Fallback when lostspace/docs/REPLAY_SKILL.md cannot be read.
GAME_RULES_BLURB = (
    "LostSpace: 4-player FFA on a 7x7x3 grid. Win by collecting 4 corner keys "
    "(one per KeyMachine) then escaping via the center capsule. Scoring: "
    "+1 key, +2 kill, -3 death. Ranking = escape order, then survivors by "
    "score. First to escape = rank 1."
)

# The full human-written rules + replay skill doc (lostspace/docs/REPLAY_SKILL.md)
# injected verbatim so the coding agent sees the authoritative rules, not a
# 4-line blurb. Falls back to GAME_RULES_BLURB if the file is missing.
RULES_DOC_RELPATH = (
    Path(__file__).resolve().parent.parent / "lostspace" / "docs" / "REPLAY_SKILL.md"
)
_rules_doc_cache: Optional[str] = None


def load_rules_doc() -> str:
    """Return the authoritative rules doc (cached), falling back to the blurb."""
    global _rules_doc_cache
    if _rules_doc_cache is None:
        try:
            _rules_doc_cache = RULES_DOC_RELPATH.read_text(encoding="utf-8").strip()
        except OSError:
            _rules_doc_cache = GAME_RULES_BLURB
    return _rules_doc_cache


DATA_SCHEMA_BLURB = (
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
    "can never win. You START with your own key; each KeyMachine grants its "
    "key at the start of your NEXT round. To START the escape you must send "
    "`interact('EscapeCapsule', True)` — `False` only aborts an in-progress "
    "escape and is rejected while Alive (`interactive_props.py:113`)."
)

# The inline playback recipe (condensed from the SKILL.md so the coding agent
# can read replays without the skill being loaded). Kept short — just enough
# to orient, with the absolute skill path for the full version.
PLAYBACK_RECIPE = """\
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

# ---------------------------------------------------------------------------
# Per-act mission blocks (context.py). Dynamic values interpolated at build.
# ---------------------------------------------------------------------------

# Fixed rules of engagement, appended right after the dynamic act header.
RULES_OF_ENGAGEMENT = (
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
    "interpretable wrapper).\n "
    "- Always try to make the program's behaviour "
    "change every time you modify agent.py. Learn from the playback to "
    "improve. Do not be afraid of doing worse since we will keep the best draft."
    "- Keep the Saiblo stdio protocol intact (read 4-byte "
    "length-prefixed JSON, send the same).\n"
)

FIRST_ACT = (
    "This is the initial version. Read `agent.py`, "
    "understand the current strategy, and make the first "
    "improvement."
)

# Template; format with version_id / content_hash / edit_type.
PREVIOUS_VERSION = (
    "- version_id: {version_id}\n"
    "- content_hash: {content_hash}\n"
    "- edit_type: {edit_type}\n"
    "The workspace currently holds your last edit. Read the "
    "current `agent.py` to see where you left off."
)

WHAT_TO_DO_NOW = (
    "## What to do now\n"
    "1. Read `agent.py`.\n"
    "2. Use the match history above to find the weakest matchup; "
    "read one of its replays if you need to see *why*.\n"
    "3. Make a targeted improvement — a new/adjusted rule, a "
    "utility/scoring function, or a bounded lookahead — "
    "whichever is the smallest change that fixes it.\n"
    "4. Leave the agent runnable (valid Python, protocol intact)."
)

REQUIRED_BEHAVIORAL_CHANGE = (
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
    "— but do not dress a no-op up as a refactor."
)

MEASUREMENT_LOCATION = (
    "### WHERE the measurement looks — edit HERE\n"
    "The harness measures the FIRST action that `play()` sends each "
    "turn. In this agent `play()` short-circuits top-down: attack "
    "branch, then Kit-at-low-hp, then `view_box(\"Box\")` / "
    "`interact(\"KeyMachine\")`, and only falls through to "
    "`test_move()` when all of those would fail. So edits INSIDE "
    "`test_move()` (bfs weights, scoring) change the move TARGET but "
    "NOT the first emitted action — they register as policy_kl=0. "
    "To register, edit the TOP of `play()`: which branch fires "
    "first (attack threshold, the Kit-use threshold, whether to "
    "interact a prop vs move now). A flip on any 'reference "
    "first-actions' row in the feedback is a valid update."
)

STAY_ON_MISSION = (
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
    "exploration. Edit, then finish."
)

# Template; format with experience_edit (the re-summarize instruction, or "").
CONSOLIDATION_MISSION = (
    "## CONSOLIDATION ACT — compress, do NOT pile on\n"
    "This act is a consolidation pass, not a new-feature act. Do NOT "
    "add new behavior.\n"
    "1. In `agent.py`: merge overlapping/duplicate decision branches, "
    "extract shared logic into helper functions, remove dead or "
    "superseded code paths, and tighten parametrization. Keep the "
    "agent's chosen action on every reference decision point "
    "UNCHANGED (behavior-preserving refactor)."
    "{experience_edit}\n"
    "3. Leave the agent runnable (valid Python, protocol intact). "
    "If you find nothing to consolidate, say so explicitly and make "
    "no edit."
)

# Template; format with rendered / path.
EXPERIENCE_SECTION = (
    "## Lessons learned so far\n"
    "{rendered}\n"
    "Experience file (read & re-summarize on consolidation acts): "
    "{path}\n"
)

# ---------------------------------------------------------------------------
# Feedback fragments (context.py) — static sentences around live numbers.
# ---------------------------------------------------------------------------

FEEDBACK_BEHAVIOR_LINE = (
    "policy_kl={kl_mean} — your edit changed the chosen "
    "action on {n_changed}/{n_total} reference decision points; "
    "occupancy_shift={occupancy}"
)

ACTION_FREQ_INTRO = (
    "- action-frequency KL={action_kl} — your "
    "full-match action mix vs last version (counts across all "
    "matches):"
)

BEST_VERSION_LINE = (
    "- best version so far: {best_id} (win_rate={wr}, avg_rank={ar}). "
    "You can call `revert_to_best` to discard your current edits and restore "
    "this code, then try a different direction next act. Use it when your last "
    "edit regressed (win rate dropped below this). revert_to_best ENDS the act."
)

REVERTED_NOTE = (
    "- you reverted to the best version {best_id} (win_rate={wr}, avg_rank={ar}) "
    "last act. Your workspace is now that code — edit from here."
)


REF_POINTS_CTA = (
    "  To register a policy update, change the FIRST action on "
    "at least one of these points this act (e.g. a different "
    "move direction, attack instead of heal, stop calling "
    "interact('Box') when the tile has no Box). The harness "
    "measures the first action the agent sends at each point."
)

NO_MEASURABLE_EFFECT = (
    "- NO MEASURABLE EFFECT: your last edit changed nothing "
    "observable (0/{n_total} decisions changed, no occupancy "
    "shift). It likely lands in a code path the game never reaches, "
    "or is dominated by other logic. This act, edit the ACTIVE "
    "decision path — the branch actually taken when seat 0 is alive "
    "— or say explicitly that the agent is optimal and make no edit."
)

# Template; format with first / last / pct / n_piling.
CODE_GROWTH = (
    "- CODE GROWTH: agent.py grew {first} -> {last} lines "
    "(+{pct:.0f}%) over the last {n_piling} piling acts "
    "with no consolidation. Consider a consolidation pass next "
    "(merge overlapping branches, extract helpers, remove dead "
    "rules) to keep the strategy from sprawling."
)

# ---------------------------------------------------------------------------
# Rules-validation act (controller.py).
# ---------------------------------------------------------------------------

# The rules_validation act prompt. The agent's deliverable is its FINAL
# MESSAGE in the exact report format below — the harness parses the
# ``SCORE_DIC`` line and cross-checks it against the replay's real ``r[-1]``.
VALIDATION_PROMPT = """\
# REPLAY_SKILL validation act — prove you read the rules and replay format correctly

This is NOT an edit act. Do NOT call `edit`. Do NOT modify any file.

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

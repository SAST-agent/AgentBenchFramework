# Design: `revert_to_best` agent-callable tool

## Problem

The HL loop edits forward from the previous act's version (`cli.py:569`
`version = ctrl.act(version_before=version)`). Once an edit regresses the
agent, every later act edits the *broken* code — there is no path back to a
known-good version. Observed in `hl-v26-08-07-round5`: act1 seed v12 = wr 0.7;
act2 edit → wr 0.0; acts 3–4 kept editing forward from the broken act2 code,
thrashing (policy_kl ig growing 0.77 → 1.54), never recovering the v12
strategy. The 0/20 win-rate cliff gives zero gradient to climb back.

## Goal

Let the coding agent **decide** to abandon a regressed line and restart the
next edit from the best-scoring version so far. Model-initiated, not
auto-reverted (avoids noise-driven reverts on near-ties; the agent has the
riches context — replays + metrics — to judge).

## Non-goals

- No automatic revert policy (no margin / patience / strict harness trigger).
- No new CLI flag for revert behavior; best metric is fixed.
- ApiCodingRunner only (deepseek/gpt — the sole live path; `cli.py:531` hard
  wires it, `ClaudeCodeRunner` is dormant/docstring-only). Wiring the claude
  CLI path is a follow-up, out of scope.
- Best metric is fixed at win_rate (primary, higher better) with avg_rank
  tiebreak (lower better). No configurability.

## Key design constraint (why revert is NOT a mid-loop tool)

The ApiCodingRunner tool loop reads `agent.py` into the message history, then
the agent edits by exact-match `old_string`. A mid-act
`read_file`(broken) → `revert_to_best`(workspace rewritten to best code) →
`edit` sequence fails: the `old_string` comes from the **stale broken-code
read**, which no longer matches the reverted best-code file →
`old_string_not_found`. Revert invalidates the agent's own read context.

**Therefore `revert_to_best` is a TERMINAL tool, symmetric with `edit`.**
Read/replay tools are the non-terminal preamble; the agent then ends the act
with EITHER `edit` (try an improvement) OR `revert_to_best` (abandon, reset to
best). Mutually exclusive per act. After a revert-act, the NEXT act reads the
best code fresh and edits from there.

## Mechanism

### 1. Best tracking — `controller.py`

New controller state:
```python
self._best: Optional[Dict] = None
# {"version_id", "content_hash", "win_rate", "avg_rank"}
```

After each act's eval (win_rate already at `controller.py:289`, avg_rank at
`:294`), when `eval_status == "complete"` and `version_after is not None`:
compute the metric and update best on **strict** improvement:
- better if `win_rate > best.win_rate`
- or `win_rate == best.win_rate and avg_rank < best.avg_rank`
- (None avg_rank treated as worst, so a version with a real avg_rank beats one
  without at equal win_rate.)

Best is initialized from act 1's eval (the seed). Before act 1 evals,
`self._best is None` and the revert tool reports "no best yet".

### 2. Tool schema — `llm.py` `SHARED_TOOLS`

Append:
```python
{
  "name": "revert_to_best",
  "description": "Restore agent.py to the best-scoring version seen so far in "
                 "this round, discarding your current (uncommitted) edits. "
                 "Use this when your recent edits regressed — e.g. win rate "
                 "dropped — and you want to restart from the strongest known "
                 "code before trying a different direction. This ENDS the act: "
                 "do not call edit in the same act. Returns the best version's "
                 "win_rate and avg_rank, or 'no best version yet'.",
  "input_schema": {"type": "object", "properties": {}},
}
```

### 3. Runner dispatch — `runner.py` `ApiCodingRunner.run`

In the per-tool-call loop (`runner.py:477`), alongside the `edit` branch, add a
terminal `revert_to_best` branch:
- Call `context["revert_fn"]()` (callback supplied by the controller, bound to
  the live codebase + current best). Returns
  `{"ok": True, "best_version_id", "win_rate", "avg_rank"}` or
  `{"ok": False, "reason": "no best yet"}`.
- Record the result in the turn transcript (`tool_results`).
- On `ok: True`: set a `reverted` flag and **break** the turn loop (terminal,
  like `edit`). On `ok: False` (no best yet): do NOT break — leave the result
  in the transcript and let the loop continue so the agent can still `edit`
  this act (a no-best revert must not waste the act).

`AgentRunResult` gains a `reverted: bool = False` field and
`reverted_to: Optional[str] = None` (best version_id). On a revert,
`edit_type="rollback"`, `files_touched=["agent.py"]`, no failure_reason.

The callback contract keeps the runner dumb: it does not touch the codebase or
version store directly; the controller owns restore + lineage.

### 4. Revert execution + lineage — `controller.py`

The controller builds each act's `context` with:
- `context["best_version"]` = `self._best` (or None) — for the prompt.
- `context["revert_fn"]` = a closure that, when called:
  1. Requires `self._best is not None` (else returns `ok: False, "no best yet"`).
  2. Calls `self.codebase.restore(self._best["content_hash"],
     parent_version_id=self._best["version_id"])` — `restore()` already
     exists (`codebase.py:127`), wipes + repopulates the workspace, returns a
     `VersionHandle(edit_type='rollback')`. The rollback's parent is the **best
     version** (code-derivation truth), not the prior act — the temporal
     act-chain is already captured by `act_id` ordering + the `agent_act`
     event's `version_before` field.
  3. Stashes the rollback handle on `self._revert_handle_this_act` so the
     post-run snapshot step uses it.
  4. Returns `{"ok": True, "best_version_id": ..., "win_rate": ...,
     "avg_rank": ...}`.

Post-run snapshot (`controller.py:233`): if the runner reported `reverted`,
the act's `version_after` is the rollback handle (content_hash ==
best.content_hash). Emit the `version` event with `edit_type='rollback'`. The
eval step recognizes `content_hash == best.content_hash` and **skips
re-evaluation** (result already known: best's metrics) — emits the `eval`
event with best's win_rate, no new matches run.

Lineage: the rollback version's `parent_version_id` = best.version_id
(code derives from best). The NEXT act's `version_before` = the rollback
handle, so the next edit snapshots with `parent_version_id` = rollback version
→ the forward walk resumes from best code. (This is the fix for the round5
thrash: act3 would start from best v12 code, not broken act2 code.)

### 5. Revert event — `events.py` + `controller.py`

New event type `revert`:
```
{act_id, from_version_id (the act's version_before), to_version_id (best),
 to_content_hash, win_rate, avg_rank}
```
Emitted once per revert-act, after the version event. Lets the research stream
show exactly when the model chose to revert and to what.

### 6. Prompt surface — `context.py` / `prompts.py`

The per-act context already reports the last edit's metrics. Add a short
"best version" line to the feedback section when `best_version` is present:
```
Best version so far: <id> (win_rate <wr>, avg_rank <ar>).
Your last edit evaluated at win_rate <wr2>, avg_rank <ar2>.
If your last edit regressed, call revert_to_best to restore the best code,
then edit from there next act. revert_to_best ENDS the act.
```
This is what makes the tool discoverable + gives the model the comparison it
needs to decide.

## Tests (TDD — written first)

`tests/hl/test_revert.py` (+ additions to `test_controller.py`):

1. **best tracking — update on strict improvement:**
   act1 eval wr=0.7 ar=2.0 → best=act1. act2 wr=0.0 → best stays act1.
   act3 wr=0.8 ar=1.5 → best=act3.
2. **best tracking — avg_rank tiebreak:** wr equal (0.7), ar 2.0 vs 1.5 →
   lower ar wins.
3. **revert tool restores workspace:** agent calls revert_to_best → workspace
   agent.py byte-equals best snapshot; tool returns best metrics.
4. **revert is terminal:** after revert_to_best, the run loop ends; no `edit`
   applied that act.
5. **revert with no best:** before act1 eval, revert_to_best returns
   `{ok: False, "no best yet"}`, workspace unchanged, and the loop is NOT
   terminated — the agent can still call `edit` in the same act.
6. **context injection:** `best_version` present in context iff best exists.
7. **lineage:** post-revert, next act's snapshot parent = rollback version;
   rollback content_hash == best.content_hash.
8. **eval skip on rollback:** revert-act's eval reuses best metrics (no new
   matches run — assert evaluator_factory not called that act).
9. **revert event emitted** with correct from/to/metrics.

Existing `test_controller.py` act-loop tests must stay green (act() return
contract unchanged; new behavior is additive).

## Files touched

- `src/agentbench_frame/hl/llm.py` — `revert_to_best` in SHARED_TOOLS.
- `src/agentbench_frame/hl/runner.py` — terminal revert branch; `AgentRunResult.reverted`/`reverted_to`.
- `src/agentbench_frame/hl/controller.py` — best tracking, `revert_fn` closure, rollback snapshot/eval-skip, revert event.
- `src/agentbench_frame/hl/events.py` — `revert` event type (if events need registration).
- `src/agentbench_frame/hl/context.py` and/or `prompts.py` — best-version feedback line.
- `tests/hl/test_revert.py` — new; `tests/hl/test_controller.py` — additions.

## Risks / open details (plan-level)

- `events.py` may need the `revert` type allow-listed wherever event types are
  validated/summarized (plot_curves, report). Verify during plan.
- `round_state.py` may already track per-round best-like state — check for
  reuse before adding `_best`.
- `_resolve_edit_type` interaction with `edit_type="rollback"` returned from
  the runner — ensure the controller trusts the runner's explicit rollback
  label rather than re-diff-classifying.

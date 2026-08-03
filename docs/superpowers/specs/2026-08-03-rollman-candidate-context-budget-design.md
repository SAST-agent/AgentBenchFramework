# Rollman Candidate Context and Smoke Design

## Objective

Enable at least three of four Rollman coding candidates per proposal cycle to reach compilation and end-to-end smoke verification within the existing 70,000 weighted-token budget, without changing K=4, `xhigh` reasoning, the linear-search lineage, frozen evaluation seeds, rollback rules, or certification gates.

## Constraints

- Preserve one reproducible run and its immutable version lineage.
- Keep one coding-agent act per candidate branch.
- Keep the provider-native 70,000 weighted-token limit and 420-second hard timeout.
- Preserve the candidate activation gate and fail-closed evaluation behavior.
- Do not expose human opponent source, hidden evaluation data, certification seeds, or secrets.
- Do not introduce opponent identities, fixed board coordinates, or seed-specific policy logic.

## Selected Architecture

The planner selects exact policy symbols for every branch. The Framework resolves those symbols into bounded source slices and embeds them in the candidate packet. The candidate receives a reusable valid Rollman smoke fixture and one exact verification contract. The candidate edits from this bounded packet and may perform at most one additional source read before the first patch.

This architecture keeps diagnosis and implementation in one act, avoids a second model call, and retains the native rollout-budget accounting contract.

## Planner Contract

Each branch brief contains a required `code_symbols` array:

```json
{
  "code_symbols": [
    "ai_func",
    "_update_phase_memory",
    "_pressure_arbitration_action"
  ]
}
```

Rules:

- The array contains 2–8 unique module-level function names.
- `ai_func` is always required.
- Every value must exist in the planner's `candidate_code_index`.
- The planner selects only symbols required to implement or preserve the branch mechanism.
- Invalid or unknown symbols fail planner-output validation.

The planner input embeds the compact `candidate_code_index`, including function names, signatures, and line ranges. It does not embed full policy source.

## Candidate Packet

The candidate packet adds `candidate_code_slices`:

```json
{
  "candidate_code_slices": [
    {
      "name": "ai_func",
      "signature": "ai_func(game_state)",
      "start_line": 1747,
      "end_line": 1931,
      "source": "def ai_func(game_state): ..."
    }
  ]
}
```

Resolution rules:

- Source is parsed from the exact candidate workspace snapshot used for the branch.
- Every slice is explicitly marked `complete` or `truncated`; an unmarked partial body is forbidden.
- The total embedded source is capped at 24,000 characters.
- If selected functions exceed the cap, `ai_func` remains complete and remaining functions are represented by signature plus a bounded head/tail excerpt. The candidate may use its single additional source read for one truncated function.
- Slice ordering follows `code_symbols` ordering.
- The packet retains the compact game digest, Experience Skill, research state, replay evidence, measurements, and opponent distillation.

## Reusable Smoke Fixture

The Framework provides a read-only `rollman_smoke_fixture.py` in the run context and exposes its exact path and invocation in the candidate packet. The module is both a state-fixture library and an executable verifier.

The fixture provides:

```python
make_state(
    *,
    level: int,
    round_id: int,
    pacman: tuple[int, int],
    ghosts: list[tuple[int, int]],
    score: tuple[int, int] = (0, 0),
    skills: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    portal_available: bool = False,
    beannumber: int = 20,
    board_size: int = 22,
)
```

Fixture invariants:

- Interior cells use a valid walkable tile value.
- Boundary cells use the frozen wall value.
- Pacman and Ghost coordinates are walkable.
- Board, coordinates, scores, and skills use frozen SDK-compatible NumPy types.
- The object implements `gamestate_to_statedict()`.
- Optional helpers construct sequential states for finite-memory mechanisms.

The fixture prevents invalid all-wall boards and repeated hand-written `GameState` adapters.

The candidate writes `.agentbench/smoke_scenario.json` rather than a custom Python smoke script. The scenario contains an ordered list of fixture arguments and two assertions:

- `activation_memory_prefix`: required on the final activation state;
- `preservation_forbidden_prefix`: forbidden on a preservation state.

The verifier imports the workspace `ai.py`, builds all states, calls public `ai_func` in order, validates actions and memory prefixes, and writes `.agentbench/candidate_smoke_result.json`. The Framework reruns the same verifier after the provider act, so a manually fabricated result file cannot satisfy partial-adoption eligibility.

## Candidate Verification Contract

The candidate performs this order:

1. Read the candidate packet once.
2. Read at most one additional source range when a selected slice is truncated.
3. Inspect at most two authorized replay windows.
4. Write the first `ai.py` patch and `experience_update.json`.
5. Run `python -m py_compile ai.py`.
6. Run the packet's exact public-entry smoke command using `rollman_smoke_fixture.py` and `.agentbench/smoke_scenario.json`.

The smoke must:

- call public `ai_func`, never an internal helper as the success assertion;
- satisfy the branch `activation_condition`;
- return a valid action;
- return a `memory_id` with the candidate's new mechanism prefix;
- include at least one preservation state whose result retains the parent mechanism prefix;
- use sequential states when the mechanism depends on memory.

Failure to compile or satisfy both activation and preservation assertions makes the candidate ineligible for partial adoption, even when provider budget exhaustion occurs.

## Prompt and Budget Guardrails

- The packet is the only required context read.
- Re-running symbol search, file discovery, full-file printing, or packet filtering is prohibited.
- Candidate-packet acts enforce at most six pre-edit tool calls and twelve total tool calls through the provider's JSONL tool-limit guard.
- The provider-native 20,000, 10,000, and 5,000 weighted-token reminders remain enabled. The prompt requires the first patch by tool call five and reserves the final calls for compile, smoke, and final response.
- Candidate acceptance after budget exhaustion requires a changed snapshot, successful compilation, successful public-entry activation smoke, clean access audit, and successful Framework activation probe.

These checks supplement the frozen match quick screen; they do not replace it.

## Data Flow

1. Framework builds the policy code index.
2. Planner receives the index and emits four branch briefs with exact `code_symbols`.
3. Framework validates all symbols and resolves bounded source slices.
4. Framework writes one candidate packet containing evidence, experience, code slices, and smoke-fixture metadata.
5. Coding agent patches one isolated candidate workspace.
6. Provider records tool calls, budget state, and file-change progress.
7. Framework compiles, probes activation, runs quick-screen matches, and applies the existing selection and rollback rules.
8. Reducer writes replay-grounded outcomes to research state and Experience Skill.

## Error Handling

- Unknown planner symbol: reject planner output before any candidate act.
- Syntax-invalid selected source: fail packet construction before a billable candidate act.
- Source-slice overflow: preserve `ai_func`, truncate only non-entry helpers, and record truncation metadata.
- Missing smoke fixture: fail candidate-context preflight.
- Invalid smoke scenario or a result not reproducible by the Framework: reject candidate before matches.
- Candidate smoke does not reach the new memory prefix: reject candidate before matches.
- Candidate smoke changes a preservation state: reject candidate before matches.
- Provider ends before compile or smoke: reject partial adoption.
- Framework activation probe reports zero changes or execution failure: reject candidate before matches.

## Verification

Automated tests cover:

- strict planner schema for `code_symbols`;
- rejection of duplicates, unknown symbols, missing `ai_func`, and arrays outside 2–8 entries;
- deterministic complete source-slice extraction;
- source-size cap and truncation metadata;
- valid SDK-compatible smoke states and sequential-state helpers;
- deterministic smoke-scenario execution and Framework rerun;
- rejection of forged, stale, or mismatched smoke-result artifacts;
- candidate prompt ordering and one-extra-read contract;
- rejection of partial candidates without compile and public-entry smoke evidence;
- preservation of activation-gate, evaluation, selection, rollback, certification, and reporting behavior;
- full repository regression suite.

## Experimental Acceptance Criteria

Run one K=4 proposal cycle under the frozen 70,000-token configuration and require:

- at least three candidates create a changed `ai.py` snapshot;
- at least three candidates compile successfully;
- at least three candidates complete public-entry activation and preservation smoke;
- no candidate uses an invalid board fixture;
- no candidate performs full-file policy reads;
- provider budget exhaustion occurs for at most one candidate;
- Framework activation and match evaluation remain fail-closed;
- the lineage head, champion, certification seeds, and stop criterion remain unchanged unless game results justify promotion.

Failure of these harness criteria blocks another paid proposal cycle until the context or verification contract is corrected.

# Miracle OpenAI-Compatible Harness Loop Design

## 1. Goal

Build the smallest reproducible high-level iteration loop for the 24th Miracle game. A tested LLM receives the current strategy, the game-specific Skills, and selected replay evidence through an OpenAI-compatible `POST /v1/chat/completions` API. It returns a complete replacement strategy. The Harness owns validation, versioning, battles, replay storage, strict KL status calculation, score aggregation, budgets, and result export.

The first implementation deliberately excludes Tool Calling. Once the complete-source loop is verified end to end, the LLM transport can be extended with tools without changing battle, replay, metrics, or result schemas.

## 2. Scope

The implementation must complete the original five requirements:

1. Run one traceable iteration from official game logic through battle, replay, LLM strategy modification, version save, and reevaluation.
2. Supply and validate the human-authored, Agent-polished Miracle Harness and replay-reading Skills.
3. Define the Miracle observation, parameterized macro-actions, action mask, termination conditions, and finite action support used by IG.
4. Save strict KL status data per decision, episode, and iteration, including explicit missing reasons.
5. Produce version-aligned score–iteration and IG–iteration data with at least one valid strategy update; preserve failures and incomplete evaluations.

The first implementation will also write the accounting fields needed later for budget curves and leaderboards. It will not run the population, environment, reward, or policy ablation studies yet.

## 3. LLM Boundary

### Request

The Harness calls an OpenAI-compatible endpoint:

```text
POST {base_url}/v1/chat/completions
```

Configuration contains:

- `base_url`
- `api_key_env`, naming the environment variable that holds the secret
- `model`
- `temperature`
- `max_tokens`
- request timeout

Secrets are never written to logs. The complete request messages, with secrets excluded, are saved for audit.

The request context contains only Harness-selected inputs:

1. system contract and output schema;
2. `skills/miracle-harness/SKILL.md`;
3. `skills/miracle-replay-reader/SKILL.md`;
4. current strategy source;
5. structured battle summary;
6. replay/trace excerpts selected within the episode and decision-read budget;
7. previous iteration metrics and failure information;
8. remaining budget.

### Response

The assistant must return one JSON object:

```json
{
  "analysis": "Concise explanation of the observed weakness and intended change.",
  "strategy_code": "Complete Python source defining CandidateAgent."
}
```

Markdown fences are not part of the contract. The Harness may accept a single fenced JSON object as a defensive compatibility fallback, but records that normalization in the iteration log.

The generated source must define `CandidateAgent`, a subclass of `MiracleAgent`, with `choose_cards(camp)` and `act(obs)`. It is loaded from the saved iteration snapshot, not added to the static `AGENTS` registry. This avoids accumulating `strategy_v2`-style historical interfaces while still preserving immutable versions for evaluation and rollback experiments.

## 4. Run, Iteration, Episode, and Round

- A **Run** is one complete LLM benchmark session.
- An **iteration** is one LLM strategy-update attempt, including failures.
- An **episode** is one complete battle.
- A **round** is one in-game Miracle turn number.

Iteration 0 is the immutable raw baseline. Iteration N loads the accepted strategy from iteration N-1, evaluates the configured evidence episodes, calls the LLM once, validates and saves the candidate, then evaluates it. A failed update still creates an iteration record and does not silently advance the accepted strategy.

## 5. Loop

For a minimal Run:

1. Create the Run directory and save configuration, git revision, Skills, initial strategy, start time, and budget limits.
2. Evaluate iteration 0 with the official Miracle logic. Save every `.mrc`, `.trace.jsonl`, stdout-equivalent result, seed, seat, opponent, duration, and error.
3. Build the LLM context from the current strategy and selected evidence.
4. Call `/v1/chat/completions`; save sanitized request, raw response, status, latency, and returned usage.
5. Parse `strategy_code`, save it before validation, import it in isolation, instantiate `CandidateAgent`, and run interface smoke validation.
6. If valid, save it as the immutable candidate for the new iteration. If invalid, record the failure and keep the prior accepted strategy.
7. Evaluate the candidate using the same configured opponent, seeds, and seats required for version alignment.
8. Compare old and new agents on the same recorded observations and save strict KL statuses.
9. Aggregate the iteration score, gain, budgets, and IG statuses.
10. Update Run-level curves and CI-compatible result files atomically.

The minimal acceptance Run may use one opponent, one seed, and one seat to prove the pipeline. The configuration supports multiple seeds and seat swapping for later statistically meaningful experiments.

## 6. Validation and Failure Semantics

Validation is staged:

1. response received;
2. response JSON parsed;
3. complete source extracted;
4. source imports without exception;
5. `CandidateAgent` exists and implements the interface;
6. card selection and action shape smoke checks pass;
7. official battle completes or reports its real termination state.

Every stage has `pending`, `passed`, `failed`, or `missing` status and an explicit reason. API failure, malformed JSON, invalid source, import failure, illegal action, timeout, host error, replay absence, and incomplete IG are distinct conditions. Failed and regressed iterations remain in the Run and its curves.

Generated code is untrusted. Strategy import and execution use the existing decision timeout and must later be placed behind a stronger process sandbox. The initial local implementation documents this limitation and never exposes API secrets to strategy code or prompts.

## 7. Decision Space and Strict KL Status

The existing Miracle decision-space implementation remains the single source for parameterized action support:

- observation: official parsed map, players, camp, and round;
- macro-actions: `summon`, `move`, `attack`, `use`, `endround`, `surrender`;
- action mask: finite parameterized legal-support enumeration for an observation;
- termination: official game end, round cap, surrender, timeout, host failure, or logic failure;
- IG support: the same finite action support, with documented treatment of the official unbounded `WindBlessing` coordinate behavior.

For deterministic policies, the Harness records:

- `unchanged`: identical supported action; strict KL is `0`;
- `infinite`: different supported action; strict KL diverges;
- `missing`: an action cannot be parsed, aligned, or found in the finite support, with a reason;
- `finite_kl_mean`: mean of genuine finite values only, otherwise `null`.

No action distance, score gain, or epsilon-smoothed proxy is labeled as KL.

## 8. Metrics

Per episode:

- result, winner, official scores, rounds, termination, errors;
- decision count and replay/trace paths;
- wall time;
- strict KL counts and ratios where applicable.

Per iteration:

- `raw`: the fixed iteration-0 baseline score;
- `evo`: the current accepted candidate score under the aligned evaluation set;
- `gain = evo - raw`;
- win rate and completion rate;
- `unchanged_ratio`, `infinite_ratio`, `missing_ratio`, and genuine `finite_kl_mean`;
- episode-read count, decision/step-read count, rollout count;
- prompt, completion, and total tokens when supplied by the API;
- API time, battle time, and total wall time.

Run-level output contains score–iteration and IG–iteration curves. AUC is computed over explicit iteration, rollout, token, episode-read, and wall-time axes only when at least two measured points exist; otherwise it is `null` with a reason.

## 9. Storage and AgentBenchResults Export

The Run is written directly beneath `$AGENTBENCH_DATA` when set, otherwise beneath local `agentbench_data`. Its standard destination is:

```text
runs/24_miracle/{agent}/{run_id}/
├── run.toml
├── summary.json
├── events.jsonl
├── score_curve.json
├── ig_curve.json
├── iterations/
│   ├── iteration-0000/
│   │   ├── strategy.py
│   │   ├── iteration.json
│   │   └── episodes/...
│   └── iteration-0001/
│       ├── llm_request.json
│       ├── llm_response.json
│       ├── candidate.py
│       ├── strategy.py
│       ├── iteration.json
│       ├── episodes/...
│       └── ig/...
└── skills/
    ├── miracle-harness.SKILL.md
    └── miracle-replay-reader.SKILL.md
```

`run.toml` and `summary.json` follow the existing AgentBenchResults contract. Miracle-specific curves and budgets are additional fields/files. Writes use temporary files followed by replacement so interruption does not leave a valid-looking partial summary. `events.jsonl` is append-only and is sufficient to reconstruct the sequence of iteration states.

## 10. CLI

Existing low-level commands remain:

```text
miracle match
miracle replay
miracle ig
```

One high-level command is added:

```text
miracle loop --config PATH
```

The configuration supplies LLM endpoint/model settings, initial strategy, opponent, seeds, seats, maximum iterations, rollout limits, episode/decision-read limits, token limit, and wall-time limit. There are no duplicate `iterate`, `strategy_v2`, or version-management entry points.

## 11. Future Tool Calling Migration

The LLM client exposes one internal operation: `propose_strategy(context) -> proposal`. The first transport implements a single Chat Completions request returning complete source. A later Tool Calling transport may implement repeated model/tool turns, but it must return the same final proposal and emit the same accounting events.

Battle execution, replay parsing, strategy snapshots, metrics, budgets, curves, and result export remain unchanged. Tool calls add explicit counters and logs so Tool Calling results are comparable rather than silently receiving extra work.

## 12. Acceptance Criteria

The implementation is accepted when one local mock OpenAI-compatible server completes a deterministic Run that:

1. evaluates an initial weak strategy with official logic;
2. sends the expected Skill, source, replay evidence, and budget context;
3. receives and saves a valid improved complete strategy;
4. reevaluates it and preserves replay/trace artifacts;
5. saves aligned score and strict KL status curves;
6. records API usage, time, episode/decision reads, and rollout counts;
7. writes valid `run.toml` and `summary.json` beneath an AgentBenchResults-compatible directory;
8. preserves a deliberately malformed-response iteration as failed without corrupting the accepted strategy or Run summary.

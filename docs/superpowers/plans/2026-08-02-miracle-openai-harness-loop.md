# Miracle OpenAI Harness Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one `miracle loop --config PATH` command that calls an OpenAI-compatible Chat Completions API for complete strategy source, runs an official-logic baseline and evolved evaluation, saves every intermediate artifact and budget event, computes aligned score and strict-KL-status curves, and emits an AgentBenchResults-compatible Run.

**Architecture:** Keep the existing `match`, `replay`, `decision_space`, and `ig` modules as low-level primitives. Add focused modules for configuration, Chat Completions transport, source snapshots, Run storage/metrics, and orchestration; the loop depends on interfaces rather than API details so a later Tool Calling transport only replaces `propose_strategy()`. Use only the Python standard library and the existing project code.

**Tech Stack:** Python 3.11, `urllib.request`, `tomllib`, `json`, `importlib`, existing official Miracle logic, pytest.

## Global Constraints

- The initial transport is OpenAI-compatible `POST {base_url}/v1/chat/completions`; Tool Calling is excluded.
- The model returns one JSON object with `analysis` and complete `strategy_code` defining `CandidateAgent`.
- API secrets are read through `api_key_env` and never persisted in requests, events, summaries, or errors.
- Historical strategies are immutable files under the Run; do not add versioned classes to `AGENTS`.
- Preserve all failed, missing, incomplete, timed-out, and regressed iterations.
- Deterministic IG uses only `unchanged`, `infinite`, and `missing`; never label a proxy as finite KL.
- Existing low-level commands remain `match`, `replay`, and `ig`; add exactly one high-level command, `loop`.
- Writes that make a result look complete use a temporary sibling file followed by `Path.replace()`.
- Do not add runtime dependencies.

---

## File Structure

- Create `src/agentbench_frame/miracle/loop_config.py`: validated TOML configuration and budget definitions.
- Create `src/agentbench_frame/miracle/llm_client.py`: Chat Completions request/response transport and proposal parsing.
- Create `src/agentbench_frame/miracle/strategy_loader.py`: immutable source snapshots and `CandidateAgent` loading.
- Create `src/agentbench_frame/miracle/run_store.py`: Run directories, append-only events, atomic JSON/TOML writes, budget counters.
- Create `src/agentbench_frame/miracle/score.py`: aligned episode scoring, gain, and trapezoidal AUC.
- Create `src/agentbench_frame/miracle/loop.py`: orchestration only; dependencies are injectable for tests.
- Modify `src/agentbench_frame/miracle/cli.py`: add the single `loop` subcommand.
- Modify `skills/miracle-harness/SKILL.md`: document configuration, loop execution, artifacts, and failure semantics.
- Create tests mirroring each new module plus one official-logic end-to-end test.

---

### Task 1: Validated Loop Configuration

**Files:**
- Create: `src/agentbench_frame/miracle/loop_config.py`
- Create: `tests/miracle/test_loop_config.py`

**Interfaces:**
- Produces: `LoopConfig.from_toml(path: Path) -> LoopConfig`
- Produces: `LoopConfig.public_dict() -> dict`, excluding any resolved API secret
- Produces dataclasses `LLMConfig`, `EvaluationConfig`, and `BudgetConfig`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_loads_minimal_loop_config(tmp_path):
    path = tmp_path / "loop.toml"
    path.write_text('''
agent = "tested_llm"
initial_strategy = "initial.py"
opponent = "sample"

[llm]
base_url = "http://127.0.0.1:8123"
api_key_env = "TEST_LLM_KEY"
model = "mock-model"

[evaluation]
seeds = [11]
seats = [0]

[budget]
max_iterations = 1
max_rollouts = 4
max_episode_reads = 1
max_decision_reads = 200
max_total_tokens = 10000
max_wall_seconds = 600
''')
    cfg = LoopConfig.from_toml(path)
    assert cfg.agent == "tested_llm"
    assert cfg.evaluation.seeds == (11,)
    assert cfg.budget.max_iterations == 1

def test_rejects_unknown_opponent_and_nonpositive_budget(tmp_path):
    path = write_config(tmp_path, opponent="not_registered", max_iterations=0)
    with pytest.raises(ValueError, match="opponent|max_iterations"):
        LoopConfig.from_toml(path)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_loop_config.py -q`

Expected: collection fails because `loop_config` does not exist.

- [ ] **Step 3: Implement frozen dataclasses and strict TOML parsing**

Implement exact defaults: `temperature=0.0`, `max_tokens=8192`, `timeout_seconds=120`, `seeds=(11,)`, `seats=(0, 1)`. Resolve `initial_strategy` relative to the config file. Reject empty agent/model, unknown `AGENTS` opponent, seats outside `{0,1}`, empty seeds/seats, and nonpositive budget values. `public_dict()` contains the configured environment-variable name, never its value.

- [ ] **Step 4: Run configuration tests and full retained tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_loop_config.py tests/miracle/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/miracle/loop_config.py tests/miracle/test_loop_config.py
git commit -m "feat(miracle): add validated loop configuration"
```

---

### Task 2: OpenAI-Compatible Complete-Source Client

**Files:**
- Create: `src/agentbench_frame/miracle/llm_client.py`
- Create: `tests/miracle/test_llm_client.py`

**Interfaces:**
- Consumes: `LLMConfig`
- Produces: `StrategyProposal(analysis: str, strategy_code: str, usage: dict, request_body: dict, raw_response: dict, latency_seconds: float, normalized_fence: bool)`
- Produces: `ChatCompletionsClient.propose_strategy(messages: list[dict]) -> StrategyProposal`
- Raises: `LLMRequestError(stage: str, reason: str, raw_response: dict | str | None = None)` with stages `request`, `response_json`, `assistant_content`, and `proposal_json`; this carries a sanitized response for failure preservation

- [ ] **Step 1: Write a local HTTP-server contract test**

Use `http.server.ThreadingHTTPServer` in a pytest fixture. Capture headers/body and return:

```json
{
  "choices": [{"message": {"content": "{\"analysis\":\"improve\",\"strategy_code\":\"class CandidateAgent: pass\"}"}}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

Assert the request path is `/v1/chat/completions`, `model`, `messages`, `temperature`, and `max_tokens` are present, the bearer token comes from the configured environment variable, and the returned proposal preserves usage. Add tests for a single fenced JSON fallback and malformed content raising `LLMRequestError(stage="proposal_json", ...)`.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_llm_client.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the standard-library client**

Join `base_url.rstrip("/") + "/v1/chat/completions"`; send UTF-8 JSON with `Content-Type: application/json` and optional `Authorization: Bearer ...`. Parse `choices[0].message.content`. Accept only a JSON object whose `analysis` and `strategy_code` are nonempty strings. Convert missing usage numbers to zero. Error text may contain HTTP status and response excerpt but never request headers or the API key.

- [ ] **Step 4: Run client tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_llm_client.py -q`

Expected: PASS, including secret-redaction assertion.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/miracle/llm_client.py tests/miracle/test_llm_client.py
git commit -m "feat(miracle): add Chat Completions strategy client"
```

---

### Task 3: Immutable Strategy Snapshots and Validation

**Files:**
- Create: `src/agentbench_frame/miracle/strategy_loader.py`
- Create: `tests/miracle/test_strategy_loader.py`

**Interfaces:**
- Produces: `save_source(path: Path, source: str) -> None`
- Produces: `load_candidate(path: Path, module_key: str) -> MiracleAgent`
- Produces: `validate_candidate(agent: MiracleAgent) -> None`
- Raises: `StrategyValidationError(stage: str, reason: str)` with stages `compile`, `import`, `class`, `instantiate`, `choose_cards`, and `action_shape`

- [ ] **Step 1: Write tests for valid, invalid, and immutable sources**

Create a valid source importing `MiracleAgent` and defining `CandidateAgent`. Assert two snapshot paths load as separate module keys. Assert syntax errors report `compile`, missing class reports `class`, and an `act()` response without `operation_type`/`operation_parameters` reports `action_shape`. Assert `save_source()` refuses to overwrite an existing snapshot with different content.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_strategy_loader.py -q`

Expected: import failure.

- [ ] **Step 3: Implement compile/import/interface validation**

Compile source before `importlib.util.spec_from_file_location`. Require `issubclass(CandidateAgent, MiracleAgent)`. Instantiate with no arguments. Check `choose_cards(0)` has one artifact and three creatures. Use a minimal observation to verify only action shape; official logic remains the legality authority. Do not register the class in `AGENTS` or retain it in a version registry.

- [ ] **Step 4: Run loader tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_strategy_loader.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/miracle/strategy_loader.py tests/miracle/test_strategy_loader.py
git commit -m "feat(miracle): validate immutable strategy snapshots"
```

---

### Task 4: Run Store, Events, and Budget Accounting

**Files:**
- Create: `src/agentbench_frame/miracle/run_store.py`
- Create: `tests/miracle/test_run_store.py`

**Interfaces:**
- Produces: `MiracleRunStore.create(config: LoopConfig, data_dir: Path | None = None, run_id: str | None = None) -> MiracleRunStore`
- Produces: `iteration_dir(index: int) -> Path`, `write_event(event: str, **fields)`, `write_json_atomic(path: Path, value: dict)`, `finish(summary: dict)`
- Produces: `BudgetLedger.charge_rollout()`, `charge_read(episodes: int, decisions: int)`, `charge_usage(usage: dict)`, `charge_api_time(seconds: float)`, `charge_battle_time(seconds: float)`, `check() -> None`, and `snapshot() -> dict`
- Raises: `BudgetExceeded(dimension: str)` before work that would exceed a configured limit

- [ ] **Step 1: Write storage and budget tests**

Assert creation at `runs/24_miracle/{agent}/{run_id}`; copied Skills; append-only JSONL events; iteration directories `iteration-0000`; `run.toml` has `[run] type="rule_iter"`; and `summary.json` appears only on `finish()`. Charge tokens and rollouts to the exact limit, then assert the next charge raises with the correct dimension. Ensure the serialized ledger includes episode reads, decision reads, prompt/completion/total tokens, API seconds, battle seconds, rollouts, and wall seconds.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_run_store.py -q`

Expected: import failure.

- [ ] **Step 3: Implement atomic storage and accounting**

Use `$AGENTBENCH_DATA` only when `data_dir` is absent. Copy the two formal Skills into `run_dir/skills/`. Write `events.jsonl` with one flush per event. Implement atomic JSON/TOML via a named sibling ending `.tmp`, flush, then `replace`. `BudgetLedger.check()` compares elapsed monotonic time and all cumulative counters.

- [ ] **Step 4: Run store tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_run_store.py -q`

Expected: PASS and no `.tmp` files remain.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/miracle/run_store.py tests/miracle/test_run_store.py
git commit -m "feat(miracle): add observable Run storage and budgets"
```

---

### Task 5: Score, Gain, and AUC Metrics

**Files:**
- Create: `src/agentbench_frame/miracle/score.py`
- Create: `tests/miracle/test_score.py`

**Interfaces:**
- Produces: `aggregate_score(episodes: list[dict], candidate_camp_by_episode: dict[str, int]) -> dict`
- Produces: `build_score_curve(iterations: list[dict]) -> dict`
- Produces: `trapezoid_auc(points: list[dict], x_key: str, y_key: str) -> dict`

- [ ] **Step 1: Write exact metric tests**

For two normal episodes where the candidate is camp 0 once and camp 1 once, assert candidate official scores are selected by seat, mean score and win rate are correct, and incomplete episodes stay in `episodes` but do not enter the measured mean. For iterations with raw `10` and evo values `10, 20, 15`, assert gains `0, 10, 5`. For points `(x,y)=(0,10),(2,20),(5,10)`, assert trapezoidal AUC is `75`. For fewer than two measured points, assert `value is None` and reason `insufficient_points`.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_score.py -q`

Expected: import failure.

- [ ] **Step 3: Implement explicit aggregation**

Use only episodes with `terminated_by == "normal"` and empty `errors` for measured score/win rate. Preserve counts for normal, timeout, failed, and missing. Each score-curve point includes iteration, version hash, update status, raw, evo, gain, win rate, completion rate, rollouts, cumulative tokens, episode reads, decision reads, and wall seconds. Compute AUC separately for iteration, rollout, total-token, episode-read, and wall-time axes.

- [ ] **Step 4: Run metric tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_score.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/miracle/score.py tests/miracle/test_score.py
git commit -m "feat(miracle): add aligned score and budget AUC metrics"
```

---

### Task 6: Harness Context and Iteration Orchestrator

**Files:**
- Create: `src/agentbench_frame/miracle/loop.py`
- Create: `tests/miracle/test_loop.py`
- Modify: `src/agentbench_frame/miracle/ig.py`

**Interfaces:**
- Produces: `build_messages(current_source: str, skills: dict[str, str], evidence: list[dict], previous_metrics: dict, budget: dict) -> list[dict]`
- Produces: `run_loop(config: LoopConfig, *, client=None, match_runner=run_match, data_dir=None) -> Path`
- Consumes: `client.propose_strategy(messages) -> StrategyProposal`, snapshot loader, `run_match`, `compare_agents_on_trace`, score functions, and Run store
- Modify IG to accept agents loaded from source snapshots without `AGENTS`

- [ ] **Step 1: Write orchestration tests with fake dependencies**

Use a fake match runner that writes a minimal trace and returns `MatchResult`; use a fake client returning a stronger complete source and usage `{prompt_tokens: 12, completion_tokens: 8, total_tokens: 20}`. Assert:

```python
run_dir = run_loop(config, client=fake_client, match_runner=fake_match, data_dir=tmp_path)
assert (run_dir / "iterations/iteration-0000/strategy.py").exists()
assert (run_dir / "iterations/iteration-0001/candidate.py").exists()
assert (run_dir / "score_curve.json").exists()
assert (run_dir / "ig_curve.json").exists()
assert json.loads((run_dir / "summary.json").read_text())["total_tokens"] == 20
```

Also test malformed proposal: iteration 1 has `status="failed"`, `failure_stage="proposal_json"`, raw response is retained, accepted strategy hash equals iteration 0, and the Run still finishes. Test regression: valid weaker candidate is accepted and recorded with negative gain, not discarded.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_loop.py -q`

Expected: import failure.

- [ ] **Step 3: Implement deterministic evidence selection and messages**

Select episodes in saved evaluation order up to `max_episode_reads`; extract observation/action summaries in trace sequence up to `max_decision_reads`. Charge reads for content actually included. System text states the exact response JSON and `CandidateAgent` contract. Preserve full selected evidence in `llm_request.json`; do not insert API keys or unselected traces.

- [ ] **Step 4: Implement baseline and update orchestration**

Evaluate iteration 0, then for each `1..max_iterations`: check budget, build context, call client, save raw proposal, validate candidate, evaluate configured `(seed, seat)` pairs, compute IG using aligned candidate observations, write `iteration.json`, and update curves. Instantiate a fresh strategy agent for every episode and a fresh old/new pair for every IG trace so stateful agents do not leak across episodes.

- [ ] **Step 5: Implement final summary and failure preservation**

Summary includes `raw`, `final_evo`, `final_gain`, `best_evo`, `best_iteration`, total iterations/episodes/steps, completion and win rates, budget totals, AUC dictionary, score history, IG history, and failure counts. A failed iteration emits `iteration_failed` and does not increment accepted-version identity; it still consumes API/time/token budgets already spent.

- [ ] **Step 6: Run loop and related tests**

Run: `uv run --with pytest python -m pytest tests/miracle/test_loop.py tests/miracle/test_ig.py tests/miracle/test_score.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/miracle/loop.py src/agentbench_frame/miracle/ig.py tests/miracle/test_loop.py
git commit -m "feat(miracle): orchestrate saved LLM strategy iterations"
```

---

### Task 7: Single High-Level CLI and Skill Instructions

**Files:**
- Modify: `src/agentbench_frame/miracle/cli.py`
- Modify: `tests/miracle/test_cli.py`
- Modify: `skills/miracle-harness/SKILL.md`
- Create: `examples/miracle-loop.toml`
- Create: `examples/miracle-initial-strategy.py`

**Interfaces:**
- Produces CLI: `uv run python -m agentbench_frame.miracle loop --config PATH [--data-dir PATH]`
- Retains CLI: `match`, `replay`, `ig`

- [ ] **Step 1: Update CLI tests first**

Change the expected command set to exactly `{"match", "replay", "ig", "loop"}`. Assert `loop --config x.toml --data-dir results` parses two paths and delegates once to `run_loop`; stdout must be JSON containing only `run_dir`, `run_id`, `status`, `score_curve`, and `ig_curve`.

- [ ] **Step 2: Run CLI test and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_cli.py -q`

Expected: failure because `loop` is absent.

- [ ] **Step 3: Implement the CLI entry and examples**

Add `_cmd_loop`; load `LoopConfig`, create `ChatCompletionsClient`, call `run_loop`, read its final summary, and print stable JSON. The example initial source defines an end-round `CandidateAgent`. The TOML uses `api_key_env="OPENAI_API_KEY"`, `base_url="https://api.openai.com"`, one seed, both seats, and one iteration; it contains no key.

- [ ] **Step 4: Update the Harness Skill**

Document the exact loop command, config fields, response contract, Run/iteration/episode/round terminology, output tree, budget meanings, strict-KL-status interpretation, failure semantics, and `AGENTBENCH_DATA=/path/to/AgentBenchResults`. State that complete-source mode is current and Tool Calling is not yet enabled.

- [ ] **Step 5: Run CLI and Skill checks**

Run: `uv run --with pytest python -m pytest tests/miracle/test_cli.py -q`

Run: `rg -n "miracle loop|CandidateAgent|AGENTBENCH_DATA|missing_ratio" skills/miracle-harness/SKILL.md`

Expected: tests pass and all four required terms are found.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/miracle/cli.py tests/miracle/test_cli.py skills/miracle-harness/SKILL.md examples/miracle-loop.toml examples/miracle-initial-strategy.py
git commit -m "feat(miracle): expose one saved harness loop command"
```

---

### Task 8: Official-Logic End-to-End Acceptance and Results Contract

**Files:**
- Create: `tests/miracle/test_loop_e2e.py`
- Modify: `AgentBenchResults/scripts/aggregate.py` only if it is intentionally in the same checked-out workspace and the existing renderer requires explicit Miracle curve discovery; otherwise record the Results change as a separate repository commit.

**Interfaces:**
- Verifies the complete public CLI and on-disk contract

- [ ] **Step 1: Write an end-to-end test with a mock Chat Completions server**

Start a local HTTP server returning a complete deterministic `CandidateAgent` source that improves over the initial end-round policy against `endround` by summoning/moving/attacking. Invoke `main(["loop", "--config", ..., "--data-dir", ...])` without replacing `run_match`, so official logic creates real `.mrc` and `.trace.jsonl` files. Use one seed and one seat to bound runtime.

Assert:

- iteration 0 and 1 strategy snapshots differ;
- both iterations contain normal episode results and nonempty official replay/trace files;
- `score_curve.json` contains aligned versions and iteration 1 has measured evo/gain;
- `ig_curve.json` contains iteration 0 baseline and iteration 1 measured strict status;
- `events.jsonl` contains API, validation, battle, IG, and finish events;
- `run.toml` and `summary.json` pass `agentbench data check` requirements;
- saved request/response/log files do not contain the test API key.

- [ ] **Step 2: Run the acceptance test and verify RED**

Run: `uv run --with pytest python -m pytest tests/miracle/test_loop_e2e.py -q`

Expected: fail at the first missing/inconsistent public artifact.

- [ ] **Step 3: Fix only integration defects revealed by the test**

Keep fixes in their owning modules. Do not add a second loop/export/version interface. If AgentBenchResults currently ignores additional curve files, retain its required `run.toml`/`summary.json` compatibility and add curve links in its game view as a separately tested Results repository change.

- [ ] **Step 4: Run complete verification**

Run: `uv run --with pytest python -m pytest tests/miracle -q`

Run: `uv run python -m agentbench_frame.cli data check --data-dir <temporary-results-root>`

Run: `git diff --check`

Expected: all Miracle tests pass; data check reports the generated Run valid; diff check is silent.

- [ ] **Step 5: Clean generated artifacts**

Delete only test-created local `agentbench_data`, `.pytest_cache`, Miracle `__pycache__`, and generated lock files that were absent before the run. Do not touch `.worktrees`, `.venv`, unrelated caches, or user data.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/miracle tests/miracle skills/miracle-harness/SKILL.md examples
git commit -m "test(miracle): verify OpenAI harness loop end to end"
```

If AgentBenchResults changed, commit it separately inside that repository:

```bash
git add scripts templates
git commit -m "feat(results): display Miracle score and IG curves"
```

---

## Final Review Checklist

- [ ] One command executes baseline, one LLM update, reevaluation, strict IG, score curve, and result export.
- [ ] The initial and generated strategies are immutable source snapshots, not `AGENTS` history entries.
- [ ] Every API request/response and usage record is saved without secrets.
- [ ] Episode/decision reads, rollouts, tokens, API/battle/total time are cumulatively recorded.
- [ ] Invalid and regressed updates remain visible.
- [ ] `raw`, `evo`, `gain`, win/completion rate, and AUC axes are unambiguous.
- [ ] `finite_kl_mean` is never fabricated.
- [ ] Real official `.mrc` and trace files exist for accepted evaluation episodes.
- [ ] `run.toml` and `summary.json` are accepted by the existing Results data checker.
- [ ] The formal Skills are copied into the Run for provenance.
- [ ] No Tool Calling, extra version registry, duplicate loop command, or new runtime dependency was added.

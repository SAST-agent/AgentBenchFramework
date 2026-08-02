# Rollman Native Rollout Budget Design

## Objective

Bound every Codex act by a provider-native weighted-token budget while preserving useful, verifiable candidate edits and complete research provenance. The Rollman k=4 loop uses a 70,000-token act budget, a 420-second hard deadline, and a 120-second no-progress deadline.

## Frozen Runtime Contract

The experiment requires `codex-cli 0.146.0-alpha.9.2`. Before the first billable act, the harness executes a secret-safe local preflight that verifies the exact CLI version and confirms that the generated `CODEX_HOME/config.toml` enables `rollout_budget`.

The generated Codex configuration contains:

```toml
[features.rollout_budget]
enabled = true
limit_tokens = 70000
reminder_at_remaining_tokens = [20000, 10000, 5000]
sampling_token_weight = 1.0
prefill_token_weight = 1.0
```

Weighted usage is recorded as:

```text
(prompt_tokens - cached_input_tokens) * prefill_token_weight
+ completion_tokens * sampling_token_weight
```

Cached input is excluded. Missing usage remains unknown rather than zero.

## Configuration Boundary

`ProviderConfig` owns a nested, secret-free `RolloutBudgetConfig` and an optional exact CLI version requirement. Strict schema validation requires a positive limit, strictly descending unique reminders inside the limit, and positive finite weights whenever the feature is enabled.

The full budget configuration and expected CLI version are serialized into `run-config.json`. Preflight facts are written to `provider-preflight.json`; neither artifact contains an API key.

## Termination and Candidate Acceptance

Codex-native `SessionBudgetExceeded` failures are identified from the JSONL stream and stderr. The provider records `termination_reason=rollout_budget_exhausted`, weighted usage, the configured limit, and access-audit results.

A coding candidate terminated by the rollout budget is eligible for one local quick screen only when all of the following hold:

1. its content hash differs from its parent;
2. the provider access audit has no violation;
3. the immutable snapshot succeeds;
4. the game quick screen completes, which exercises Python compilation, SDK protocol startup, and a real frozen-backend match.

If the quick screen completes, the invocation is recorded as completed with `accepted_after_budget_exhaustion=true` and retains the original termination reason. Otherwise it remains failed and cannot become a branch representative, finalist, search parent, champion, Experience update, or curve point.

Planner and reducer acts do not receive partial-result adoption. Their required files must exist, validate against the existing schema, and originate from an ordinarily completed invocation.

## Timeout Semantics

The hard timeout is 420 seconds per act. The idle timeout is 120 seconds without JSONL or stderr byte progress. Timeout and budget exhaustion are distinct terminal facts. Partial streams are persisted and audited in both cases.

## Observability and Recovery

Each act checkpoint and `act_completed` event records:

- provider status and original termination reason;
- exact first-hand token usage;
- derived weighted usage and configured limit;
- timeout kind, if applicable;
- whether a budget-terminated candidate passed safe adoption;
- raw provider JSONL reference and provider fingerprint.

Resume continues to reject config drift. A safely adopted candidate is indistinguishable from an ordinary completed candidate for deterministic local evaluation and selection, while its budget termination remains available for audit and ablation.

## Verification

Automated tests cover strict budget parsing, generated TOML, CLI/feature preflight, budget-exhaustion classification, weighted-token accounting, safe candidate adoption, rejection of unchanged/invalid candidates, and persistence of termination metadata. The full repository test suite and a no-model preflight run must pass before the paid k=4 cycle starts.

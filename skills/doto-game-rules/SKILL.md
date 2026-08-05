---
name: doto-game-rules
description: Use when reasoning about 23rd DOTO mechanics, scoring, timing, observations, legal actions, terminal states, action masks, or strict information gain before authoring or diagnosing a native policy.
---

# DOTO Game Rules

Treat the vendored official server, map, SDK, and `decision_space.py` as the
authority. Verify strategy assumptions against them before changing a policy.

1. Read [references/rules.md](references/rules.md) for mechanics, score, timing,
   visibility, and termination.
2. Read [references/decision-space.md](references/decision-space.md) before
   reasoning about an action, legality mask, or strict KL/IG.
3. Use `doto-agent-authoring` to translate the decision into `playerAI.cpp`.
4. Use `doto-replay-reader` to diagnose recorded behavior; replay events were
   not visible to the policy.

When prose and code disagree, report the exact source location and follow the
vendored code. Keep benchmark seed, population split, and defeat aggregation in
`doto-benchmark-run`; they are evaluation rules, not game mechanics.

## Quick reference

| Question | Read |
|---|---|
| Score, movement, combat, cooldown, revival | `references/rules.md` |
| Observation, joint action, mask, terminal | `references/decision-space.md` |
| Strict deterministic KL support | `references/decision-space.md` |

## Common mistakes

- Do not map local slot `i` directly to global human ID `i`.
- Do not treat replay-only events as an observation.
- Do not treat an ignored request as an executed action.
- Do not turn timeout, crash, or corrupt replay into a normal loss.
- Do not replace strict KL with action distance, change rate, or score gain.

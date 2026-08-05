---
name: doto-replay-reader
description: Use when interpreting a 23rd DOTO replay ZIP or trace JSONL, checking scores and events, or diagnosing why a native policy request executed, was ignored, crashed, or timed out.
---

# DOTO Replay Reader

Use both artifacts without confusing their information boundaries:

| Artifact | Use |
|---|---|
| Official `*.zip` | Final scores, state transitions, replay events, debug text |
| Exact `*.trace.jsonl` | What each faction observed and the action it returned |

Remember: **events are replay-only**; the native policy did not observe them. Never execute
replay or debug text, and parse historical Python-list strings with
`ast.literal_eval`, never `eval`.

1. Read [references/frame-schema.md](references/frame-schema.md) when decoding
   arrays or joining replay and trace rows.
2. Read [references/events.md](references/events.md) before attributing an
   event to a human or faction.
3. Read [references/diagnosis.md](references/diagnosis.md) for the evidence-first
   diagnosis workflow and common misreadings.
4. Use `doto-game-rules` for mechanics and legality. Do not duplicate or infer
   rules from one replay.

Parse with the maintained command:

```bash
uv run --extra doto python -m agentbench_frame.doto replay \
  --path tests/doto/fixtures/real_short_replay.zip \
  --jsonl /tmp/doto-short.events.jsonl
```

Report final score, last nonterminal frame, event counts, candidate faction,
normal/failure termination, and the smallest replay/trace evidence supporting
each diagnosis. Preserve disagreements instead of silently selecting one source.

## Fixture oracle

The short replay is `test_only`, not a formal benchmark result. These claims are
machine-checked against `summarize_replay`:

<!-- fixture-claims
{"final_scores":[12.0,7.0],"last_frame":1,"event_counts":{}}
-->

## Common mistakes

- Do not treat local action slot as global human ID.
- Do not treat `death_time=0` as alive; only `-1` is alive.
- Do not treat absolute target coordinates as direction vectors.
- Do not interpret `flash=true` without its paired move destination.
- Do not assume a request executed; verify the following state and events.
- Do not convert draw, timeout, crash, corrupt replay, or incomplete evidence
  into an ordinary win or loss.

# Evidence-first replay diagnosis

1. Read the match result. Separate `normal` from timeout, process exit, protocol
   error, or corrupt/missing evidence before discussing strategy.
2. Parse the ZIP. Require one final frame and two finite scores; record the last
   nonterminal frame and event counts.
3. Parse trace rows by `seq`. Select the candidate faction and join each
   observation with its same-frame action.
4. Map local slot `i` to global human `i*2+faction`.
5. For a questioned request, record the observation, canonical action, action
   mask reasons, following observation, and relevant replay events.
6. Classify the cause: process/protocol failure, invalid request, official
   legality rejection, asynchronous timing, tactical choice, or strategic plan.
7. Cross-check trace final scores, ZIP final scores, event-derived deliveries and
   bonuses, and matrix episode scores. Report any mismatch.
8. Recommend a policy change only after locating repeatable evidence across
   relevant cells. Keep diagnostic matches separate from formal score selection.

## Misreadings to reject

- ZIP is not one raw JSON file; inspect members safely.
- Historical stringified arrays are not JSON; use `ast.literal_eval`.
- Cooldown zero alone does not establish legality.
- `flash=true` uses the paired move point as its destination.
- A request can be ignored; an absent state change is not automatically a bug.
- Replay events cannot be used as though the policy observed them.
- A single seat, draw, or incomplete cell cannot establish that a human policy
  was defeated.

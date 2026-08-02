---
name: doto-replay-reader
description: Use when interpreting 23rd DOTO replay ZIPs or trace JSONL, diagnosing a native C++ policy from frames and events, or translating historical numeric arrays and stringified fields into game meaning.
---

# Reading DOTO Replays

## Use the right artifact

One match produces two complementary artifacts:

| Artifact | Meaning |
|---|---|
| `*.zip` | Official replay: one frame-array JSON plus optional player debug text |
| `*.trace.jsonl` | Harness record of exact server observations and native-AI actions |

Use the ZIP for authoritative events and final scores. Use the trace to determine
what a policy saw and returned. Never execute replay text.

```bash
uv run python -m agentbench_frame.doto replay \
  --path agentbench_data/replays/23_doto/match.zip \
  --jsonl agentbench_data/replays/23_doto/match.events.jsonl
```

Historical fields are often Python list strings such as `"[[0, 25.5, ...]]"`.
Parse with `ast.literal_eval`, not `eval` and not direct `json.loads`.

## Rules that matter for diagnosis

- The formal map is 320×320, two factions, five humans per faction, 20 FPS, and
  300 seconds (6000 frames).
- Human ownership is `human_id % faction_number`. With two factions, IDs
  `0,2,4,6,8` belong to faction 0; `1,3,5,7,9` belong to faction 1.
- A policy's five action slots correspond to its controlled humans in increasing
  local index: real ID `local_index * 2 + faction`.
- Carry the opposing crystal into your own target circle for 80 points. A dead
  carrier drops it. A delivered crystal resets to its original position.
- A kill contributes 1 point split by damage contribution. A bonus gives 10.
- Humans revive after 160 frames and are invincible for 40 frames.
- Normal movement is at most 0.6 per frame. Flash range is 20 with 60-frame
  cooldown; a crystal carrier cannot flash.
- Fireball cooldown is 10 frames; it moves 3 per frame and deals 5 splash damage.
- Meteor range is 30, delay 40 frames, cooldown 160 frames, and damage 100.
- Friendly fire is disabled in the official configuration.
- Final winner is the faction with the larger score. Equal scores are a draw;
  do not give an automatic win to either seat.

## Frame schema

Initialization is sent only to the AI/trace:

```json
{"frame": 0, "map": 0, "faction": 0}
```

An ordinary official replay frame contains `frame`, `humans`, `fireballs`,
`meteors`, `balls`, `scores`, `bonus`, and `events`. The native AI receives the
same state except `events`; events are replay-only.

### Human row

`[id, x, y, hp, meteor_num, meteor_time, flash_num, flash_time, fireball_time, death_time, inv_time]`

| Index | Meaning |
|---:|---|
| 0 | global human ID |
| 1–2 | absolute x, y position |
| 3 | HP; full HP is 100 |
| 4 | remaining meteor uses |
| 5 | meteor cooldown frames |
| 6 | remaining flash uses |
| 7 | flash cooldown frames |
| 8 | fireball cooldown frames |
| 9 | `-1` when alive; otherwise frames until revival |
| 10 | remaining invincibility frames |

### Other rows

- Fireball: `[x, y, rotation_radians, source_human_id, fireball_id]` in replay;
  older AI views may omit the final ID.
- Meteor: `[x, y, frames_until_impact, source_human_id, meteor_id]` in replay;
  older AI views may omit the final ID.
- Crystal: `[x, y, carrier_human_id, original_faction]`; carrier `-1` means free.
- `scores`: `[faction0_score, faction1_score]`.
- `bonus[i]`: whether bonus point `i` currently contains a collectible bonus.
- Final frame: `{"frame": -1, "scores": "[...]"}`.

## Action schema in trace

One action jointly controls five humans:

```json
{
  "flag": 0,
  "move": [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
  "shoot": [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
  "meteor": [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
  "flash": [false, false, false, false, false],
  "debug": ""
}
```

Coordinates are absolute targets. `[-1,-1]` is no-op for move/shoot/meteor.
`flash[i]=true` changes `move[i]` into a flash request; it is not a separate
destination. Invalid, out-of-range, wall, cooldown, dead-human, or prohibited
actions are ignored by official logic.

## Event table

Each event is a list whose first item is its type:

| Type | Remaining values | Meaning |
|---:|---|---|
| 1 | `human_id, fireball_id` | fireball fired |
| 2 | `victim_id, damage, source_id` | damage applied |
| 3 | `human_id, x, y` | human died |
| 4 | `human_id, meteor_id` | meteor cast |
| 5 | `human_id` | opposing crystal picked up |
| 6 | `x, y, source_id` | fireball exploded |
| 7 | `x, y, source_id` | meteor impacted |
| 8 | `human_id` | human revived |
| 9 | `human_id, from_x, from_y, to_x, to_y` | flash executed |
| 10 | `human_id, crystal_faction` | crystal delivered |
| 11 | `bonus_id` | bonus spawned |
| 12 | `human_id, bonus_id` | bonus collected |

Attribute humans and sources with `id % 2`. For event 2, `victim_id` is the
damaged human and `source_id` is the attacker; do not reverse them.

## Reliable diagnosis workflow

1. Parse the ZIP and confirm it contains a final frame and two numeric scores.
2. Parse trace rows in `seq` order; never infer order from timestamps alone.
3. For the candidate faction, join each ordinary `observation` with the next
   same-frame `action`.
4. Map local action slot `i` to global human `i*2+faction`.
5. Compare the requested action with the following observation and replay
   events. No state change may mean the request was illegal, on cooldown, too
   late for the asynchronous frame, or overwritten by a later request.
6. Separate strategic failure from process/protocol failure using the match
   result's `terminated_by` and `errors`.
7. Cross-check trace final scores, ZIP final scores, event-derived goals/bonuses,
   and CLI summary. Report any disagreement instead of choosing one silently.

Python API:

```python
from agentbench_frame.doto.replay import iter_replay, summarize_replay

summary = summarize_replay(replay_path)
for frame in iter_replay(replay_path):
    if frame.frame >= 0 and frame.events:
        print(frame.frame, frame.events)
print(summary.final_scores, summary.event_counts, summary.by_faction)
```

## Common misreadings

1. Do not treat the ZIP as a single raw JSON file; open its JSON member safely.
2. Do not use `eval` on stringified lists.
3. Do not treat action slot index as global human ID.
4. Do not read `death_time=0` as alive; only `-1` is alive.
5. Do not read cooldown zero alone as sufficient legality; death, range, wall,
   uses, and held crystal still matter.
6. Do not interpret aim coordinates as direction vectors; they are absolute.
7. Do not interpret `flash=true` without its paired move destination.
8. Do not assume a requested action executed; verify the next state/events.
9. Do not expose replay `events` as though the native policy observed them.
10. Do not turn a draw, timeout, crash, corrupt replay, or incomplete match into
    an ordinary win/loss.

## Fixture-backed validation

`tests/doto/fixtures/real_short_replay.zip` is generated through the historical
framing path with two original-schema no-op AIs and is explicitly `test_only`.
`real_short_replay.expected.json` records its SHA-256, seed, final score
`[12.0, 7.0]`, final nonterminal frame `1`, and event counts. The replay parser
test requires exact agreement; this validates field decoding without presenting
the short fixture as a formal benchmark result.

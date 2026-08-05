# Replay and trace schemas

## Official replay ZIP

Open the ZIP safely and select its JSON member. The JSON contains an array of
ordinary frames followed by a final frame. Many historical fields are strings
containing Python-list syntax; the maintained parser uses `ast.literal_eval`.

An ordinary frame contains `frame`, `humans`, `fireballs`, `meteors`, `balls`,
`scores`, `bonus`, and `events`. The final frame is
`{"frame":-1,"scores":"[...]"}`.

Human row:

```text
[id, x, y, hp, meteor_num, meteor_time, flash_num, flash_time,
 fireball_time, death_time, inv_time]
```

- Fireball: `[x,y,rotation,source_human_id,fireball_id]`; older AI views may
  omit the final ID.
- Meteor: `[x,y,frames_until_impact,source_human_id,meteor_id]`; older AI views
  may omit the final ID.
- Crystal: `[x,y,carrier_human_id,original_faction]`; carrier `-1` means free.
- Scores: `[faction0_score,faction1_score]`.

## Trace JSONL

Trace rows are ordered by integer `seq`. An initialization observation is
`{"frame":0,"map":0,"faction":f}`. Ordinary `observation` payloads contain
the current state but no events. `action` payloads contain the same-frame joint
`move[5]`, `shoot[5]`, `meteor[5]`, and `flash[5]` request plus debug text.

Join a candidate observation with the next same-faction, same-frame action.
Map local action slot `i` to global human `i*2+faction`. Compare the request to
the following observation and replay event; asynchronous application means the
same-frame replay state precedes that request.

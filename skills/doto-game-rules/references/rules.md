# Authoritative DOTO mechanics

All paths are relative to `src/agentbench_frame/doto/`.

## World and time

- The fixed map is **320 × 320**, with two factions and five humans per
  faction. Source: `official_server/Maps/0.json` fields `width`, `height`,
  `faction_number`, and `human_number`.
- A match lasts 300 seconds at **20 FPS**, hence **6000 frames**. Sources:
  `official_server/Maps/0.json:time_of_game` and
  `official_server/Arguments.py:frames_per_second,frames_of_game`.
- The server sends state, waits one frame interval, then applies the most recent
  joint request. Source: `official_server/main.py:RunGame`.
- The native policy observes the full current arrays of humans, fireballs,
  meteors, crystals, scores, and bonuses plus its faction and map. It does not
  observe the replay `events` array. Sources: `official_server/main.py:RunGame`,
  `sdk/main.cpp:readMap,readFrame`, and `sdk/logic.h:Logic`.

## Identity and movement

- Global human IDs interleave factions. Local slot `i` for faction `f` maps to
  `i * faction_number + f`, which is `i*2+f` on map 0. Sources:
  `official_server/main.py:RunGame` action loops and `sdk/logic.h:Logic`.
- Normal movement is at most 0.6 units per frame. Flash replaces that slot's
  move, reaches at most 20 units, has a 60-frame cooldown, and is prohibited
  while carrying a crystal. Sources: `official_server/Arguments.py` and
  `official_server/main.py:move,flash`.
- Destinations outside the map or inside walls are ignored. Coordinates are
  absolute targets, not direction vectors. Source: `official_server/main.py`.

## Combat, death, and revival

- Humans start with 100 HP. Death lasts 160 frames; revival grants 40 frames of
  invincibility. Sources: `official_server/Arguments.py:human_hp`,
  `frames_of_death`, `frames_of_invincible`, and `official_server/main.py`.
- Fireballs have a 10-frame cooldown, travel 3 units per frame, and deal 5
  splash damage. Source: `official_server/Arguments.py` fireball constants.
- Meteor targets must be within 30 units. Impact is delayed 40 frames, cooldown
  is 160 frames, and damage is 100. Source: `official_server/Arguments.py`
  meteor constants and `official_server/main.py:cast`.
- Friendly fire is disabled. Dead, invincible, out-of-range, on-cooldown, or
  otherwise illegal requests are ignored. Sources: `official_server/Arguments.py`
  and `official_server/main.py:fireball_hurt,meteor_hurt,shoot,cast`.

## Crystals, bonuses, and score

- Only an opposing human can pick up a crystal. A dead carrier drops it. Carry
  it into that carrier faction's target circle to score **80 points**; the
  crystal resets afterward. Sources: `official_server/main.py:pickupball,goal`
  and `official_server/Arguments.py:goal_score`.
- A kill contributes 1 point divided among damage sources in proportion to
  damage. A collected bonus contributes 10 points. Sources:
  `official_server/main.py:death,getbonus` and score constants in
  `official_server/Arguments.py`.
- Higher final score wins; equal scores are a draw. Candidate-oriented score
  difference and the two-seat defeat rule are benchmark aggregation semantics,
  documented by `doto-benchmark-run` rather than game logic.

## Termination

Normal termination is the official final payload `frame=-1` with two numeric
scores after all 6000 frames. A process exit, protocol violation, timeout,
missing/corrupt replay, or absent final payload is non-normal and must remain an
explicit failure. Sources: `official_server/main.py:RunGame`, `match.py`, and
`evaluation.py`.

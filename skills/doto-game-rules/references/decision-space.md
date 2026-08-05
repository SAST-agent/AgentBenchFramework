# Observation and decision space

The executable contract is implemented by `sdk/main.cpp`, `sdk/logic.h`, and
`decision_space.py`. Use the vendored source if this reference becomes stale.

## Observation

Initialization is `{"frame":0,"map":0,"faction":f}`. Each ordinary
observation contains `frame`, `humans`, `fireballs`, `meteors`, `balls`,
`scores`, and `bonus`; map metadata is loaded from `Maps/0.json`. Replay-only
`events` are absent. `decision_space.py:parse_observation` validates the
normalized form used for IG comparison.

## Canonical joint macro-action

One decision jointly controls five local humans:

```text
DotoAction(move[5], shoot[5], meteor[5], flash[5])
```

Each move, shoot, or meteor entry is a finite absolute `(x,y)` point or the
no-op sentinel `(-1,-1)`. Each flash entry is boolean; `flash[i]` changes
`move[i]` into a flash request. `decision_space.py:canonicalize_action` requires
exactly five entries in each field and rejects non-finite coordinates.

## Explanatory action mask

`decision_space.py:action_mask` checks each controlled human for:

- local slot to global ID mapping;
- alive/dead state;
- map boundary and wall destination;
- normal move, meteor, and flash range;
- fireball, meteor, and flash cooldown;
- remaining meteor/flash uses;
- flash target presence and carrier prohibition.

The mask explains requests on the recorded observation. The asynchronous
official server remains the final legality arbiter, so verify execution in the
next state and replay events rather than assuming a valid-looking request ran.

## Terminal and failure states

The only normal terminal observation is `frame=-1` with two scores. Match
timeouts, crashes, protocol errors, and corrupt or missing evidence are failure
terminations, not action-space outcomes or ordinary losses.

## Information-gain support

Coordinates make the legal joint action set continuous; do not invent a finite
enumeration. Use the observation-dependent measurable legal set as theoretical
support. For a deterministic old/new comparison on one exact observation, the
computational support is the union of their two canonical actions:

- identical canonical actions: strict KL = 0;
- different deterministic actions: strict KL = infinity, stored with numeric
  `value: null` and an explicit reason;
- missing, invalid, timed-out, crashed, or misaligned action: IG status
  `missing` with an explicit reason.

Never label coordinate distance, action-change rate, score gain, or behavior
features as strict KL or IG. Sources: `ig.py:compare_policies_on_trace` and
`decision_space.py`.

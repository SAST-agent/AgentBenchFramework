# Safe policy patterns

## Indexing and coordinates

Map local slot `i` to global row `i*2+faction` on the official two-faction map,
or use `i * map.faction_number + faction` generically. Pass finite absolute coordinates.
Never pass NaN, infinity, a normalized direction, or an unchecked
point outside `[0,width) × [0,height)`.

Use explicit cancellation for no-op; the serialized sentinel is `(-1,-1)`.
Before requesting a point, check the destination against `logic->isWall` and
keep a margin when geometry or floating-point rounding approaches a wall.

## Per-human legality

For every controlled human, check:

- `death_time == -1` before any action;
- `fire_time == 0` before shooting;
- `meteor_number > 0 && meteor_time == 0` before meteor;
- `flash_num > 0 && flash_time == 0` before flash;
- target range and a non-degenerate shoot direction;
- whether that human carries a crystal before flash.

The official logic is asynchronous. A cooldown of zero alone does not prove
execution; use replay evidence to confirm the next state and events.

## Strategy structure

Compute observations first, choose one bounded role per local human, then emit
one joint action. Useful roles may include carrier, escort, defender, bonus
collector, or pressure, but derive assignments from the current state instead
of hard-coding a hidden opponent identity.

Prioritize requests explicitly when abilities conflict. For example, decide a
move target first, then opt into flash only if the same target is legal; never
set flash independently.

## Persistent state

Use small function-static or namespace-static values for persistent state only
when the next frame can validate them. Store stable IDs or goals rather than
references into SDK vectors. Clear a target when its human dies, an object
resets, the faction changes, or the state no longer supports the plan.

Keep loops bounded by the small SDK collections and avoid sleeping, threads,
network access, filesystem dependencies, randomness without a recorded seed,
and large debug output. Deterministic decisions make replay diagnosis and strict
KL comparison meaningful.

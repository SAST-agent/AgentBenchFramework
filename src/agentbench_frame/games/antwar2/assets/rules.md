# AntWar2 frozen rules

## Game and roles

AntWar2 is a symmetric two-player tower-defense contest on a 19 × 19 offset
hex grid. The players are P0 and P1. Each owns a camp with 50 HP at `(2, 9)`
or `(16, 9)`, receives the same public state schema, and submits an ordered
bundle of operations each round. Ant movement, targeting, attacks, spawning,
status effects, and combat resolution are environment transitions; the policy
does not directly choose an ant move.

Both roles are evaluated. Global tower IDs are shared across the two players,
so ownership must be checked rather than inferred from an ID range.

## Public state

The SDK constructs `BackendState` from the public stream. It exposes the round,
map, tower deltas, ants, both coin balances and camp HP values, production and
ant-HP levels, weapon cooldowns, and active effects. `towers` is a delta stream:
an entry with `type == -1` deletes that global tower ID; another entry creates
or updates it. The complete tower set must be reconstructed across rounds.
Live policies read camp HP, production level, and ant-HP level from
`state.bases[player].hp`, `.generation_level`, and `.ant_level`. The replay
JSON names the corresponding arrays `camps`, `speedLv`, and `anthpLv`; those
replay names are not fields on the live `BackendState` object.

## Terminal result

A camp reaching zero HP loses. If both camps reach zero in the same resolution,
P0 wins. A game also ends at round 512. The frozen backend then compares, in
order: camp HP, opponent ants killed, fewer super-weapons used, lower total AI
time; an exact tie is awarded to P0. The replay field `winner` is authoritative
only when the match completed without a protocol or infrastructure fault.

## Atomic operations

The policy returns an ordered list. An empty list is `HOLD` for behavioral
measurement. The protocol operations are:

- `BUILD_TOWER(x, y)` (11): build a level-0 `Basic` tower on an empty cell
  owned by the acting player. There is no tower-type argument.
- `UPGRADE_TOWER(tower_id, target_type)` (12): upgrade one owned tower along
  the legal upgrade tree.
- `DOWNGRADE_TOWER(tower_id)` (13): downgrade an upgraded owned tower, or
  remove a Basic tower, with the backend-defined HP-proportional refund.
- `USE_LIGHTNING_STORM(x, y)` (21), `USE_EMP_BLASTER(x, y)` (22),
  `USE_DEFLECTOR(x, y)` (23), and `USE_EMERGENCY_EVASION(x, y)` (24).
- `UPGRADE_GENERATION_SPEED` (31) and `UPGRADE_GENERATED_ANT` (32).

Legality is evaluated in execution order against the public state plus the
accepted prefix of the same bundle. Bounds, cell ownership and occupancy,
tower ownership and upgrade path, per-round tower/base reuse, EMP restrictions,
coins, cooldowns, and level caps all apply. A rejected operation is diagnostic
evidence and is excluded from the accepted behavioral sequence.

## Economy

Each player starts with 50 coins. Basic income is 3 coins every two rounds.
The next tower build costs `15 * 3**(n//2)`, doubled when active tower count
`n` is odd. Tower upgrades cost 60 and 200 coins for the two upgrade levels.
The two camp upgrades cost 200 and 250 coins. Weapon costs for codes 21–24 are
90, 135, 60, and 60 coins. Ant kills, breaches, and tower downgrade/removal
apply the frozen backend rewards or refunds.

## Authority

The rule authority is the content-hashed official C++ backend archive. The
canonical Python SDK defines the candidate-facing public interface and helper
legality checks. If prose and executable behavior differ, a live match against
the frozen backend decides; transport adaptations may preserve byte framing but
must not alter rules, state, accepted operations, timing outcomes, or opponents.

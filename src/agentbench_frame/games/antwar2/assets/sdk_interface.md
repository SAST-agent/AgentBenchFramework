# AntWar2 candidate interface

The submitted package contains `ai.py`, `common.py`, `main.py`, `protocol.py`,
and the frozen `SDK/` tree. `ai.py` exports class `AI`. The public policy entry
point is:

```python
AI.choose_operations(
    state: SDK.backend.state.BackendState,
    player: int,
    bundles: list[SDK.utils.actions.ActionBundle] | None = None,
) -> list[SDK.backend.model.Operation]
```

`player` is `0` for P0 and `1` for P1. The same source must support both roles.
The returned list is ordered. The protocol wrapper filters it with
`state.can_apply_operation(player, operation, accepted_prefix)` and sends only
accepted operations.

`BackendState` supplies public fields including `round_index`, `towers`,
`ants`, `bases`, `coins`, `weapon_cooldowns`, and `active_effects`. Public
helpers include `towers_of`, `ants_of`, `tower_at`, `tower_by_id`,
`tower_count`, `build_tower_cost`, `upgrade_tower_cost`, `weapon_cost`,
`strategic_slots`, and `can_apply_operation`. A policy may maintain finite
private memory derived from its observed public stream.

Construct protocol operations with `Operation(OperationType, arg0, arg1)`:

- build and weapon atoms use `arg0=x`, `arg1=y`;
- tower upgrade uses `arg0=tower_id`, `arg1=target_type`;
- downgrade uses `arg0=tower_id`;
- camp upgrades have no arguments.

Candidate code may use visible-state conditionals, graph algorithms, planning,
state machines, finite memory, and other interpretable mechanisms. It may not
read opponent source, branch on opponent identity, seed, or replay ID, access
hidden engine state, or invoke the evaluation-only oracle.

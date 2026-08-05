# Replay event table

Events are replay-only lists. The first value is the type:

| Type | Remaining values | Meaning |
|---:|---|---|
| 1 | `human_id, fireball_id` | Fireball fired |
| 2 | `victim_id, damage, source_id` | Damage applied |
| 3 | `human_id, x, y` | Human died |
| 4 | `human_id, meteor_id` | Meteor cast |
| 5 | `human_id` | Opposing crystal picked up |
| 6 | `x, y, source_id` | Fireball exploded |
| 7 | `x, y, source_id` | Meteor impacted |
| 8 | `human_id` | Human revived |
| 9 | `human_id, from_x, from_y, to_x, to_y` | Flash executed |
| 10 | `human_id, crystal_faction` | Crystal delivered |
| 11 | `bonus_id` | Bonus spawned |
| 12 | `human_id, bonus_id` | Bonus collected |

Attribute a human or source to faction `id % 2`. For damage type 2, the first ID
is the victim and the last ID is the attacker; reversing them produces a false
diagnosis. Absence of an event is not proof that the request was never made;
inspect the trace action first.

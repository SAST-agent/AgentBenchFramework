# Native LostSpace replay format

The game logic writes one JSON file per match via its embedded `Replay` object
(`backend_sources/corpus/25_lostspace/logic/gamecode_logic/src/replay.py`). The
harness points the logic at a replay path in the init handshake
(`{"replay": <path>}`) and the logic writes the file at game-over.

## Top-level structure

The file is a JSON array:

```
[
  birthplaces,     # index 0
  round_1,         # index 1
  round_2,
  ...
  round_N,         # up to 100
  score_dic        # last index
]
```

So `len == 2 + number_of_rounds` (typically 102 for a full 100-round game).

### `birthplaces` (index 0)

```
[[x, y, z], [x, y, z], [x, y, z], [x, y, z]]
```

Four spawn points, one per player (0..3). **Coordinates are shifted by -3**
relative to the grid (the grid is 0..6; the replay centres it on -3..3). Layer
`z` is 0 (top), 1, 2 (bottom). Default spawns:

| player | replay coord | grid coord |
|--------|--------------|------------|
| 0 | `[-3, -3, 1]` | `[0, 0, 1]` |
| 1 | `[ 3, -3, 1]` | `[6, 0, 1]` |
| 2 | `[ 3,  3, 1]` | `[6, 6, 1]` |
| 3 | `[-3,  3, 1]` | `[0, 6, 1]` |

### each `round_i` (index 1 .. N)

A list of **4 turns** (one per player, in seat order 0..3). Each turn is a list
of **action dicts** recorded that turn (an empty list if the player skipped /
was dead / had escaped). Each action dict has a `"type"` and (except map
updates) a `"playerid"`. Action types (see `replay.py`):

| type | meaning | key fields |
|------|---------|------------|
| `move` / `flink` | player moved / flink-moved | `pos` (shifted) |
| `attack` | player attacked | `attack` = `[startpos, endpos]` (shifted) |
| `died` | player died | optional `box` (drop position) |
| `regenerate` | player respawned | `pos` (shifted) |
| `escaped` | player escaped (win) | — |
| `hp_update` / `cure` / `kit` | hp change | `hp` |
| `getkey` | player gained keys | `keyid` = list |
| `keymachine` | opened a key machine | `pos` (shifted) |
| `escape_capsule` | started/aborted escape | `to_escape` bool |
| `place_trap` | placed LandMine/Sticky | `trap_type`, `pos` |
| `detect` | probed a tile | `tar_pos` (shifted) |
| `use_kit` | used a medkit | `hp` |
| `ai_error` | the AI crashed/misbehaved | `error_log` |
| `inspect` | media-player inspect | `pos`, `interprops` |
| `tool_update` | inventory changed | `tools` |
| `map_update` | map event (not a player action) | `args` = `[kind, ...]` |

`map_update` kinds: `box_disappear`, `door`, `trap_trigger`, `trap_destroy`.

### `score_dic` (last index)

```
{"0": 4, "1": 3, "2": 2, "3": 1}
```

Rank points: **4 = 1st, 1 = 4th** (always a total order, no ties). Convert to a
ranking by sorting player ids by score descending.

## Viewing

Text (now):

```bash
python -m agentbench_frame.lostspace.replay_view path/to/replay.json
python -m agentbench_frame.lostspace.replay_view replay.json --round 12 --player 0
```

HTML player (planned, see `replay_view_html.py`): not yet implemented.

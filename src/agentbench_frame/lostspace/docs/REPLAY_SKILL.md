# LostSpace — Rules & Replay Skill (agent-consumable)

> Human-written rules + replay format for the HL coding agent, in the
> six-section shape of the SnakeGo REPLAY_SKILL. **The source wins**: every
> rule cites its `gamecode_logic` file:line so you can verify. Do not edit
> rules by paraphrasing this doc — read the linked source.
>
> - Rules authority: `E:\HL_Agent\AgentBench\backend_sources\corpus\25_lostspace\logic\gamecode_logic\`
> - Replay writer: `...\gamecode_logic\src\replay.py`

## 0. Replay file shape

One JSON array per match, written by the logic at game-over
(`src/replay.py`):

```
[ birthplaces, round_1, round_2, ..., round_N, score_dic ]
```

`len == 2 + number_of_rounds` (up to 102 for a full 100-round game).

- **index 0 `birthplaces`** — `[[x,y,z]×4]`, one spawn per player (0..3).
  **Coordinates are shifted by −3** relative to the grid (grid is 0..6,
  replay centers it on −3..3). Layer z: 0=top, 1, 2=bottom. Default spawns
  (player → replay coord → grid coord):
  `0 [−3,−3,1]→[0,0,1]`, `1 [3,−3,1]→[6,0,1]`, `2 [3,3,1]→[6,6,1]`,
  `3 [−3,3,1]→[0,6,1]`.
- **each `round_i`** — a list of **4 turns**, one per player in seat order
  0..3. Each turn is a list of action dicts (empty if the player skipped /
  was dead / had escaped). Each dict has a `"type"` and (except map updates) a
  `"playerid"`.
- **last index `score_dic`** — `{"0": 4, "1": 3, "2": 2, "3": 1}`. Rank
  points: **4 = 1st, 1 = 4th**, always a total order (no ties).

## 1. Field table

Action `type` values in a turn (`src/replay.py`):

| type | meaning | key fields |
|------|---------|------------|
| `move` / `flink` | moved / flink-moved | `pos` (shifted) |
| `attack` | attacked | `attack` = `[startpos, endpos]` (shifted) |
| `died` | died | optional `box` (drop position) |
| `regenerate` | respawned | `pos` (shifted) |
| `escaped` | escaped (win) | — |
| `hp_update` / `cure` / `kit` | hp change | `hp` |
| `getkey` | gained keys | `keyid` = list |
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

`roundbegin` observation fields your AI receives (`src/communicate.py`,
`AIClient.py:356-464`): `type`, `inturn`, `status`, `state`, `hp`, `keys`,
`tools`, `others`, plus the legal-action projection `attack` / `move` /
`detect` / `interprops` (`src/player.py:230-262`).

## 2. Op / action codes + legality

Legal-action set per turn = `Player.get_legal_actions()` (`player.py:230-262`):

```python
{
  'attack':    [0, 1]                       # player IDs you may attack
  'move':      [True, False, ...] (len 8)   # 8-direction reachability
  'detect':    True/False                    # detect off cooldown?
  'interprops': ['Box', 'KeyMachine', ...]  # interactable props on your tile
}
```

Wire formats the logic parses in `GameController.solve()`
(`GameController.py:173-258`):

| action[0] | format | effect | legality |
|---|---|---|---|
| `move` | `["move",[x,y,z]]` | move one step; SP cost = path distance | reachable edge, tile enabled |
| `attack` | `["attack",[x,y,z],id]` | adjacent/same-tile player, −70 HP | in `attack` list |
| `interact` | `["interact","Box"]` | loot a dropped Box | Box on your tile |
| `interact` | `["interact","Materials",tool]` | gather materials → pick a tool | Materials on tile |
| `interact` | `["interact","KeyMachine"]` | use a key machine (gain a key) | KeyMachine on tile |
| `interact` | `["interact","EscapeCapsule",True]` | **start** escape | capsule on tile; need all 4 keys; enter WaitForEscape |
| `interact` | `["interact","EscapeCapsule",False]` | **abort** an in-progress escape | only legal while WaitForEscape |
| `trap` | `["trap","LandMine"\|"Sticky"]` | place a trap on your tile | have it in inventory |
| `tool` | `["tool","Kit"]` | use medkit (+100 HP) | have Kit |
| `tool` | `["tool","Transport",[x,y,z]]` | teleport | have Transport |
| `detect` | `["detect",[x,y,z]]` | probe adjacent tile for traps | off cooldown |
| `finish` | — | end your turn | **always legal for a live player** |

8 movement directions (`DirectionSeqence`, `config.py:29`):
`0:(0,1,0) 1:(0,-1,0) 2:(1,0,0) 3:(-1,0,0) 4:(1,-1,0) 5:(-1,1,0) 6:(1,1,0)
7:(-1,-1,0)` (first 4 are cardinal/door directions). Board is a fixed graph
with hand-coded edges (`map.py`), elevators at `config.py:24`, shrink at
`config.py:26` (`ShrinkTime = [40,50,60,70,80]`).

Player status (`config.py:37-43`): `Alive=0, Died=1, Escaped=2, Skip=3,
WaitForEscape=4, Error=5`. In overtime (round ≥ 80) dead players do not
respawn (`GameController.player_died()` `:336-339`).

## 3. Key events (extract these to understand a game)

- **Win / rank** — `GameController.gameover()` (`GameController.py:360-391`):
  1. escapers in escape order; 2. alive survivors by **score** descending
  (ties by seat order, later seats first); 3. dead/error last. Rank points
  `{"0":4,"1":3,"2":2,"3":1}` (`:385-388`).
- **Scoring** (`player.py`): **+1 per key-pickup** (`:45-48`), **+2 per kill**
  (`:76-77`), **−3 per death** (`:158`); death drops a Box of your keys
  (`:163`).
- **Win condition** — hold all **4 keys** then enter the escape capsule at
  `(3,3,0)` (`player.py:83-84`, `config.py:25`). You **start with your own key**
  (`player.py:35` `key = {id}`). Each KeyMachine grants its key at the start
  of your **NEXT** round and makes you Skip that round
  (`interactive_props.py:144-157`); a same-turn key-count check right after
  `interact("KeyMachine")` is always False.
- **Escape** — start with `["interact","EscapeCapsule",True]`. You then enter
  `WaitForEscape` and must survive **3 of your own rounds** defenseless — the
  only legal action is `["interact","EscapeCapsule",False]` (abort)
  (`interactive_props.py:120`, `GameController.py:190-199`).
- **Vitals** — HP 200, SP 1 (refresh each of your small rounds,
  `GameController.py:121`), attack 70, LandMine 120, Kit +100, respawn 5
  rounds, detect cooldown 5, prop refresh 8 (`config.py`).
- **ai_error on seats 1–3 is background noise** — ranked algorithms
  *correctly* stall/crash on the judger and rely on its TLE to advance; only
  seat 0's `ai_error` matters to you.

## 4. Common misreads (trap list)

1. **`interprops` is INTEGER-CODES, not strings.** `self.view.nodes[i]
   .interprops` is a list of int/object codes (1=EscapeCapsule, 2=KeyMachine;
   the client appends the string `'Box'`). `if 'KeyMachine' in interprops` is
   ALWAYS False and silently disables key collection. Call
   `self.interact('KeyMachine')` and branch on `result['success']`.
2. **Coordinates are shifted by −3.** A replay `pos` of `[3,3,0]` is grid
   `[6,6,0]`; the escape capsule is grid `[3,3,0]` = replay `[0,0,0]`. Do not
   mix the two frames.
3. **`score_dic` is rank points, not kills/keys.** `4` means 1st place. A
   high-score loser can outrank a low-score survivor.
4. **Empty turn ≠ skipped-only.** A turn can be empty because the player was
   dead or escaped, not just skipped.
5. **`map_update` is not a player action** — it has no `playerid`; don't
   attribute it to seat 0.
6. **Escape start flag is `True`, not `False`.** `["interact","EscapeCapsule",
   False]` only **aborts** an in-progress escape; while Alive it is rejected
   (`map.py:340`, `interactive_props.py:108-127`). To win you MUST send `True`.

## 5. Agent learning flow

Standard per-act flow for reading a replay:

1. `python -c "import json;print(json.load(open('REPLAY'))[-1])"` — score_dic.
2. `python -m agentbench_frame.lostspace.replay_view path/to/replay.json`
   — text digest (or `--round N --player 0` for one turn).
3. Extract key events (§3) for seat 0: keys/escaped/died/rounds/last-action.
4. Write the Lesson into `EXPERIENCE.md`; on consolidation acts, compress and
   mark superseded lessons.

## 6. 验证 (Verification) — one real replay parse, checked

Before trusting your understanding, **parse one real replay and prove it**:

1. Pick one replay JSON and load it: `python -c "import json; d=json.load(
   open('REPLAY')); print(d[-1])"`.
2. Write, for a chosen round `i` and your seat 0: the turn's action list, the
   meaning of each field in the first action dict, and what you conclude the
   player did.
3. State the `score_dic` you read and the resulting ranking (sorted by score
   descending).
4. **Validate against ground truth**: independently parse `d[-1]` (the actual
   `score_dic`) and confirm your stated value matches. If it does, your parse
   is correct; if not, find where the offset/wrong-index happened and fix
   your mental model.

Write the result as `RULES_VALIDATION.md` (fields checked, stated score_dic,
pass/fail, mismatch explanation). This is the harness's acceptance test that
the rules doc is being read and the replay format understood.

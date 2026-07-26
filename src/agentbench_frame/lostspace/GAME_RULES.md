# LostSpace — Game Rules Reference

> **This document points at the authoritative source.** LostSpace has no
> standalone rulebook; the rules *are* the game logic. Where this doc and the
> source disagree, **the source wins** — file refs are given for every rule so
> you can verify. Do not edit rules by paraphrasing this doc; read the linked
> file.

## Where the rules live (authoritative)

The game backend is checked into the corpus, never copied into the framework:

```
E:\HL_Agent\AgentBench\backend_sources\corpus\25_lostspace\
├── README.md                  # contest metadata (届次 25, type)
├── archives/                  # original zips + SHA256SUMS (immutable)
│   ├── gamecode_logic__1.zip
│   ├── judge_dev_logic__lostspace.zip
│   └── judge_dev_sample_ai__lostspace_ai.zip
└── logic/
    ├── gamecode_logic/        # ← THE authoritative rules
    │   ├── main.py
    │   └── src/
    │       ├── config.py          # ★ all numeric constants
    │       ├── GameController.py # ★ round flow, win/lose, scoring
    │       ├── player.py         # ★ player state + legal actions
    │       ├── map.py            # board, edges, shrink, elevators
    │       ├── interactive_props.py  # Box / Materials / KeyMachine / EscapeCapsule
    │       ├── tools.py / toolbag.py  # Kit / LandMine / Sticky / Transport
    │       ├── traps.py
    │       ├── position.py
    │       ├── communicate.py    # stdio wire protocol
    │       ├── replay.py         # replay file writer
    │       └── mapconf2.map      # static map config
    └── judge_dev_sample_ai/      # official sample AI (the API surface a coder uses)
        ├── AIClient.py
        └── main.py
```

**When in doubt, read `config.py` first** — every HP/SP/score/cost/dimension
constant is there. Then `GameController.py` for flow, `player.py` for the
legal-action set, `map.py` for the board.

The framework-side harness (how the framework spawns the logic + AIs) is at
`AgentBenchFramework/src/agentbench_frame/lostspace/` — see `match.py`,
`evaluator.py`, `ladder.py`, and `replay_format.md` (replay file layout).

---

## 1. What kind of game it is

- **四人回合制生存收集对抗** — 4-player, turn-based survival / collection /
  confrontation. (source: `25_lostspace/README.md`)
- Each match is one episode. Up to **100 big rounds** (`RoundMax = 100`,
  `config.py:8`). A big round = one turn per player (4 small rounds).
- The board is **3 layers × 7×7** (`MAP_HEIGHT=3`, `MAP_LENGTH=7`,
  `MAP_WIDTH=7`, `config.py:20-22`). Layer 0 = top (escape layer), 2 = bottom.
- Four players spawn at corners of the middle layer:
  `BirthPos = [(0,0),(6,0),(6,6),(0,6)]` (`config.py:23`), all at `z=1`.

## 2. Win condition & scoring

A player **escapes (wins)** by collecting all **4 keys** and entering the
escape capsule — `Player.can_escaped()` returns true at 4 keys
(`player.py:83-84`). The escape capsule sits at `(3,3,0)` on the top layer
(`config.py:25`).

Ranking at game-over (`GameController.gameover()`, `GameController.py:360-391`):
1. Players who escaped, in escape order.
2. Players still alive, sorted by **score** descending (ties broken by seat
   order — later seats rank first).
3. Dead/error players last.

Final rank points: `{"0":4,"1":3,"2":2,"3":1}` (4 = 1st, always a total order,
no ties) — `GameController.py:385-388`.

### Score rules (`player.py`)
- **+1** per key-pickup event (regardless of how many keys at once)
  — `player.py:45-48`.
- **+2** for killing another player — `player.py:76-77`.
- **−3** for dying — `player.py:158`.
- Dying drops a **Box** containing your keys onto the ground (`player.py:163`).

## 3. Player vitals & action economy

| Stat | Max | Source |
|---|---|---|
| HP | 200 (`HpMax`) | `config.py:6` |
| SP (action points) | 1 (`SpMax`) | `config.py:7` |
| Attack damage | 70 (`AttackHarm`) | `config.py:13` |
| LandMine damage | 120 (`LandMineHarm`) | `config.py:14` |
| Kit heal | 100 (`KitRecovery`) | `config.py:12` |
| Respawn delay | 5 rounds | `player.py:183` |
| Detect cooldown | 5 rounds | `player.py:210` |
| Prop refresh interval | 8 rounds (`RefreshInterval`) | `config.py:15` |
| Carry limits | LandMine 3, Sticky 3, Transport 2, Kit 2 | `config.py:18` |

SP refreshes to `SpMax` at the start of each of your small rounds
(`GameController.turn()`, `GameController.py:121`). Most actions cost 1 SP
(`PropSpCost`, `AttackCost`, `player.py:66`).

## 4. Player status (`config.py:32-43`)

```
Alive            = 0   normal play
Died             = 1   dead, awaiting respawn (5 rounds; in overtime, gone)
Escaped          = 2   won
Skip             = 3   skip next round (e.g. Sticky trap)
WaitForEscape    = 4   in escape countdown
Error            = 5   AI crashed/errored → treated as a loss
```

In **overtime** (after the last shrink at `round >= ShrinkTime[-1] = 80`,
`config.py:26`, `GameController.big_round_end()` `:115-116`), dead players do
**not** respawn (`GameController.player_died()` `:336-339`).

## 5. The legal action set (★ authoritative)

The definitive per-turn legal-action set is **`Player.get_legal_actions()`**
at `player.py:230-262`. It returns a dict with these keys:

```python
{
  'attack':    [0, 1]                       # player IDs you may attack
  'move':      [True, False, ...] (len 8)   # 8-direction reachability
  'detect':    True/False                    # detect off cooldown?
  'interprops': ['Box', 'KeyMachine', ...]  # interactable props on your tile
}
```

The wire-level action formats the AI sends back (parsed in
`GameController.solve()`, `GameController.py:173-258`) are:

| action[0] | format | effect |
|---|---|---|
| `move` | `["move", [x,y,z]]` | move one step; SP cost = path distance |
| `attack` | `["attack", [x,y,z], target_id]` | attack adjacent/same-tile player, −70 HP |
| `interact` | `["interact", "Box"]` | loot a dropped Box |
| `interact` | `["interact", "Materials", tool]` | gather materials → pick a tool |
| `interact` | `["interact", "KeyMachine"]` | use a key machine (gain a key) |
| `interact` | `["interact", "EscapeCapsule", False]` | start/abort escape |
| `trap` | `["trap", "LandMine"\|"Sticky"]` | place a trap on your tile |
| `tool` | `["tool", "Kit"]` | use a medkit (+100 HP) |
| `tool` | `["tool", "Transport", [x,y,z]]` | teleport to a tile |
| `detect` | `["detect", [x,y,z]]` | probe an adjacent tile for traps |
| `finish` | (round end) | end your turn |

Sticky trap makes the victim **skip 2 rounds** (`config.py` + GameController
notes). Detect disables a found enemy trap (`player.py:211-213`).

The 8 movement directions (`DirectionSeqence`, `config.py:29`):
```
0:(0,1,0) 1:(0,-1,0) 2:(1,0,0) 3:(-1,0,0)
4:(1,-1,0) 5:(-1,1,0) 6:(1,1,0) 7:(-1,-1,0)
```
(first 4 are the cardinal/door directions)

## 6. The board, edges, elevators, shrink

- The map is a fixed graph, not a free grid — each tile has a hand-coded edge
  list (`map.py`, the big `self.node[..].edges = [...]` block). Movement is
  only along declared edges.
- **Elevators** connect layers at `ELEVATOR = [(3,0),(5,3),(3,6),(1,3)]`
  (`config.py:24`) — movement between z-layers happens through elevator tiles.
- **Shrink** (缩圈) progressively disables tiles at rounds
  `ShrinkTime = [40,50,60,70,80]` (`config.py:26`); disabled tiles become
  unusable (`node.able = 0`). Birth tiles are never shrunk (`player.py:191`).
- **Materials points** refresh every 8 rounds (`RefreshInterval`);
  high-tier materials have a countdown tracked in `Map.count`.

## 7. The API surface a coder writes against

The official sample AI (`judge_dev_sample_ai/AIClient.py`) is the **de-facto
SDK**. Your `agent.py` subclasses `AIClient` and overrides `play()`. Key helpers
(exact signatures at the cited lines):

| method | line | returns |
|---|---|---|
| `get_my_num()` | 490 | your player id |
| `get_my_pos()` | 692 | `[x,y,z]` |
| `get_other_pos(id)` | 695 | another player's pos |
| `get_my_hp()` / `get_other_hp(id)` | 699/702 | HP |
| `get_keys()` | 496 | your key set |
| `get_other_keys(id)` | 706 | another's key count |
| `get_neighbors(pos)` | 510 | adjacent tiles |
| `get_check_cd()` | 503 | detect cooldown remaining |
| `move(pos)` | 524 | move one step |
| `attack(pos, pid)` | 544 | attack |
| `interact(tool, capsule=False)` | 558 | Box/Materials/KeyMachine/EscapeCapsule |
| `view_box(box_type, tool_type)` | 574 | inspect a box |
| `put_trap(trap_type)` | 599 | place LandMine/Sticky |
| `use_tool(tool, transport_pos=None)` | 620 | Kit / Transport |
| `detect(detect_pos)` | 647 | probe for traps |
| `get_spawn_pos(x)` / `get_escape_pos()` / `get_landmine_pos()` / `get_sticky_pos()` | 658-692 | key locations |
| `receive_data()` / `send_opt(data)` | 331/349 | low-level stdio |
| `init_game()` / `start_turn()` / `in_turn()` / `off_turn()` / `end_turn()` | 356-464 | turn lifecycle |
| `play()` | 710 | **override this** — your strategy |
| `run()` | 716 | main loop (do not override) |

The framework's candidate agents
(`AgentBenchFramework/.../lostspace/candidates/v1..v4/agent.py`) are copies of
this sample AI with `play()` customized — that's the iteration target.

## 8. Wire protocol (how the logic talks to your AI)

Stdio JSON frames, defined in `communicate.py`. Message `type` values your AI
receives in its main loop (`AIClient.run()`, the sample's `run()`):

- `"id"` — game init (call `init_game()`)
- `"roundbegin"` — your small round starts (call `start_turn()`)
- `"offround"` — someone else's round (call `off_turn()`)
- otherwise — in-round action request (call `in_turn()`)

You respond with `["action", ...]` frames (see §5) or `finish` to end your
turn. The harness's TLE/error handling (`match.py`) sends `{"player":-1,...}`
on timeout so the logic calls `player_error` and the game advances — do not
rely on a stalled game hanging.

## 9. Replay format

Documented in `AgentBenchFramework/.../lostspace/replay_format.md`. One JSON
array per match: `[birthplaces, round_1, …, round_N, score_dic]`. Each round is
4 turns; each turn is a list of action dicts (`move`/`attack`/`died`/`escaped`/
`place_trap`/`detect`/`ai_error`/…). Coordinates are shifted by −3. View with
`python -m agentbench_frame.lostspace.replay_view path/to/replay.json`.

---

## Citing the rules in code

When an HL candidate or a test needs a rule value, **import from the source**
rather than hardcoding a paraphrase:

```python
# Prefer the source over a copied constant:
from backend...gamecode_logic.src.config import HpMax, RoundMax, AttackHarm
```

(Note: the corpus is read-only reference; the framework does not import it at
runtime — it spawns the logic as a subprocess. For tests, copy the specific
constant with a `# source: config.py:6` comment so drift is visible.)

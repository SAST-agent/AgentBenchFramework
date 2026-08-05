# Fixed C++ SDK API

Sources are `sdk/logic.h`, `sdk/logic.cpp`, `sdk/main.cpp`, `sdk/geometry.h`, and
`sdk/playerAI.h`. The Framework copies these fixed files and official map into
an isolated build directory, adds only `playerAI.cpp`, and runs `make`.

## State

Use `Logic *logic = Logic::Instance()` inside `playerAI()`.

| Field | Meaning |
|---|---|
| `logic->frame` | Current positive frame |
| `logic->faction` | Candidate faction |
| `logic->map` | Dimensions, factions, humans, walls, spawns, crystals, targets, bonuses, duration |
| `logic->humans` | All interleaved global humans |
| `logic->fireballs`, `meteors`, `crystal`, `bonus` | Current objects |

For local slot `i`, the corresponding global row is
`i * logic->map.faction_number + logic->faction`. Operation methods still take
the local slot `i`, not that global ID.

## Operations

- `move(i, Point(x,y))`, `shoot(i, Point(x,y))`, `meteor(i, Point(x,y))`
- `flash(i)` changes the same frame's move into a flash
- `unmove(i)`, `unshoot(i)`, `unmeteor(i)`, `unflash(i)` explicitly cancel
- `debug(text)` replaces frame debug; `debugAppend(text)` appends, within the
  SDK's bounded debug field

The serialized no-op sentinel is `Point(-1,-1)`. Prefer cancellation methods
instead of manually mutating `logic->ope`.

## Minimal complete source

This example deliberately encodes no competitive strategy. It demonstrates
the complete interface and explicit no-op behavior and must compile unchanged.

```cpp
#include "playerAI.h"

void playerAI()
{
    Logic *logic = Logic::Instance();
    for (int i = 0; i < logic->map.human_number; ++i)
    {
        logic->unmove(i);
        logic->unshoot(i);
        logic->unmeteor(i);
        logic->unflash(i);
    }
}
```

Build a complete candidate with:

```bash
uv run --extra doto python -m agentbench_frame.doto build \
  --player-ai /path/to/playerAI.cpp --output-dir /tmp/doto-candidate
```

Success requires `exit_code == 0` and a fresh `main.out`. The builder deletes a
stale executable after failure.

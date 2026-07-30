"""Demo: save two strategy versions → load → diff → play both.

    python -m snakego.versioning_demo
"""
from __future__ import annotations

from pathlib import Path

from .versioning import VersionStore


# A simple "always move +x" strategy — version 1
CODE_V1 = '''\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from snakego.agents import BaseAgent
from snakego.env import ACT_MOVE_BASE, DX, DY

class RightAgent(BaseAgent):
    name = "right_v1"
    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        # prefer +x (action 1), fallback to any legal
        if 1 in acts:
            return 1
        return acts[0]

def create_agent():
    return RightAgent()
'''

# Improved: prefer +x, but dodge walls/items — version 2
CODE_V2 = '''\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from snakego.agents import BaseAgent
from snakego.env import ACT_MOVE_BASE, DX, DY

class RightAgentV2(BaseAgent):
    name = "right_v2"
    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        # prefer +x, then +y, then any legal
        for pref in (1, 2, 3, 4):
            if pref in acts:
                return pref
        return acts[0]

def create_agent():
    return RightAgentV2()
'''


def main() -> int:
    store_dir = Path("outputs/snakego_versions")
    import shutil
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store = VersionStore(store_dir)

    print("=== Strategy Versioning Demo ===\n")

    # save v1
    m1 = store.save("mover", CODE_V1, description="always move +x")
    print(f"saved {m1.version_id}: {m1.description}  hash={m1.code_hash}")

    # save v2 (child of v1)
    m2 = store.save("mover", CODE_V2, description="prefer +x, fallback cycle",
                    parent_version=m1.version)
    print(f"saved {m2.version_id}: {m2.description}  hash={m2.code_hash}")

    # list all
    print(f"\nstrategies: {store.list_strategies()}")
    for v in store.list_versions("mover"):
        print(f"  {v.version_id}  score={v.score}  parent=v{v.parent_version}")

    # load both agents
    a1 = store.load_agent("mover", 1)
    a2 = store.load_agent("mover", 2)
    print(f"\nloaded: {a1!r}  /  {a2!r}")

    # diff
    d = store.diff("mover", 1, 2)
    print(f"\ndiff v1→v2: code_changed={d['code_changed']}  "
          f"lines {d['lines_v1']}→{d['lines_v2']}  delta={d['delta_lines']:+d}")

    # play them against each other
    from .smoke_test import play_game
    from .agents import RandomAgent
    r1 = play_game(a1, RandomAgent(0), seed=3)
    r2 = play_game(a2, RandomAgent(0), seed=3)
    print(f"\nv1 vs random: winner=P{r1['winner']} score={r1['scores']}")
    print(f"v2 vs random: winner=P{r2['winner']} score={r2['scores']}")

    # record scores
    store.update_score("mover", 1, r1["scores"][0])
    store.update_score("mover", 2, r2["scores"][0])
    m1b = store.load_meta("mover", 1)
    m2b = store.load_meta("mover", 2)
    print(f"\nscored: v1={m1b.score}  v2={m2b.score}")

    # assertions
    assert store.latest_version("mover") == 2
    assert m2.parent_version == 1
    assert d["code_changed"] is True
    print("\nVERSIONING DEMO PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

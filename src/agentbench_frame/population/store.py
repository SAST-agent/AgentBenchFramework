"""
Population: versioned on-disk registry of strategies.

Layout:
    {root}/{game}/{strategy_id}/
        strategy.json      # current (latest) manifest
        <artifacts>        # rule.py | policy.pt | bot.py
        versions.jsonl     # append-only version log
        versions/{v}/      # snapshot of each historical version
    {root}/{game}/index.json
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.strategy.base import BaseStrategy, StrategyMeta, load_strategy


def default_root() -> str:
    return os.environ.get(
        "AGENTBENCH_POPULATION",
        os.path.join(os.getcwd(), "population"),
    )


class Population:
    def __init__(self, root: Optional[str] = None, game: str = "30_antwar2"):
        self.root = root or default_root()
        self.game = game
        os.makedirs(self.game_dir, exist_ok=True)

    @property
    def game_dir(self) -> str:
        return os.path.join(self.root, self.game)

    def path_for(self, strategy_id: str) -> str:
        return os.path.join(self.game_dir, strategy_id)

    def exists(self, strategy_id: str) -> bool:
        return os.path.exists(os.path.join(self.path_for(strategy_id), "strategy.json"))

    def latest_version(self, strategy_id: str) -> int:
        history = self.history(strategy_id)
        if not history:
            return -1
        return max(int(h.get("version", -1)) for h in history)

    def register(self,
                 strategy: BaseStrategy,
                 parent_id: Optional[str] = None,
                 note: str = "") -> Tuple[str, int]:
        sid = strategy.meta.strategy_id or strategy.name
        strategy.meta.strategy_id = sid
        strategy.meta.game = self.game
        strat_dir = self.path_for(sid)

        if self.exists(sid):
            old_version = self._current_version(sid)
            self._snapshot(sid, old_version)
            version = old_version + 1
            strategy.meta.version = version
            strategy.meta.parent_id = parent_id or sid
        else:
            version = 0
            strategy.meta.version = 0
            strategy.meta.parent_id = parent_id or ""

        strategy.save(strat_dir)
        self._append_history(sid, version, strategy.meta.parent_id, note)
        self._refresh_index()
        return sid, version

    def get(self, strategy_id: str, version: Optional[int] = None) -> BaseStrategy:
        if version is None:
            return load_strategy(self.path_for(strategy_id))
        snap = os.path.join(self.path_for(strategy_id), "versions", str(version))
        if not os.path.exists(snap):
            raise KeyError(f"strategy {strategy_id} has no version {version}")
        return load_strategy(snap)

    def list_strategies(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(self.game_dir):
            return out
        for name in sorted(os.listdir(self.game_dir)):
            meta_path = os.path.join(self.game_dir, name, "strategy.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    out.append(json.load(f))
        return out

    def list_ids(self) -> List[str]:
        return [m.get("strategy_id", "") for m in self.list_strategies()]

    def history(self, strategy_id: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.path_for(strategy_id), "versions.jsonl")
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def remove(self, strategy_id: str) -> bool:
        strat_dir = self.path_for(strategy_id)
        if not os.path.isdir(strat_dir):
            return False
        shutil.rmtree(strat_dir)
        self._refresh_index()
        return True

    def _current_version(self, strategy_id: str) -> int:
        meta_path = os.path.join(self.path_for(strategy_id), "strategy.json")
        if not os.path.exists(meta_path):
            return -1
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("version", 0))

    def _snapshot(self, strategy_id: str, version: int):
        strat_dir = self.path_for(strategy_id)
        snap_dir = os.path.join(strat_dir, "versions", str(version))
        os.makedirs(snap_dir, exist_ok=True)
        for name in os.listdir(strat_dir):
            if name == "versions":
                continue
            src = os.path.join(strat_dir, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(snap_dir, name))

    def _append_history(self, strategy_id: str, version: int,
                        parent_id: str, note: str):
        path = os.path.join(self.path_for(strategy_id), "versions.jsonl")
        record = {
            "version": version,
            "parent_id": parent_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": note,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _refresh_index(self):
        index_path = os.path.join(self.game_dir, "index.json")
        index = {
            "game": self.game,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strategies": self.list_strategies(),
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def import_player_zip(self, zip_path: str, strategy_id: str,
                          note: str = "") -> Tuple[str, int]:
        """Register an extracted ladder player package as an external strategy."""
        from agentbench_frame.strategy.external_strategy import ExternalStrategy
        meta = StrategyMeta(
            strategy_id=strategy_id, game=self.game, kind="external",
            source=zip_path, tags=["ladder"],
        )
        strategy = ExternalStrategy(name=strategy_id, meta=meta)
        return self.register(strategy, note=note or "imported from " + zip_path)

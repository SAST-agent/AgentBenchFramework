"""
DataSplit: partition the population into train / validation / hidden-test.
"""

from __future__ import annotations

import json
import os
import random as _random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SplitAssignment:
    train: List[str] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    hidden: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitAssignment":
        return cls(
            train=list(d.get("train", [])),
            validation=list(d.get("validation", [])),
            hidden=list(d.get("hidden", [])),
        )


class DataSplit:
    def __init__(self, population):
        self.pop = population

    @property
    def path(self) -> str:
        return os.path.join(self.pop.game_dir, "split.json")

    def load(self) -> Optional[SplitAssignment]:
        if not os.path.exists(self.path):
            return None
        with open(self.path) as f:
            data = json.load(f)
        if "assignment" in data:
            data = data["assignment"]
        return SplitAssignment.from_dict(data)

    def save(self, assignment: SplitAssignment):
        with open(self.path, "w") as f:
            json.dump({"game": self.pop.game, "assignment": assignment.to_dict()}, f, indent=2)

    def random_split(self, train_ratio: float = 0.6,
                     validation_ratio: float = 0.2,
                     seed: int = 42) -> SplitAssignment:
        ids = self.pop.list_ids()
        rng = _random.Random(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * validation_ratio))
        assignment = SplitAssignment(
            train=ids[:n_train],
            validation=ids[n_train:n_train + n_val],
            hidden=ids[n_train + n_val:],
        )
        self.save(assignment)
        return assignment

    def split_by_rule(self, predicate: Callable[[str], str],
                      ids: Optional[List[str]] = None) -> SplitAssignment:
        ids = ids if ids is not None else self.pop.list_ids()
        assignment = SplitAssignment()
        for sid in ids:
            bucket = predicate(sid)
            if bucket == "train":
                assignment.train.append(sid)
            elif bucket == "validation":
                assignment.validation.append(sid)
            elif bucket == "hidden":
                assignment.hidden.append(sid)
        self.save(assignment)
        return assignment

    def manual(self, train=None, validation=None, hidden=None) -> SplitAssignment:
        assignment = SplitAssignment(
            train=list(train or []),
            validation=list(validation or []),
            hidden=list(hidden or []),
        )
        self.save(assignment)
        return assignment

    def train_ids(self) -> List[str]:
        a = self.load()
        return a.train if a else []

    def validation_ids(self) -> List[str]:
        a = self.load()
        return a.validation if a else []

    def hidden_ids(self) -> List[str]:
        a = self.load()
        return a.hidden if a else []

    def visible_ids(self) -> List[str]:
        a = self.load()
        if a is None:
            return self.pop.list_ids()
        return a.train + a.validation

"""Version management with rollback.

Each iteration produces an immutable snapshot: the consolidated Weights plus
the Experience at that point, plus the eval result and IG. Snapshots are kept
forever so any past version can be restored. Restoring = rebind the live
decide() to that snapshot's weights.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import copy


@dataclass
class Snapshot:
    iteration: int
    label: str
    weights: Dict[str, float]
    experience_lessons: int
    eval_summary: Dict = field(default_factory=dict)
    ig: Dict = field(default_factory=dict)
    note: str = ""


class VersionStore:
    """Append-only history. rollback() returns a past snapshot's weights."""

    def __init__(self):
        self._snapshots = []

    def save(self, snap: Snapshot):
        self._snapshots.append(snap)

    def latest(self):
        return self._snapshots[-1] if self._snapshots else None

    def get(self, iteration):
        for s in self._snapshots:
            if s.iteration == iteration:
                return s
        return None

    def rollback(self, iteration):
        """Return the weights snapshot at iteration (does not delete later)."""
        s = self.get(iteration)
        if s is None:
            raise KeyError("no snapshot at iteration %d" % iteration)
        return copy.deepcopy(s.weights)

    def list_versions(self):
        return [(s.iteration, s.label, s.eval_summary, s.ig.get("ig_kl"))
                for s in self._snapshots]

    def export(self):
        return [asdict(s) for s in self._snapshots]


def weights_to_dict(w):
    return asdict(w)


def dict_to_weights(d):
    from snakego.strategy_core import Weights
    return Weights(**{k: d[k] for k in d})

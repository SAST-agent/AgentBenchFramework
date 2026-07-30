"""The Experience Skill: a single, evolving document the agent reads before
deciding and appends to after each iteration.

This is NOT another code version. It is a structured record of lessons learned
from replay, distilled into actionable weight/feature guidance. The strategy
core (strategy_core.py) consumes it as its starting Weights, and the loop
(loop.py) appends a new entry after every iteration. Compression happens here:
when two lessons conflict, the later one supersedes and the older is archived
rather than kept -- so the document stays coherent, not an ever-growing pile.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json


@dataclass
class Lesson:
    iteration: int
    source: str            # which replay(s) produced this
    evidence: str          # the measured fact (quantitative)
    insight: str           # what it means in plain language
    weight_delta: Dict[str, float] = field(default_factory=dict)
    feature_to_add: Optional[str] = None
    superseded_by: Optional[int] = None   # iteration id that replaced this


@dataclass
class Experience:
    """The whole accumulated, compressed experience."""
    lessons: List[Lesson] = field(default_factory=list)
    weights_snapshot: Dict[str, float] = field(default_factory=dict)
    archived: List[Lesson] = field(default_factory=list)

    def active_lessons(self):
        return [l for l in self.lessons if l.superseded_by is None]

    def add(self, lesson: Lesson):
        # compression: if a new lesson contradicts an older one on the same
        # weight axis, mark the older as superseded (archived, not deleted).
        for l in self.lessons:
            if l.superseded_by is not None:
                continue
            if set(l.weight_delta) & set(lesson.weight_delta):
                l.superseded_by = lesson.iteration
                self.archived.append(l)
        self.lessons = [l for l in self.lessons if l.superseded_by is None]
        self.lessons.append(lesson)
        # apply weight deltas to the running snapshot
        for k, v in lesson.weight_delta.items():
            self.weights_snapshot[k] = round(self.weights_snapshot.get(k, 0.0) + v, 4)

    def to_dict(self):
        return asdict(self)


# The seed experience, distilled from real human-vs-human replay analysis
# (work/analyze_humans.py output). These are FACTS measured from ranked humans,
# not guesses.
SEED_LESSONS = [
    Lesson(
        iteration=0,
        source="match_rank09_rank15, match_r11_r13, match_r15_r06 (human vs human)",
        evidence="ranked humans split at round 10-22 when snake length 9-16; "
                 "they reach 4 snakes; I (v6) capped at 3 and lost the tempo race.",
        insight="Split aggressively once length>=9 and there is room; cap should "
                "be 4 snakes, not 3, to match human action economy.",
        weight_delta={"split_value": 1.5},
    ),
    Lesson(
        iteration=0,
        source="match_rank09_rank15: big_seals at rounds 25/47/114 (+6..+20 walls)",
        evidence="Humans perform burst seals from round ~25 onward, gaining "
                 "+6 to +20 walls per seal. v6 almost never sealed big.",
        insight="Sealing is the primary territory mechanism, not slow crawling. "
                "Weight seal_area higher and lower the seal threshold in mid-game.",
        weight_delta={"seal_area": 0.8, "approach_sealable": 0.3},
    ),
    Lesson(
        iteration=0,
        source="railgun uses: humans fire 1-3 per game",
        evidence="Ranked humans use railgun to clear blocking walls and reopen "
                 "expansion lanes. v6 never fired one.",
        insight="Hold the railgun for when a wall blocks expansion; don't ignore it.",
        weight_delta={"railgun_value": 1.0},
    ),
    Lesson(
        iteration=0,
        source="myrollout_v6_rank15: I die at round 152-371",
        evidence="Against rank15 my snakes get boxed in and die mid-game; "
                 "single-step flood-fill keep-alive undercounts future traps.",
        insight="Survival is the ceiling. Add a trap_penalty and pocket-avoidance "
                "term; consider 2-step lookahead for the survival gate.",
        weight_delta={"trap_penalty": -1.0, "overcommit_corner": -0.5},
    ),
]


def seed_experience():
    """Start from the full default Weights, then apply distilled lessons on top.

    Without this, deltas replace the tuned defaults with raw offsets, producing
    a much weaker policy than intended (e.g. split_value 1.5 instead of 7.5).
    """
    from dataclasses import asdict
    from snakego.strategy_core import Weights
    exp = Experience()
    exp.weights_snapshot = asdict(Weights())
    for l in SEED_LESSONS:
        exp.add(l)
    return exp

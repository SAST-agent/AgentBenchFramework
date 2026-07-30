"""Coding-agent iteration loop: evaluate -> propose -> validate -> re-evaluate -> accept/reject.

This module implements the closed loop that connects every other piece of
the framework:

    strategy registry  ->  arena (payoff matrix + Elo)  ->  coding agent
    proposes new code  ->  validate (loads + plays)  ->  accept / reject
    ->  save new version  ->  repeat

The LLM seam is a thin protocol (LLMInterface).  When no real LLM
is available (offline, CI), StaticLLM applies a sequence of known
heuristic improvements, each one a valid strategy.py that defines
create_agent().  This lets the loop run end-to-end without network
access and still produce a measurable Elo / win-rate gain.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import BaseAgent, GreedyAgent, RandomAgent
from .arena import Arena, EloTracker
from .versioning import VersionStore


# --------------------------------------------------------------------------- #
# LLM seam
# --------------------------------------------------------------------------- #


class LLMInterface:
    """Protocol: generate(prompt: str) -> str producing strategy code."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# baseline strategy templates (used by StaticLLM)
# Each template is a complete strategy.py defining create_agent().
# They form a strict improvement ladder verified by arena evaluation.
# --------------------------------------------------------------------------- #

# v0 -- random moves (intentionally weak baseline).
TEMPLATE_V0 = '''\
"""Strategy v0: random valid actions (weak baseline)."""
from __future__ import annotations
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from snakego.agents import BaseAgent


class RandomStrategy(BaseAgent):
    name = "random_v0"
    def __init__(self):
        self.rng = random.Random(0)
    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        return self.rng.choice(acts)


def create_agent():
    return RandomStrategy()
'''

# v1 -- first-safe-move: picks the first non-suicidal move, no item logic.
# Better than random because it never wastes moves, but no strategy.
TEMPLATE_V1 = '''\
"""Strategy v1: deterministic first-safe-move (survival-only)."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from snakego.agents import BaseAgent
from snakego.env import DX, DY, ACT_MOVE_BASE


class SafeFirstAgent(BaseAgent):
    name = "safe_first_v1"
    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        head = snake.coor_list[0]
        for a in acts:
            if a > 4:
                continue
            d = a - ACT_MOVE_BASE
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if 0 <= nx < obs["length"] and 0 <= ny < obs["width"]:
                if obs["snake_map"][nx][ny] != snake_id:
                    return a
        return acts[0]


def create_agent():
    return SafeFirstAgent()
'''

# v2 -- full greedy: item-chase + split + solidify + railgun.
# A close port of the proven GreedyAgent that beats random ~87%.
TEMPLATE_V2 = '''\
"""Strategy v2: full greedy item-chase + split + solidify + railgun."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from snakego.agents import BaseAgent
from snakego.env import DX, DY, ACT_MOVE_BASE, ACT_RAILGUN, ACT_SPLIT


class FullGreedyAgent(BaseAgent):
    name = "full_greedy_v2"

    def act(self, obs, snake_id, game):
        acts = game.valid_actions()
        snake = game._get_snake(snake_id)
        coor = snake.coor_list
        head = coor[0]

        # 1. fire railgun immediately if held
        if ACT_RAILGUN in acts and snake.has_railgun():
            return ACT_RAILGUN

        my_snakes = [s for s in obs["snakes"] if s["camp"] == snake.camp]
        is_first = (len(my_snakes) > 0 and my_snakes[0]["id"] == snake_id)

        # 2. split when first snake is long enough
        if is_first and ACT_SPLIT in acts and snake.length >= 10 and len(my_snakes) < 4:
            return ACT_SPLIT

        # classify moves
        L, W = obs["length"], obs["width"]
        safe = []
        solidify = []
        for d in range(4):
            a = ACT_MOVE_BASE + d
            if a not in acts:
                continue
            nx, ny = head[0] + DX[d], head[1] + DY[d]
            if not (0 <= nx < L and 0 <= ny < W):
                continue
            occ = obs["snake_map"][nx][ny]
            if occ == snake_id:
                solidify.append(a)
            else:
                safe.append(a)

        # 3. first snake chases items
        if is_first:
            best = self._toward_item(obs, snake_id, head, safe)
            if best is not None:
                return best
        else:
            # other snakes: solidify or follow tail
            if solidify:
                return solidify[0]
            tail = coor[-1]
            best = self._toward(head, tail, safe)
            if best is not None:
                return best

        # 4. fallback
        if safe:
            return safe[0]
        if solidify:
            return solidify[0]
        return acts[0]

    def _toward_item(self, obs, snake_id, head, safe):
        cur_round = obs["turn"]
        best_action = None
        best_dist = 10 ** 9
        for it in obs["items"]:
            if it["time"] > cur_round:
                continue
            if it["time"] + 16 <= cur_round:
                continue
            dist = abs(it["x"] - head[0]) + abs(it["y"] - head[1])
            if it["time"] <= cur_round + dist and it["time"] + 16 > cur_round + dist:
                for a in safe:
                    d = a - ACT_MOVE_BASE
                    nx, ny = head[0] + DX[d], head[1] + DY[d]
                    nd = abs(it["x"] - nx) + abs(it["y"] - ny)
                    if nd <= dist and nd < best_dist:
                        best_dist = nd
                        best_action = a
        return best_action

    def _toward(self, src, dst, safe):
        dist = abs(dst[0] - src[0]) + abs(dst[1] - src[1])
        best = None
        best_d = dist
        for a in safe:
            d = a - ACT_MOVE_BASE
            nx, ny = src[0] + DX[d], src[1] + DY[d]
            nd = abs(dst[0] - nx) + abs(dst[1] - ny)
            if nd <= best_d:
                best_d = nd
                best = a
        return best


def create_agent():
    return FullGreedyAgent()
'''


# --------------------------------------------------------------------------- #
# StaticLLM -- offline "coding agent" that applies known improvements
# --------------------------------------------------------------------------- #


class StaticLLM(LLMInterface):
    """A deterministic, offline stand-in for a real coding-agent LLM.

    It cycles through a fixed ladder of strategy templates (v0 -> v1 -> v2),
    each strictly better than the last.  This lets the iteration loop
    demonstrate a measurable Elo / win-rate gain without any network
    calls, which is essential for CI.
    """

    def __init__(self, templates=None):
        self._templates = templates if templates is not None else [TEMPLATE_V0, TEMPLATE_V1, TEMPLATE_V2]
        self._idx = 0

    def reset(self):
        self._idx = 0

    def generate(self, prompt):
        if self._idx >= len(self._templates):
            return self._templates[-1]
        code = self._templates[self._idx]
        self._idx += 1
        return code

    @property
    def step_name(self):
        names = ["random_v0", "safe_first_v1", "full_greedy_v2"]
        idx = min(self._idx - 1, len(names) - 1) if self._idx > 0 else 0
        return names[max(idx, 0)]


# --------------------------------------------------------------------------- #
# evaluation helper
# --------------------------------------------------------------------------- #


def evaluate_strategy(agent, opponents, n_seeds=4, base_seed=0):
    """Play agent vs each opponent (swapped sides, n_seeds each).

    Returns (win_rate, h2h) where h2h maps each opponent name to the
    fraction of games won by agent.
    """
    from .smoke_test import play_game

    agent.name = "candidate"
    h2h = {}
    total_wins = 0
    total_games = 0
    for opp_name, opp in opponents.items():
        wins = 0
        games = 0
        for s in range(n_seeds):
            seed = base_seed + s
            r = play_game(agent, opp, seed=seed)
            if r["winner"] == 0:
                wins += 1
            games += 1
            r2 = play_game(opp, agent, seed=seed + 1000)
            if r2["winner"] == 1:
                wins += 1
            games += 1
        rate = wins / games if games else 0.0
        h2h[opp_name] = round(rate, 4)
        total_wins += wins
        total_games += games
    overall = total_wins / total_games if total_games else 0.0
    return round(overall, 4), h2h


# --------------------------------------------------------------------------- #
# IterationLoop data classes
# --------------------------------------------------------------------------- #


@dataclass
class IterationStep:
    step: int
    description: str
    accepted: bool
    win_rate: float
    h2h: dict
    elo: float
    delta_elo: float
    code_lines: int
    error: Optional[str] = None


@dataclass
class IterationReport:
    strategy_name: str
    steps: List[IterationStep] = field(default_factory=list)
    best_version: Optional[int] = None
    best_win_rate: float = 0.0
    best_elo: float = 1500.0

    @property
    def improved(self):
        return len(self.steps) >= 2 and self.steps[-1].elo > self.steps[0].elo

    def summary(self):
        lines = [f"Iteration report: {self.strategy_name}"]
        for s in self.steps:
            tag = "ACCEPT" if s.accepted else "REJECT"
            lines.append(
                f"  step {s.step}: {s.description:20s} "
                f"wr={s.win_rate:.1%}  Elo={s.elo:.0f} "
                f"(+{s.delta_elo:.0f})  [{tag}]"
            )
        lines.append(
            f"  best: v{self.best_version}  wr={self.best_win_rate:.1%}  "
            f"Elo={self.best_elo:.0f}"
        )
        lines.append(f"  improved: {self.improved}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# IterationLoop
# --------------------------------------------------------------------------- #


class IterationLoop:
    """Orchestrates the propose -> validate -> evaluate -> accept loop.

    Parameters
    ----------
    store : VersionStore
        Where to persist each accepted (and rejected) version.
    opponents : dict
        Named reference opponents used to evaluate each candidate.
    llm : LLMInterface
        The coding-agent seam.  Defaults to StaticLLM.
    n_seeds : int
        Games per opponent per side per evaluation.
    """

    def __init__(self, store, opponents, llm=None, n_seeds=4):
        self.store = store
        self.opponents = opponents
        self.llm = llm if llm is not None else StaticLLM()
        self.n_seeds = n_seeds
        self._elo = EloTracker()

    def _try_load(self, code):
        """Attempt to compile+load code; return (agent, error)."""
        try:
            import types
            mod_name = f"_candidate_{abs(hash(code))}"
            mod = types.ModuleType(mod_name)
            mod.__dict__["__file__"] = "<candidate>"
            exec(compile(code, "<candidate>", "exec"), mod.__dict__)
            if not hasattr(mod, "create_agent"):
                return None, "missing create_agent()"
            agent = mod.create_agent()
            if not hasattr(agent, "act"):
                return None, "agent has no act() method"
            return agent, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def run(self, strategy_name, n_iterations=3, verbose=True):
        """Run n_iterations rounds of the loop.

        Each round:
          1. Ask the LLM for improved code.
          2. Validate it loads and produces an agent.
          3. Evaluate it against opponents.
          4. Compare Elo to the previous best; accept or reject.
          5. Persist the version either way (rejected ones are tagged).
        """
        report = IterationReport(strategy_name=strategy_name)
        best_elo = 0.0
        best_version = None
        prev_win_rate = 0.0

        for i in range(n_iterations):
            step_num = i + 1
            prompt = self._build_prompt(strategy_name, best_elo, prev_win_rate)
            code = self.llm.generate(prompt)
            desc = getattr(self.llm, "step_name", f"iter_{step_num}")

            if verbose:
                print(f"\n--- iteration {step_num}: {desc} ---")

            agent, err = self._try_load(code)
            if agent is None:
                if verbose:
                    print(f"  REJECT: validation failed -- {err}")
                report.steps.append(IterationStep(
                    step=step_num, description=desc, accepted=False,
                    win_rate=0.0, h2h={}, elo=best_elo, delta_elo=0.0,
                    code_lines=code.count("\n") + 1, error=err,
                ))
                self.store.save(strategy_name, code, description=f"[REJECTED] {desc}",
                                parent_version=best_version)
                continue

            wr, h2h = evaluate_strategy(
                agent, self.opponents, n_seeds=self.n_seeds)
            if verbose:
                print(f"  win_rate={wr:.1%}  h2h={h2h}")

            self._elo.ratings.clear()
            self._elo.ensure("candidate")
            for opp_name, rate in h2h.items():
                self._elo.ensure(opp_name)
                self._elo.update("candidate", rate, opp_name)
            candidate_elo = self._elo.get("candidate")
            delta = candidate_elo - best_elo

            accepted = candidate_elo >= best_elo
            if accepted:
                best_elo = candidate_elo
                prev_win_rate = wr

            tag = "" if accepted else "[REJECTED] "
            meta = self.store.save(
                strategy_name, code,
                description=f"{tag}{desc}",
                parent_version=best_version,
                score=round(candidate_elo),
                score_detail={"win_rate": wr, "h2h": h2h},
            )
            if accepted:
                best_version = meta.version

            report.steps.append(IterationStep(
                step=step_num, description=desc, accepted=accepted,
                win_rate=wr, h2h=h2h, elo=round(candidate_elo),
                delta_elo=round(delta), code_lines=code.count("\n") + 1,
            ))

        report.best_version = best_version
        report.best_win_rate = prev_win_rate
        report.best_elo = best_elo
        return report

    def _build_prompt(self, name, best_elo, prev_wr):
        opp_names = list(self.opponents.keys())
        return textwrap.dedent(f"""\
            You are a coding agent improving a SnakeGo strategy.
            Current best: {name} Elo={best_elo:.0f} win_rate={prev_wr:.1%}
            Opponents: {opp_names}
            Write a complete strategy.py defining create_agent() -> BaseAgent.
            Improve territory control, item collection, and survival.
        """)


__all__ = [
    "LLMInterface", "StaticLLM",
    "IterationLoop", "IterationReport", "IterationStep",
    "evaluate_strategy",
    "TEMPLATE_V0", "TEMPLATE_V1", "TEMPLATE_V2",
]

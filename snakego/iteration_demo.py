"""Demo: full closed loop -- baseline -> iterate -> measurable improvement.

Runs the IterationLoop with StaticLLM (offline), showing:
  1. Baseline (v0) is saved and evaluated
  2. Iteration produces progressively better strategies
  3. Each version is saved to VersionStore with Elo/win-rate metadata
  4. The final version shows a clear Elo gain

    python -m snakego.iteration_demo
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .agents import RandomAgent, GreedyAgent
from .iteration import IterationLoop, StaticLLM
from .versioning import VersionStore


def main() -> int:
    store_dir = Path("outputs/snakego_iteration")
    if store_dir.exists():
        shutil.rmtree(store_dir)

    store = VersionStore(store_dir)
    print("=== Coding Agent Iteration Demo ===\n")

    # opponents: random agents with different seeds (held-out from iteration)
    opponents = {
        "random_0": RandomAgent(0),
        "random_1": RandomAgent(1),
        "random_2": RandomAgent(2),
    }

    loop = IterationLoop(
        store=store,
        opponents=opponents,
        llm=StaticLLM(),
        n_seeds=3,
    )

    report = loop.run(strategy_name="snake_ai", n_iterations=3, verbose=True)

    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    # show all saved versions
    print("\nSaved versions:")
    for v in store.list_versions("snake_ai"):
        print(f"  {v.version_id}  Elo={v.score}  {v.description}")

    # assertions: the loop must show improvement
    assert len(report.steps) == 3, "expected 3 iteration steps"
    assert report.improved, "iteration should produce measurable Elo gain"
    assert report.best_version is not None, "best version should be set"

    step0 = report.steps[0]
    step_last = report.steps[-1]
    assert step_last.elo > step0.elo, \
        f"Elo should improve: {step0.elo} -> {step_last.elo}"

    print(f"\nElo progression: {step0.elo:.0f} -> {step_last.elo:.0f} "
          f"(+{step_last.elo - step0.elo:.0f})")
    print("\nITERATION DEMO PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

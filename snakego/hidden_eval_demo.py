"""End-to-end demo: iteration loop -> hidden evaluation.

Runs the full pipeline:
  1. IterationLoop produces 3 candidate strategies, saves the best.
  2. Hidden evaluation tests the best version against unseen opponents.
  3. Report shows: training performance vs blind performance.

    python -m snakego.hidden_eval_demo
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .agents import RandomAgent
from .hidden_eval import hidden_evaluate
from .iteration import IterationLoop, StaticLLM
from .versioning import VersionStore


def main() -> int:
    store_dir = Path("outputs/snakego_hidden_eval")
    if store_dir.exists():
        shutil.rmtree(store_dir)

    store = VersionStore(store_dir)
    print("=== Full Closed-Loop Demo: Iteration + Hidden Evaluation ===\n")

    # Phase 1: Iteration loop (training opponents)
    print("--- Phase 1: Iteration Loop ---")
    train_opponents = {
        "random_0": RandomAgent(0),
        "random_1": RandomAgent(1),
        "random_2": RandomAgent(2),
    }

    loop = IterationLoop(
        store=store,
        opponents=train_opponents,
        llm=StaticLLM(),
        n_seeds=3,
    )
    report = loop.run(strategy_name="snake_ai", n_iterations=3, verbose=True)

    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    best_v = report.best_version
    assert best_v is not None, "no best version produced"

    # Phase 2: Hidden evaluation
    print("\n\n--- Phase 2: Hidden Evaluation ---")
    result = hidden_evaluate(store, "snake_ai", best_v, n_seeds=4, verbose=True)

    # Summary comparison
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    train_wr = report.best_win_rate
    blind_wr = result["win_rate"]
    print(f"  Strategy:       snake_ai/v{best_v}")
    print(f"  Training win%:  {train_wr:.1%}  (vs random opponents)")
    print(f"  Blind win%:     {blind_wr:.1%}  (vs {result['n_opponents']} unseen opponents)")
    print(f"  Elo (train):    {report.best_elo:.0f}")
    print(f"  Generalization: {'GOOD' if blind_wr > 0.4 else 'CHECK'}")

    # assertions
    assert report.improved, "iteration should improve"
    assert result["win_rate"] > 0, "blind win rate should be positive"
    assert len(result["h2h"]) >= 3, "should evaluate against multiple opponents"

    # save full result
    out = Path("outputs/snakego_hidden_eval/result.json")
    out.write_text(json.dumps({
        "iteration_report": {
            "steps": [
                {"step": s.step, "desc": s.description, "accepted": s.accepted,
                 "win_rate": s.win_rate, "elo": s.elo, "h2h": s.h2h}
                for s in report.steps
            ],
            "best_version": best_v,
            "best_elo": report.best_elo,
            "best_win_rate": train_wr,
        },
        "hidden_eval": result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Result saved to {out}")

    print("\nFULL CLOSED-LOOP DEMO PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

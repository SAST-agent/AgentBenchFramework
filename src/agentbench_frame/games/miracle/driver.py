"""Glue between finalized Miracle :class:`GameOutcome` instances and the
framework's :class:`Run`.

This module is the single place that defeats framework risks #2 / #3 / #5
(see docs/games/24_miracle_adapter_status.md):

* #2 / #3 — ``Match``/``Run`` count ``raw_winner == 0`` as a win. We feed
  ``Run.log_episode`` the **normalized** winner (0 = evaluated agent won),
  so ``Run._build_summary`` persists a win_rate that is correct even after
  side-swapping.
* #5 — ``BaseRunner.run()`` writes ``summary.json`` before merging the
  subclass result. We never use ``BaseRunner``; the caller drives ``Run``
  directly and everything is injected *before* ``finish()``.

Error games are recorded as ``game`` audit events but are NOT logged as
episodes, so they fall out of ``total_episodes`` and ``win_rate`` (有效对局).
"""
from __future__ import annotations

from typing import Dict, Sequence

from agentbench_frame.games.miracle.result import (
    DRAW,
    LOSS,
    WIN,
    GameOutcome,
    compute_h2h,
    compute_win_rate,
    to_event_record,
)
from agentbench_frame.tracking.run import Run

#: map normalized result -> the integer winner Run.log_episode expects.
#: Run counts ``winner == 0`` as a win, so 0 = evaluated-agent win.
_EPISODE_WINNER = {WIN: 0, LOSS: 1, DRAW: -1}


def feed_outcomes_to_run(run: Run, outcomes: Sequence[GameOutcome]) -> Dict:
    """Write every outcome as a ``game`` audit event and log the valid ones as
    episodes with normalized winners + auditable step counts.

    Does **not** call ``run.finish()`` — the caller owns the run lifecycle so it
    can set ``run_type`` / ``data_dir`` correctly (risk #4 / #6) before any
    summary is written.
    """
    for o in outcomes:
        run.write(**to_event_record(o))
        if o.valid:
            evaluated_score = o.score0 if o.evaluated_agent_camp == 0 else o.score1
            run.log_episode(
                reward=float(evaluated_score or 0.0),
                steps=int(o.steps),
                winner=_EPISODE_WINNER[o.normalized_result],
                info={
                    "game_id": o.game_id,
                    "opponent": o.opponent,
                    "evaluated_agent_camp": o.evaluated_agent_camp,
                    "raw_winner": o.raw_winner,
                },
            )
    run.log_h2h(compute_h2h(outcomes))
    return {
        "win_rate": compute_win_rate(outcomes),
        "valid_games": sum(1 for o in outcomes if o.valid),
    }

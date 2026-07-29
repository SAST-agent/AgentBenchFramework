"""Isolated adapter for the external 24_miracle Judge runtime.

The package wraps subprocess/runtime behavior and normalizes results without
reimplementing Judge rules in process.
"""

from agentbench_frame.games.miracle.result import (
    DRAW,
    ERROR,
    LOSS,
    VALID_RESULTS,
    WIN,
    GameOutcome,
    compute_h2h,
    compute_win_rate,
    derive_raw_winner,
    finalize,
    normalize,
    outcome_counts,
    read_replay_header,
    select_games_to_run,
    sha256_file,
    to_event_record,
    would_rerun_successful,
)

__all__ = [
    "GameOutcome",
    "WIN",
    "LOSS",
    "DRAW",
    "ERROR",
    "VALID_RESULTS",
    "derive_raw_winner",
    "finalize",
    "normalize",
    "compute_win_rate",
    "compute_h2h",
    "outcome_counts",
    "read_replay_header",
    "sha256_file",
    "select_games_to_run",
    "would_rerun_successful",
    "to_event_record",
]

"""24_miracle adapter — wraps the original external Judge + AI subprocesses
(高翔's run_match.py) and feeds normalized results into the framework's `Run`,
bypassing the framework's `Match`/`BaseRunner` whose win attribution is unsafe
under side-swapping. See docs/games/24_miracle_adapter_status.md."""

from agentbench_frame.games.miracle.result import (
    GameOutcome,
    WIN,
    LOSS,
    DRAW,
    ERROR,
    VALID_RESULTS,
    derive_raw_winner,
    finalize,
    normalize,
    compute_win_rate,
    compute_h2h,
    outcome_counts,
    read_replay_header,
    sha256_file,
    select_games_to_run,
    would_rerun_successful,
    to_event_record,
)

__all__ = [
    "GameOutcome", "WIN", "LOSS", "DRAW", "ERROR", "VALID_RESULTS",
    "derive_raw_winner", "finalize", "normalize", "compute_win_rate",
    "compute_h2h", "outcome_counts", "read_replay_header", "sha256_file",
    "select_games_to_run", "would_rerun_successful", "to_event_record",
]

"""Pure result-normalization logic for the 24_miracle adapter.

No subprocess, no framework imports — only the standard library. Fully
unit-testable. Semantics are pinned to:

* Judge ``main.py`` (judge_dev_logic): ``end_info`` format, winner derivation,
  crash/timeout handling, replay header layout.
* ``SKILL.md`` sections: 胜负归一化, h2h方向, total_steps定义,
  AgentBenchResults数据契约.

Keeping this module dependency-free means the contract tests run without the
Judge, the framework, or any external process.
"""
from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ---- normalized-result vocabulary (SKILL.md 胜负归一化) ----
WIN = "win"
LOSS = "loss"
DRAW = "draw"
ERROR = "error"
#: results that count as 有效对局 (valid games). Errors are excluded.
VALID_RESULTS: Tuple[str, ...] = (WIN, LOSS, DRAW)


# --------------------------------------------------------------------------- #
# end_info parsing  (Judge main.py:440-459)
# --------------------------------------------------------------------------- #
def derive_raw_winner(end_info: Optional[dict]) -> Optional[int]:
    """Map the Judge's terminal ``end_info`` to a camp index.

    ``end_info = {"0": player0_score, "1": player1_score}`` (main.py:457) and
    ``winner = 0 if score0 > score1 else 1`` (main.py:450). Ties are broken
    toward player1 (main.py:448-449), so a decisive 0/1 is always produced when
    the Judge finishes normally. Returns ``None`` when ``end_info`` is absent or
    malformed — i.e. the Judge crashed or the match ended catastrophically.
    """
    if not isinstance(end_info, dict) or "0" not in end_info or "1" not in end_info:
        return None
    try:
        s0, s1 = int(end_info["0"]), int(end_info["1"])
    except (TypeError, ValueError):
        return None
    return 0 if s0 > s1 else 1


def scores_from_end_info(end_info: Optional[dict]) -> Tuple[Optional[int], Optional[int]]:
    """Return (player0_score, player1_score) from end_info, (None, None) if absent."""
    if not isinstance(end_info, dict):
        return None, None
    try:
        return int(end_info.get("0")), int(end_info.get("1"))
    except (TypeError, ValueError):
        return None, None


# --------------------------------------------------------------------------- #
# per-game outcome
# --------------------------------------------------------------------------- #
@dataclass
class GameOutcome:
    """Everything the adapter knows about one played game.

    ``evaluated_agent_camp`` is the player index (0/1) the evaluated agent sat
    at for THIS game; side-swapping is expressed by flipping it across games.
    """

    game_id: str
    evaluated_agent: str
    opponent: str
    evaluated_agent_camp: int

    # verdict inputs
    raw_winner: Optional[int] = None          # camp index 0/1, or None
    score0: Optional[int] = None
    score1: Optional[int] = None
    ai_error_player: Optional[int] = None     # camp index of an AI that crashed
    ai_timeout_player: Optional[int] = None   # camp index of an AI that timed out
    judge_ok: bool = True                      # end_info present AND run_match exited 0
    replay_ok: bool = True

    # process-level signals (filled by the subprocess wrapper)
    judge_exit: Optional[int] = None
    ai0_exit: Optional[int] = None
    ai1_exit: Optional[int] = None
    exception: Optional[str] = None
    timeout_s: Optional[float] = None

    # provenance / budget
    realized_randomization: Optional[Dict[str, int]] = None  # {map_type, day_time} from replay
    steps: int = 0                                # ai_operation count == environment steps
    score_tie: bool = False                       # score0 == score1 (Judge resolves to player1)
    judge_tiebreak_applied: bool = False          # a tie was resolved to player1 by the Judge
    duration_s: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    replay_path: Optional[str] = None
    replay_sha256: Optional[str] = None
    evaluated_source_sha256: Optional[str] = None
    opponent_source_sha256: Optional[str] = None
    is_resume: bool = False
    is_rerun: bool = False

    # derived — filled by finalize()
    winner_agent: Optional[str] = None
    normalized_result: str = ERROR
    valid: bool = False
    draw: bool = False

    @property
    def opponent_camp(self) -> int:
        return 1 - self.evaluated_agent_camp

    def agent_at_camp(self, camp: int) -> str:
        return self.evaluated_agent if camp == self.evaluated_agent_camp else self.opponent


def normalize(outcome: GameOutcome) -> str:
    """Classify a game into ``{win, loss, draw, error}`` (SKILL.md 胜负归一化).

    Any AI crash, Judge crash, timeout, or missing replay => ``error`` and is
    NOT counted as a capability win (rank03's opponent crash is exactly this).
    The Judge never produces a draw (ties -> player1, main.py:448-449); DRAW is
    retained only for protocol safety.
    """
    if (not outcome.judge_ok) or (not outcome.replay_ok) or outcome.raw_winner is None:
        return ERROR
    if outcome.ai_error_player is not None or outcome.ai_timeout_player is not None:
        return ERROR
    if outcome.raw_winner not in (0, 1):
        return DRAW
    return WIN if outcome.raw_winner == outcome.evaluated_agent_camp else LOSS


def finalize(outcome: GameOutcome) -> GameOutcome:
    """Fill derived fields in place and return the outcome."""
    outcome.normalized_result = normalize(outcome)
    outcome.valid = outcome.normalized_result in VALID_RESULTS
    outcome.draw = outcome.normalized_result == DRAW
    outcome.winner_agent = (
        outcome.agent_at_camp(outcome.raw_winner) if outcome.raw_winner in (0, 1) else None
    )
    outcome.score_tie = (outcome.score0 is not None and outcome.score1 is not None
                         and outcome.score0 == outcome.score1)
    outcome.judge_tiebreak_applied = outcome.score_tie
    return outcome


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #
def compute_win_rate(outcomes: Sequence[GameOutcome]) -> float:
    """``win_rate = 有效胜局数 / 有效对局数`` (SKILL.md).

    Draws are valid games but not wins, so they stay in the denominator; error
    games are excluded entirely. Returns 0.0 when there are no valid games
    (mirrors the framework Run's ``n = max(1, 0)`` behaviour).
    """
    valid = [o for o in outcomes if o.normalized_result in VALID_RESULTS]
    if not valid:
        return 0.0
    wins = sum(1 for o in valid if o.normalized_result == WIN)
    return wins / len(valid)


def compute_h2h(outcomes: Sequence[GameOutcome]) -> Dict[str, Dict[str, float]]:
    """``h2h[row][col]`` = fraction of valid games between row and col that row
    won (SKILL.md h2h方向: 行策略战胜列策略的胜率).

    Error games and games without a decisive winner are excluded. With draws,
    ``h2h[row][col] + h2h[col][row]`` need not equal 1 (draws in denominator).
    For Miracle (Judge never draws) the two entries over a pair sum to 1.
    """
    games: Dict[Tuple[str, str], int] = {}
    row_wins: Dict[Tuple[str, str], int] = {}
    for o in outcomes:
        if o.normalized_result == ERROR:
            continue
        a0, a1 = o.agent_at_camp(0), o.agent_at_camp(1)
        # every valid game (win/loss/draw) counts toward the denominator
        for pair in ((a0, a1), (a1, a0)):
            games[pair] = games.get(pair, 0) + 1
        if o.raw_winner in (0, 1):  # draws add to denominator but not to wins
            winner = o.agent_at_camp(o.raw_winner)
            loser = a1 if winner == a0 else a0
            row_wins[(winner, loser)] = row_wins.get((winner, loser), 0) + 1
    h2h: Dict[str, Dict[str, float]] = {}
    for (row, col), n in games.items():
        h2h.setdefault(row, {})[col] = row_wins.get((row, col), 0) / n
    return h2h


def outcome_counts(outcomes: Sequence[GameOutcome]) -> Dict[str, int]:
    """Tally {win, loss, draw, error} counts (every key always present)."""
    c = Counter(o.normalized_result for o in outcomes)
    return {k: int(c.get(k, 0)) for k in (WIN, LOSS, DRAW, ERROR)}


# --------------------------------------------------------------------------- #
# run-level statistics (written into summary.json by MiracleEvalRunner)
# --------------------------------------------------------------------------- #
#: No deterministic seed support this round: the Judge draws map_type/day_time
#: via random.randint and reads no external seed. map_type/day_time are the
#: realized random environment parameters, NOT a seed.
DETERMINISTIC_SEED_SUPPORTED = False


def build_seed_provenance(realized_randomization: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """Build the per-game seed-provenance record. The Judge supports no seed, so
    requested/effective seed are null; realized_randomization records the actual
    random environment parameters (map_type/day_time) from the replay."""
    return {
        "requested_seed": None,
        "effective_seed": None,
        "deterministic_seed_supported": DETERMINISTIC_SEED_SUPPORTED,
        "reproducible_from_seed": False,
        "realized_randomization": realized_randomization,
    }


def compute_run_stats(outcomes: Sequence[GameOutcome]) -> Dict[str, Any]:
    """Run-level statistics for summary.json.

    * attempted_games  : every game whose attempt was recorded
    * valid_games      : games with a decisive, evidence-consistent result
    * win_rate_denominator == valid_games; win_rate = wins / valid_games
    * total_steps counts only valid games; attempted_steps counts all attempts
    * win_rate is None and evaluation_status is NO_VALID_GAMES when valid_games == 0
      (we never report a fabricated 0% strength conclusion).
    """
    attempted = len(outcomes)
    valid = [o for o in outcomes if o.normalized_result in VALID_RESULTS]
    valid_games = len(valid)
    wins = sum(1 for o in valid if o.normalized_result == WIN)
    losses = sum(1 for o in valid if o.normalized_result == LOSS)
    draws = sum(1 for o in valid if o.normalized_result == DRAW)
    attempted_steps = sum(int(getattr(o, "steps", 0) or 0) for o in outcomes)
    total_steps = sum(int(getattr(o, "steps", 0) or 0) for o in valid)
    return {
        "attempted_games": attempted,
        "valid_games": valid_games,
        "invalid_games": attempted - valid_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_denominator": valid_games,
        "attempted_steps": attempted_steps,
        "total_steps": total_steps,
        "win_rate": (wins / valid_games) if valid_games > 0 else None,
        "evaluation_status": ("COMPLETE" if valid_games > 0 else "NO_VALID_GAMES"),
    }


# --------------------------------------------------------------------------- #
# replay / file hashing
# --------------------------------------------------------------------------- #
def read_replay_header(path) -> Optional[Dict[str, int]]:
    """Read the actual random environment parameters from a Miracle replay file.

    The Judge writes the replay as big-endian signed int32s; the first 7 ints
    are ``[0, 0, 0, map_type, day_time, 0, 0]`` (main.py:310-313). The Judge
    draws ``map_type``/``day_time`` via ``random.randint`` (main.py:85-86) and
    reads no external seed, so these are NOT a seed — they are the realized
    random environment parameters recorded in the replay. Returns None if the
    file is missing or too short.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(28)
    except OSError:
        return None
    if len(head) < 28:
        return None
    try:
        vals = struct.unpack(">7i", head)
    except struct.error:
        return None
    return {"map_type": int(vals[3]), "day_time": int(vals[4])}


def sha256_file(path, chunk: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# resumability (SKILL.md: 可恢复但不重复成功对局)
# --------------------------------------------------------------------------- #
def select_games_to_run(planned_ids: Sequence[str],
                         completed_valid_ids: Sequence[str]) -> List[str]:
    """Return planned game ids that still need to run, i.e. those without an
    existing valid (successful) recorded result. Order preserved."""
    done = set(completed_valid_ids)
    return [gid for gid in planned_ids if gid not in done]


def would_rerun_successful(game_id: str, completed_valid_ids: Sequence[str]) -> bool:
    """True if ``game_id`` already has a valid result and would be re-run."""
    return game_id in set(completed_valid_ids)


# --------------------------------------------------------------------------- #
# event-record builder (SKILL.md events.jsonl per-game fields)
# --------------------------------------------------------------------------- #
#: fields SKILL.md requires in each per-game event. Used by tests to assert
#: the record is contract-complete.
REQUIRED_EVENT_FIELDS = (
    "game_id", "seed", "policy_ids", "policy_source_sha256", "camps",
    "raw_winner", "winner_agent", "normalized_result", "scores", "draw",
    "started_at", "finished_at", "duration", "judge_exit", "ai0_exit", "ai1_exit",
    "timeout_s", "exception", "replay_path", "replay_sha256",
    "valid", "is_resume", "is_rerun",
)


def to_event_record(o: GameOutcome) -> Dict:
    """Build the per-game event dict for ``Run.write("game", **...)``.

    Covers every SKILL.md events.jsonl per-game field; ``REQUIRED_EVENT_FIELDS``
    lists the contract keys so tests can assert completeness.
    """
    return {
        "event": "game",
        "game_id": o.game_id,
        "seed": build_seed_provenance(o.realized_randomization),
        "score_tie": o.score_tie,
        "judge_tiebreak_applied": o.judge_tiebreak_applied,
        "evaluated_agent": o.evaluated_agent,
        "opponent": o.opponent,
        "evaluated_agent_camp": o.evaluated_agent_camp,
        "policy_ids": [o.evaluated_agent, o.opponent],
        "policy_source_sha256": [o.evaluated_source_sha256, o.opponent_source_sha256],
        "camps": [0, 1],
        "raw_winner": o.raw_winner,
        "winner_agent": o.winner_agent,
        "normalized_result": o.normalized_result,
        "scores": {"0": o.score0, "1": o.score1},
        "draw": o.draw,
        "valid": o.valid,
        "started_at": o.started_at,
        "finished_at": o.finished_at,
        "duration": o.duration_s,
        "judge_exit": o.judge_exit,
        "ai0_exit": o.ai0_exit,
        "ai1_exit": o.ai1_exit,
        "timeout_s": o.timeout_s,
        "exception": o.exception,
        "ai_error_player": o.ai_error_player,
        "ai_timeout_player": o.ai_timeout_player,
        "judge_ok": o.judge_ok,
        "replay_ok": o.replay_ok,
        "steps": o.steps,
        "replay_path": o.replay_path,
        "replay_sha256": o.replay_sha256,
        "is_resume": o.is_resume,
        "is_rerun": o.is_rerun,
    }

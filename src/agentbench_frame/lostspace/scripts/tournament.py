#!/usr/bin/env python3
"""Tournament over the LostSpace human ladder -> payoff matrix + Elo ranking.

Schedules ``--matches`` 4-player FFA games sampling all 16 ranked human
algorithms (a seeded random 4-subset per match with a random seat
permutation), runs each through the official logic+AI harness
(:func:`agentbench_frame.lostspace.match.run_match`), and derives:

* a 16x16 **payoff matrix** of pairwise win-rates, and
* a pairwise **Elo ranking** (via :class:`agentbench_frame.arena.rating.EloTracker`).

Every 4-player match yields ``C(4,2)=6`` pairwise outcomes — for each pair of
seats, the seat with the higher ``end_info`` rank-points "beats" the other (a
points tie is a draw). Those outcomes feed both the payoff matrix and the Elo
tracker, so the 2-player Elo lib is reused unchanged for the 4-FFA pool.

LostSpace is 4-player FFA, so the 2-player ``arena.Arena`` does not fit; this
script is the per-game-subprocess analogue of a round-robin, mirroring
``build_ladder.py``.

Resumable: each finished match is appended to ``matches.jsonl``; re-run with
``--resume`` to skip indices already present.

Usage::

    uv run python -m agentbench_frame.lostspace.scripts.tournament --matches 2
    uv run python -m agentbench_frame.lostspace.scripts.tournament --matches 200
    uv run python -m agentbench_frame.lostspace.scripts.tournament \\
        --matches 200 --out-name pool-200 --resume
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

from agentbench_frame.arena.rating import EloTracker
from agentbench_frame.lostspace import ladder, match

# Official logic backend (loads src/mapconf2.map + `from src import main`
# relatively, so cwd MUST be this dir). Default interpreter is the one
# build_ladder.py proved works (it ships antlr4 4.9).
_BACKEND = Path(
    r"E:\HL_Agent\AgentBench\backend_sources\corpus\25_lostspace\logic\gamecode_logic"
)
_DEFAULT_LOGIC_PY = r"D:\pymol\python.exe"


def _logic_command(python_exe: str) -> str:
    return f'cd /d "{_BACKEND}" && "{python_exe}" main.py'


def _out_root() -> Path:
    env = os.environ.get("AGENTBENCH_DATA")
    base = Path(env).resolve() if env else Path.cwd() / "agentbench_data"
    return base / "tournaments" / "lostspace"


def _resolve_pool(ranks: list[int]) -> dict[int, str]:
    """rank -> harness launch command (builds C++ into the cache as a side effect)."""
    pool: dict[int, str] = {}
    for r in ranks:
        entry = ladder.entries()[r]
        pool[r] = ladder.command_for(f"rank={r}")
        print(
            f"  pool  rank{r:02d}  {entry.username:<14}  {entry.display_name:<10}  ({entry.language})",
            file=sys.stderr,
        )
    return pool


def _schedule(ranks: list[int], n_matches: int, seed: int) -> list[list[int]]:
    """N match lineups; each is a length-4 list of ranks indexed by seat 0..3."""
    rng = random.Random(seed)
    lineups: list[list[int]] = []
    for _ in range(n_matches):
        chosen = rng.sample(ranks, 4)
        rng.shuffle(chosen)  # random seat assignment -> averages seat/map bias
        lineups.append(chosen)
    return lineups


def _run_one(
    pool: dict[int, str],
    seats: list[int],
    logic_cmd: str,
    timeout: float,
    out_dir: Path,
    idx: int,
) -> dict:
    cmds = [pool[r] for r in seats]
    replay = out_dir / "replays" / f"m{idx:04d}.json"
    t0 = time.time()
    try:
        res = match.run_match(
            logic_cmd,
            cmds,
            timeout,
            replay,
            trace_path=None,
            media_player_seat=None,
        )
    except Exception as exc:  # noqa: BLE001 - record and keep going
        return {
            "idx": idx,
            "seats": seats,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.time() - t0, 2),
        }
    return {
        "idx": idx,
        "seats": seats,
        "ranking": res["ranking"],
        "end_info": res["end_info"],
        "turns": res["turns"],
        "winner": res["winner"],
        "wall_s": round(time.time() - t0, 2),
    }


def _completed_indices(matches_path: Path) -> set[int]:
    if not matches_path.is_file():
        return set()
    done: set[int] = set()
    with matches_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _aggregate(records: list[dict], ranks: list[int]) -> tuple[dict, list, dict]:
    """Build payoff matrix, Elo, and per-rank appearance/placement stats."""
    rankset = set(ranks)
    # co_occ[(a,b)] with a<b ; wins[a][b] = times a beat b (a,b any order)
    co_occ: dict[tuple[int, int], int] = {}
    wins: dict[int, dict[int, int]] = {r: {} for r in ranks}
    draws: dict[tuple[int, int], int] = {}
    appearances: dict[int, int] = {r: 0 for r in ranks}
    placement_sum: dict[int, float] = {r: 0.0 for r in ranks}

    elo = EloTracker()  # 1500 / K=32 / scale=400
    for r in ranks:
        elo.add_player(f"rank{r:02d}")

    for rec in records:
        if "error" in rec or "ranking" not in rec:
            continue
        seats: list[int] = rec["seats"]
        end_info = rec["end_info"]
        ranking: list[int] = rec["ranking"]  # seats, best->worst
        # placement per seat (1=best)
        placement: dict[int, int] = {seat: ranking.index(seat) + 1 for seat in range(4)}
        for seat in range(4):
            appearances[seats[seat]] += 1
            placement_sum[seats[seat]] += placement[seat]
        # 6 pairwise outcomes from rank-points
        pts = {seat: float(end_info[str(seat)]) for seat in range(4)}
        for i, j in itertools.combinations(range(4), 2):
            a, b = seats[i], seats[j]  # ranks
            pa, pb = pts[i], pts[j]
            key = (a, b) if a < b else (b, a)
            co_occ[key] = co_occ.get(key, 0) + 1
            if pa == pb:
                draws[key] = draws.get(key, 0) + 1
                elo.update(f"rank{a:02d}", f"rank{b:02d}", -1)
            elif pa > pb:
                wins.setdefault(a, {})[b] = wins.setdefault(a, {}).get(b, 0) + 1
                elo.update(f"rank{a:02d}", f"rank{b:02d}", 0)
            else:
                wins.setdefault(b, {})[a] = wins.setdefault(b, {}).get(a, 0) + 1
                elo.update(f"rank{a:02d}", f"rank{b:02d}", 1)

    # payoff matrix P[a][b] = (wins a-over-b + 0.5 draws) / co_occ
    payoff: dict[int, dict[int, float]] = {a: {} for a in ranks}
    for a, b in itertools.combinations(sorted(rankset), 2):
        key = (a, b)
        n = co_occ.get(key, 0)
        if n == 0:
            payoff[a][b] = float("nan")
            payoff[b][a] = float("nan")
            continue
        w_ab = wins.get(a, {}).get(b, 0)
        w_ba = wins.get(b, {}).get(a, 0)
        d = draws.get(key, 0)
        p_ab = (w_ab + 0.5 * d) / n
        payoff[a][b] = p_ab
        payoff[b][a] = 1.0 - p_ab  # wins_ba + 0.5d over n

    # Elo rows + appearance/placement
    rank_map = ladder.entries()
    rows = []
    elo_lookup = {name: (rating, gp) for name, rating, gp in elo.get_rankings()}
    for r in ranks:
        rating, pairwise_gp = elo_lookup.get(f"rank{r:02d}", (1500.0, 0))
        apps = appearances[r]
        rows.append(
            {
                "rank": r,
                "username": rank_map[r].username,
                "display_name": rank_map[r].display_name,
                "elo": round(rating, 1),
                "matches": apps,
                "pairwise_games": pairwise_gp,
                "mean_placement": round(placement_sum[r] / apps, 3) if apps else float("nan"),
            }
        )
    rows.sort(key=lambda x: x["elo"], reverse=True)
    return payoff, rows, {"co_occ": co_occ, "wins": wins, "draws": draws}


def _write_matrix(payoff: dict, rows: list[dict], out_dir: Path) -> None:
    order = [r["rank"] for r in rows]  # Elo-desc order for readability
    # CSV
    csv_path = out_dir / "payoff_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank\\vs"] + [f"r{r:02d}" for r in order])
        for a in order:
            cells = []
            for b in order:
                if a == b:
                    cells.append("")
                else:
                    v = payoff[a][b]
                    cells.append("" if v != v else f"{v:.3f}")  # NaN check
            w.writerow([f"r{a:02d}"] + cells)
    # JSON (rank-keyed)
    json_path = out_dir / "payoff_matrix.json"
    json_path.write_text(
        json.dumps(
            {
                "order": order,
                "matrix": {
                    str(a): {str(b): (None if payoff[a][b] != payoff[a][b] else round(payoff[a][b], 4))
                             for b in order if a != b}
                    for a in order
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_ranking(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "elo_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["elo_rank", "ladder_rank", "username", "display", "elo",
                    "matches", "pairwise_games", "mean_placement"])
        for i, row in enumerate(rows, 1):
            w.writerow([i, f"r{row['rank']:02d}", row["username"], row["display_name"],
                        row["elo"], row["matches"], row["pairwise_games"], row["mean_placement"]])
    json_path = out_dir / "elo_ranking.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_leaderboard(rows: list[dict], n_matches: int, errors: int) -> None:
    print("\n=== LostSpace ladder tournament — Elo ranking ===", file=sys.stderr)
    print(f"{'#':>2}  {'ladder':>6}  {'username':<14}  {'elo':>7}  {'plc':>5}  {'games':>5}",
          file=sys.stderr)
    print("-" * 52, file=sys.stderr)
    for i, row in enumerate(rows, 1):
        print(f"{i:>2}  r{row['rank']:02d}     {row['username']:<14}  {row['elo']:>7.1f}  "
              f"{row['mean_placement']:>5.2f}  {row['matches']:>5}", file=sys.stderr)
    print("-" * 52, file=sys.stderr)
    print(f"{n_matches} matches scheduled, {errors} errored.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=200, help="4-player matches to schedule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=15.0, help="per-round TLE (s)")
    parser.add_argument(
        "--rank", type=int, nargs="*", default=None,
        help="restrict pool to these ladder ranks (default: all 16; need >=4)",
    )
    parser.add_argument("--logic-python", default=_DEFAULT_LOGIC_PY,
                        help="interpreter for the official logic (must have antlr4 4.9)")
    parser.add_argument("--out-name", default=None,
                        help="run label / output dir name (default: pool-<matches>)")
    parser.add_argument("--resume", action="store_true",
                        help="skip match indices already in matches.jsonl")
    args = parser.parse_args(argv)

    ranks = sorted(args.rank) if args.rank else ladder.all_ranks()
    if len(ranks) < 4:
        parser.error(f"need at least 4 ranks in the pool, got {len(ranks)}")

    out_name = args.out_name or f"pool-{args.matches}"
    out_dir = _out_root() / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replays").mkdir(exist_ok=True)
    matches_path = out_dir / "matches.jsonl"

    print(f"Resolving pool ({len(ranks)} ranks) ...", file=sys.stderr)
    pool = _resolve_pool(ranks)
    logic_cmd = _logic_command(args.logic_python)

    schedule = _schedule(ranks, args.matches, args.seed)
    done = _completed_indices(matches_path) if args.resume else set()

    # Run
    errored = 0
    completed_records: list[dict] = []
    # load existing completed (non-error) records for final aggregation
    existing: dict[int, dict] = {}
    if matches_path.is_file():
        with matches_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing[rec.get("idx")] = rec

    wall_samples: list[float] = []
    print(f"Running {args.matches} matches -> {out_dir}", file=sys.stderr)

    with matches_path.open("a", encoding="utf-8") as mfh:
        for idx, seats in enumerate(schedule):
            if idx in done:
                rec = existing.get(idx)
                if rec and "error" not in rec:
                    completed_records.append(rec)
                continue
            rec = _run_one(pool, seats, logic_cmd, args.timeout, out_dir, idx)
            if "error" in rec:
                errored += 1
                print(f"  m{idx:04d} ERROR ({rec['wall_s']}s): {rec['error']}", file=sys.stderr)
            else:
                completed_records.append(rec)
                wall_samples.append(rec["wall_s"])
                print(f"  m{idx:04d} ok  turns={rec['turns']:<4} "
                      f"winner=p{rec['winner']}  {rec['wall_s']}s", file=sys.stderr)
            mfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mfh.flush()

    # Aggregate + write
    payoff, rows, _ = _aggregate(completed_records, ranks)
    _write_matrix(payoff, rows, out_dir)
    _write_ranking(rows, out_dir)

    summary = {
        "config": {"matches": args.matches, "seed": args.seed, "timeout": args.timeout,
                   "ranks": ranks, "logic_python": args.logic_python},
        "completed": len(completed_records),
        "errored": errored,
        "wall_per_match": (
            {"min": round(min(wall_samples), 2), "mean": round(sum(wall_samples) / len(wall_samples), 2),
             "max": round(max(wall_samples), 2), "n": len(wall_samples)}
            if wall_samples else None
        ),
        "top5": [{k: v for k, v in row.items()} for row in rows[:5]],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _print_leaderboard(rows, args.matches, errored)
    print(f"\nOutputs in {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

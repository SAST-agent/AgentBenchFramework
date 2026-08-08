#!/usr/bin/env python3
"""Label-generic LostSpace tournament over an arbitrary AI pool.

Reads a ``pool.json`` = ``{label: launch_command}`` (e.g. produced by
``submissions_pool.py`` for the full 251-player anonymized corpus), schedules N
seeded 4-FFA matches sampling the pool (random 4-subset + random seat
permutation), runs each via :func:`match.run_match`, and derives a pairwise
payoff matrix + Elo ranking — same method as ``tournament.py`` but with
opaque string labels instead of ladder ranks.

With a large pool (241 players, ~29 000 pairs) the matrix is necessarily
**sparse**: a feasible match budget gives each pair few or zero direct
meetings, so the Elo ranking is driven by transitive win-chains rather than
dense head-to-head. More matches = sharper Elo, linearly more hours.

Resumable: each finished match appends to ``matches.jsonl``; ``--resume`` skips
indices already present.

Usage::

    uv run python -m agentbench_frame.lostspace.scripts.pool_tournament \\
        --pool-file <pool.json> --matches 6000 --timeout 15 --out-name pool-251-6000
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
from agentbench_frame.lostspace import match

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


def _load_pool(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"pool.json must be a non-empty {{label: command}} map: {path}")
    return {str(k): str(v) for k, v in data.items()}


def _schedule(labels: list[str], n_matches: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    out: list[list[str]] = []
    for _ in range(n_matches):
        if len(labels) < 4:
            raise SystemExit("pool has fewer than 4 players")
        chosen = rng.sample(labels, 4)
        rng.shuffle(chosen)
        out.append(chosen)
    return out


def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants (Windows: taskkill /T)."""
    import subprocess as _sp
    try:
        _sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except OSError:
        pass


def _run_one(pool, seats, logic_cmd, timeout, out_dir, idx, wall=300):
    """Run match `idx` in an isolated subprocess with a hard wall-clock cap.

    A few seeded 4-subsets deadlock in a way the per-read TLE doesn't catch
    (a logic-side stall with no pending action request). Running each match in
    its own worker process means such a hang is bounded: at `wall` seconds the
    worker + its logic/AI children are killed and the match is recorded as a
    wall_timeout error, so the tournament advances instead of hanging forever.
    """
    import subprocess as _sp
    cmds = [pool[s] for s in seats]
    replay = out_dir / "replays" / f"m{idx:05d}.json"
    result_path = out_dir / f".result_{idx:05d}.json"
    job_path = out_dir / f".job_{idx:05d}.json"
    job_path.write_text(json.dumps({
        "logic_cmd": logic_cmd, "ai_cmds": cmds, "timeout": timeout,
        "replay_path": str(replay), "result_path": str(result_path),
    }, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    proc = None
    try:
        proc = _sp.Popen(
            [sys.executable, "-m",
             "agentbench_frame.lostspace.scripts.pool_tournament_worker",
             str(job_path)],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        )
        proc.wait(timeout=wall)
    except _sp.TimeoutExpired:
        if proc is not None:
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except _sp.TimeoutExpired:
                pass
        return {"idx": idx, "seats": seats,
                "error": f"wall_timeout({wall}s)",
                "wall_s": round(time.time() - t0, 2)}
    finally:
        job_path.unlink(missing_ok=True)
    if not result_path.is_file():
        return {"idx": idx, "seats": seats, "error": "worker_no_result",
                "wall_s": round(time.time() - t0, 2)}
    out = json.loads(result_path.read_text(encoding="utf-8"))
    result_path.unlink(missing_ok=True)
    rec = {"idx": idx, "seats": seats,
           "wall_s": out.get("wall_s", round(time.time() - t0, 2))}
    if out.get("ok"):
        rec.update({"ranking": out["ranking"], "end_info": out["end_info"],
                    "turns": out["turns"], "winner": out["winner"]})
    else:
        rec["error"] = out.get("error", "worker_error")
    return rec


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


def _aggregate(records, labels):
    co_occ: dict[tuple[str, str], int] = {}
    wins: dict[str, dict[str, int]] = {s: {} for s in labels}
    draws: dict[tuple[str, str], int] = {}
    apps: dict[str, int] = {s: 0 for s in labels}
    plc_sum: dict[str, float] = {s: 0.0 for s in labels}

    elo = EloTracker()
    for s in labels:
        elo.add_player(s)

    for rec in records:
        if "error" in rec or "ranking" not in rec:
            continue
        seats: list[str] = rec["seats"]
        end_info = rec["end_info"]
        ranking: list[int] = rec["ranking"]
        placement = {seat: ranking.index(seat) + 1 for seat in range(4)}
        for seat in range(4):
            apps[seats[seat]] += 1
            plc_sum[seats[seat]] += placement[seat]
        pts = {seat: float(end_info[str(seat)]) for seat in range(4)}
        for i, j in itertools.combinations(range(4), 2):
            a, b = seats[i], seats[j]
            pa, pb = pts[i], pts[j]
            key = (a, b) if a < b else (b, a)
            co_occ[key] = co_occ.get(key, 0) + 1
            if pa == pb:
                draws[key] = draws.get(key, 0) + 1
                elo.update(a, b, -1)
            elif pa > pb:
                wins.setdefault(a, {})[b] = wins.setdefault(a, {}).get(b, 0) + 1
                elo.update(a, b, 0)
            else:
                wins.setdefault(b, {})[a] = wins.setdefault(b, {}).get(a, 0) + 1
                elo.update(a, b, 1)

    payoff = {a: {} for a in labels}
    for a, b in itertools.combinations(sorted(labels), 2):
        n = co_occ.get((a, b), 0)
        if n == 0:
            payoff[a][b] = float("nan")
            payoff[b][a] = float("nan")
            continue
        w_ab = wins.get(a, {}).get(b, 0)
        d = draws.get((a, b), 0)
        p_ab = (w_ab + 0.5 * d) / n
        payoff[a][b] = p_ab
        payoff[b][a] = 1.0 - p_ab

    elo_lookup = {name: (rating, gp) for name, rating, gp in elo.get_rankings()}
    rows = []
    for s in labels:
        rating, pgp = elo_lookup.get(s, (1500.0, 0))
        a = apps[s]
        rows.append({"label": s, "elo": round(rating, 1), "matches": a,
                     "pairwise_games": pgp,
                     "mean_placement": round(plc_sum[s] / a, 3) if a else float("nan")})
    rows.sort(key=lambda x: x["elo"], reverse=True)
    return payoff, rows, co_occ


def _write_matrix(payoff, rows, out_dir):
    order = [r["label"] for r in rows]
    with (out_dir / "payoff_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label\\vs"] + order)
        for a in order:
            cells = []
            for b in order:
                if a == b:
                    cells.append("")
                else:
                    v = payoff[a][b]
                    cells.append("" if v != v else f"{v:.3f}")
            w.writerow([a] + cells)
    # sparse JSON: only cells with co-occurrence
    (out_dir / "payoff_matrix.json").write_text(
        json.dumps({"order": order}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_ranking(rows, co_occ, out_dir):
    n_pairs = sum(1 for v in co_occ.values() if v > 0)
    with (out_dir / "elo_ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["elo_rank", "label", "elo", "matches", "pairwise_games",
                    "mean_placement"])
        for i, row in enumerate(rows, 1):
            w.writerow([i, row["label"], row["elo"], row["matches"],
                        row["pairwise_games"], row["mean_placement"]])
    (out_dir / "elo_ranking.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return n_pairs


def _print_leaderboard(rows, n_matches, errors, n_pairs, pool_size):
    print("\n=== LostSpace pool tournament — Elo (top 25) ===", file=sys.stderr)
    print(f"{'#':>3}  {'label':<20}  {'elo':>7}  {'plc':>5}  {'games':>5}",
          file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    for i, row in enumerate(rows[:25], 1):
        print(f"{i:>3}  {row['label']:<20}  {row['elo']:>7.1f}  "
              f"{row['mean_placement']:>5.2f}  {row['matches']:>5}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    print(f"pool={pool_size}  matches={n_matches}  errored={errors}  "
          f"distinct_pairs_observed={n_pairs}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-file", required=True, help="pool.json = {label: command}")
    p.add_argument("--matches", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--logic-python", default=_DEFAULT_LOGIC_PY)
    p.add_argument("--out-name", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    pool = _load_pool(Path(args.pool_file))
    labels = sorted(pool.keys())
    if len(labels) < 4:
        raise SystemExit(f"pool has {len(labels)} players; need >=4")

    out_name = args.out_name or f"pool-{len(labels)}-{args.matches}"
    out_dir = _out_root() / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replays").mkdir(exist_ok=True)
    matches_path = out_dir / "matches.jsonl"
    logic_cmd = _logic_command(args.logic_python)

    schedule = _schedule(labels, args.matches, args.seed)
    done = _completed_indices(matches_path) if args.resume else set()
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

    errored = 0
    completed: list[dict] = []
    wall_samples: list[float] = []
    print(f"Pool {len(labels)} players. Running {args.matches} matches -> {out_dir}",
          file=sys.stderr)
    with matches_path.open("a", encoding="utf-8") as mfh:
        for idx, seats in enumerate(schedule):
            if idx in done:
                rec = existing.get(idx)
                if rec and "error" not in rec:
                    completed.append(rec)
                continue
            rec = _run_one(pool, seats, logic_cmd, args.timeout, out_dir, idx)
            if "error" in rec:
                errored += 1
                if errored <= 20 or errored % 50 == 0:
                    print(f"  m{idx:05d} ERROR ({rec['wall_s']}s): {rec['error'][:120]}",
                          file=sys.stderr)
            else:
                completed.append(rec)
                wall_samples.append(rec["wall_s"])
            mfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mfh.flush()
            if (idx + 1) % 100 == 0 or idx + 1 == len(schedule):
                avg = sum(wall_samples) / len(wall_samples) if wall_samples else 0
                eta = avg * (len(schedule) - (idx + 1))
                print(f"  [{idx+1}/{len(schedule)}] err={errored} "
                      f"avg={avg:.1f}s eta~{eta/60:.0f}min", file=sys.stderr)

    payoff, rows, co_occ = _aggregate(completed, labels)
    _write_matrix(payoff, rows, out_dir)
    n_pairs = _write_ranking(rows, co_occ, out_dir)

    summary = {
        "config": {"matches": args.matches, "seed": args.seed, "timeout": args.timeout,
                   "pool_file": str(args.pool_file), "pool_size": len(labels)},
        "completed": len(completed), "errored": errored,
        "distinct_pairs_observed": n_pairs,
        "wall_per_match": (
            {"min": round(min(wall_samples), 2),
             "mean": round(sum(wall_samples) / len(wall_samples), 2),
             "max": round(max(wall_samples), 2), "n": len(wall_samples)}
            if wall_samples else None),
        "top10": rows[:10],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _print_leaderboard(rows, args.matches, errored, n_pairs, len(labels))
    print(f"\nOutputs in {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

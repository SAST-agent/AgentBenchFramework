#!/usr/bin/env python3
"""24_miracle A-plan 32-game formal matrix runner (single process).

Binds match_runner + matrix.py + matrix_runner + Framework Run/events/summary.
Run under ONE Python 3.11.15 ``uv run`` process (constant env). Modes:
  --dry-run : create session + manifest + verify hashes + print plan; NO game/subprocess.
  (default) : verify, then execute() the 32 attempts (per-rank audit, infra-stop),
              then write Run-compatible output + aggregate.

Opponent runnable dirs (frozen builds, NOT recompiled):
  rank04/05/07 (Python): protected extracted dirs
  rank01/02/03/06/08-15 (C++): .smoke/precheck/20260721-184652_da899d/strategies/rankNN
  rank16 (C++): .smoke/rank16build/20260721-195738_baff71/rank16_copy
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.matrix_runner import MatrixRunner  # noqa: E402

from agentbench_frame.games.miracle.paths import judge_dir, ifelse_dir, extracted_dir, archives_dir
JUDGE = judge_dir()
IFELSE = ifelse_dir()
EXTRACTED = extracted_dir()
ARCHIVES = archives_dir()
VENDOR = REPO / "vendor" / "miracle_local" / "run_match.py"
FW_SRC = REPO / "src"
PROTOCOL = REPO / "docs" / "games" / "24_miracle_evaluation_protocol.v0.3.json"
ROSTER = REPO / "docs" / "games" / "24_miracle_roster_manifest.json"
PRECHECK_9A = REPO / ".smoke" / "precheck" / "20260721-184652_da899d"
RANK16_BUILD = REPO / ".smoke" / "rank16build" / "20260721-195738_baff71"
SESSION_ROOT = REPO / ".smoke" / "matrix"
PROTOCOL_SHA = "866696fd9e094da85e3f2c04dc5ba20d0500faf8461531a242362323c4efe0b3"
AUTH_TEXT = "用户授权 A 方案 32 局正式矩阵（v0.3）；逐对手分批审计；成功局不重跑；infra 即停；完成后聚合+本地 Results 验收；不 push/不上传。"

PYTHON_RANKS = (4, 5, 7)
CPP_RANKS = (1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16)
LOG = None


def log(msg=""):
    print(msg, flush=True)
    if LOG:
        LOG.write(str(msg) + "\n"); LOG.flush()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def opponent_dir_of(rank: int) -> Path:
    if rank in PYTHON_RANKS:
        return next(EXTRACTED.glob(f"rank{rank:02d}__*"))
    if rank == 16:
        return RANK16_BUILD / "rank16_copy"
    return PRECHECK_9A / "strategies" / f"rank{rank:02d}"


def verify_hashes(v3: dict, roster: dict):
    """Verify every opponent's runnable identity matches the frozen hashes."""
    mismatches = []
    ba = v3["frozen_identities"]["build_artifacts_win64_mingw"]
    for rank in range(1, 17):
        d = opponent_dir_of(rank)
        if not d.exists():
            mismatches.append(f"rank{rank:02d}: opponent dir missing {d}")
            continue
        if rank in CPP_RANKS:
            me = d / "main.exe"
            if not me.exists():
                mismatches.append(f"rank{rank:02d}: main.exe missing")
            else:
                got = sha(me)
                want = ba[f"rank{rank:02d}"]
                if got != want:
                    mismatches.append(f"rank{rank:02d}: main.exe sha {got} != frozen {want}")
    if sha(IFELSE / "main.py") != v3["frozen_identities"]["evaluated_agent"]["sha256"]:
        mismatches.append("ifelse sha mismatch")
    if sha(JUDGE / "main.py") != v3["frozen_identities"]["judge"]["main_py_sha256"]:
        mismatches.append("judge sha mismatch")
    return mismatches


def main() -> int:
    global LOG
    dry = "--dry-run" in sys.argv
    v3 = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    if sha(PROTOCOL) != PROTOCOL_SHA:
        print(f"FATAL: protocol v0.3 sha mismatch (expected {PROTOCOL_SHA})", file=sys.stderr)
        return 2

    r = MatrixRunner(
        session_root=SESSION_ROOT, judge_dir=JUDGE, ifelse_dir=IFELSE,
        opponent_dir_of=opponent_dir_of, vendor_script=VENDOR, framework_src=FW_SRC,
        timeout=8.0, wrapper_timeout_s=180.0, protocol_sha=PROTOCOL_SHA,
        python=sys.executable, evaluated_agent="miracle_ifelse", auth_text=AUTH_TEXT,
    )
    r.prepare_session()
    LOG = open(r.session_dir / "matrix.full.log", "w", encoding="utf-8")
    log(f"session_id: {r.session_id}")
    log(f"run_id: {r.run_id}")
    log(f"python: {sys.executable} ({platform.python_version()})")
    log(f"protocol v0.3 sha: {PROTOCOL_SHA} (verified)")
    log(f"mode: {'DRY_RUN' if dry else 'EXECUTE'}")

    mismatches = verify_hashes(v3, roster)
    opp_arch = {s["rank"]: s["archive_sha256"] for s in roster["strategies"]}
    cpp_build = {int(k.replace("rank", "")): v for k, v in v3["frozen_identities"]["build_artifacts_win64_mingw"].items()
                 if k.startswith("rank") and isinstance(v, str) and "_" not in k}
    r.record_manifest(
        opponent_hashes=opp_arch, build_hashes=cpp_build,
        ifelse_sha=sha(IFELSE / "main.py"), judge_sha=sha(JUDGE / "main.py"),
        code_hashes={"matrix": sha(REPO / "src/agentbench_frame/games/miracle/matrix.py"),
                     "matrix_runner": sha(REPO / "src/agentbench_frame/games/miracle/matrix_runner.py"),
                     "match_runner": sha(REPO / "src/agentbench_frame/games/miracle/match_runner.py"),
                     "vendor_run_match": sha(VENDOR)},
    )
    log(f"manifest: {r.session_dir / 'manifest.json'}")
    if mismatches:
        log("FATAL: hash mismatches -> STOP before any game:")
        for m in mismatches:
            log("  - " + m)
        return 2
    log("hash verification: ALL_MATCH")

    if dry:
        out = r.dry_run()
        log(f"plan: {out['plan_count']} attempts; first={out['plan'][0]['game_id']} last={out['plan'][-1]['game_id']}")
        log(f"timeout={r.timeout} wrapper_timeout_s={r.wrapper_timeout_s}")
        log("DRY_RUN complete — no game/subprocess started.")
        return 0

    log("\n===== EXECUTE 32 attempts (single process) =====")
    t0 = time.time()
    result = r.execute()
    log(f"execute result: {json.dumps({k: v for k, v in result.items() if k != 'record'}, ensure_ascii=False)}")
    log(f"elapsed: {round(time.time() - t0, 1)}s")
    run_dir = r.write_run_compatible_output()
    log(f"Run-compatible output: {run_dir}")
    agg = r.aggregate_from_events()
    log(f"aggregate: total={agg['total_attempts']} valid={agg['valid_games']} invalid={agg['invalid_games']} "
        f"wins={agg['wins']} losses={agg['losses']} win_rate={agg['win_rate']}")
    (r.session_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"aggregate.json: {r.session_dir / 'aggregate.json'}")
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

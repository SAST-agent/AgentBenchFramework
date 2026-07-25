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
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.matrix_runner import (  # noqa: E402
    MatrixRunner,
    verify_session_for_resume,
)
from agentbench_frame.games.miracle.matrix import make_attempt_plan  # noqa: E402

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


class PreflightError(RuntimeError):
    """A stable, user-facing input validation failure."""


@dataclass(frozen=True)
class ControlInputs:
    protocol: dict
    roster: dict
    hashes: dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 24_miracle evaluation matrix")
    parser.add_argument("--dry-run", action="store_true", help="plan only; do not start a game")
    parser.add_argument("--resume", metavar="SESSION_ID", help="resume an existing session")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--roster", type=Path, default=ROSTER)
    parser.add_argument("--protocol-sha", default=None)
    return parser.parse_args(argv)


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


def _load_json_object(path: Path, label: str) -> dict:
    if not path.exists():
        raise PreflightError(f"{label} file missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"{label} JSON invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} JSON must be an object: {path}")
    return value


def load_control_inputs(
    protocol_path: Path,
    roster_path: Path,
    expected_protocol_sha: str | None = None,
) -> ControlInputs:
    protocol_path = Path(protocol_path)
    roster_path = Path(roster_path)
    protocol = _load_json_object(protocol_path, "protocol")
    roster = _load_json_object(roster_path, "roster")
    strategies = roster.get("strategies")
    if not isinstance(strategies, list):
        raise PreflightError("roster strategies must be a list")
    ranks = [item.get("rank") for item in strategies if isinstance(item, dict)]
    if ranks != list(range(1, 17)):
        raise PreflightError(f"roster ranks must be exactly 1..16: {ranks}")
    protocol_hash = sha(protocol_path)
    if expected_protocol_sha and protocol_hash != expected_protocol_sha:
        raise PreflightError(
            f"protocol sha mismatch: got {protocol_hash}, expected {expected_protocol_sha}"
        )
    return ControlInputs(
        protocol=protocol,
        roster=roster,
        hashes={"protocol": protocol_hash, "roster": sha(roster_path)},
    )


def resolve_unique_dir(root: Path, pattern: str) -> Path:
    root = Path(root)
    matches = sorted(path for path in root.glob(pattern) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"no opponent directory matching {pattern}: {root}")
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"multiple opponent directories matching {pattern}: {names}")
    return matches[0]


def resolve_unique_file(root: Path, pattern: str) -> Path:
    root = Path(root)
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"no archive file matching {pattern}: {root}")
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"multiple archive files matching {pattern}: {names}")
    return matches[0]


def resolve_opponent_dir(
    rank: int,
    roster: dict | None = None,
    *,
    extracted_root: Path = EXTRACTED,
    precheck_root: Path = PRECHECK_9A,
    rank16_build_root: Path = RANK16_BUILD,
) -> Path:
    if rank in PYTHON_RANKS:
        return resolve_unique_dir(extracted_root, f"rank{rank:02d}__*")
    if rank == 16:
        return Path(rank16_build_root) / "rank16_copy"
    return Path(precheck_root) / "strategies" / f"rank{rank:02d}"


def opponent_dir_of(rank: int) -> Path:
    return resolve_opponent_dir(rank)


def verify_python_strategy_hashes(
    strategy: dict,
    extracted_root: Path,
    archives_root: Path | None = None,
) -> list[str]:
    rank = strategy.get("rank", "unknown")
    label = f"rank{int(rank):02d}" if isinstance(rank, int) else f"rank{rank}"
    errors: list[str] = []
    try:
        directory = resolve_unique_dir(extracted_root, f"rank{int(rank):02d}__*")
    except (FileNotFoundError, RuntimeError) as exc:
        return [f"{label}: {exc}"]
    entry = strategy.get("entry")
    if not isinstance(entry, str) or not entry or Path(entry).is_absolute() or ".." in Path(entry).parts:
        return [f"{label}: invalid runnable entry"]
    entry_path = directory / entry
    if not entry_path.is_file():
        errors.append(f"{label}: runnable entry missing: {entry_path}")
    else:
        expected_entry_sha = strategy.get("runnable_sha256")
        if not expected_entry_sha:
            errors.append(f"{label}: runnable sha missing from roster")
        elif sha(entry_path) != expected_entry_sha:
            errors.append(f"{label}: runnable sha mismatch")
    if archives_root is not None:
        try:
            archive = resolve_unique_file(Path(archives_root), f"rank{int(rank):02d}__*.zip")
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"{label}: {exc}")
        else:
            expected_archive_sha = strategy.get("archive_sha256")
            if expected_archive_sha and sha(archive) != expected_archive_sha:
                errors.append(f"{label}: archive sha mismatch")
    return errors


def verify_hashes(
    v3: dict,
    roster: dict,
    *,
    extracted_root: Path = EXTRACTED,
    archives_root: Path = ARCHIVES,
    precheck_root: Path = PRECHECK_9A,
    rank16_build_root: Path = RANK16_BUILD,
):
    """Verify every opponent's runnable identity matches the frozen hashes."""
    mismatches = []
    ba = v3["frozen_identities"]["build_artifacts_win64_mingw"]
    strategies = {item["rank"]: item for item in roster.get("strategies", []) if isinstance(item, dict)}
    for rank in range(1, 17):
        try:
            d = resolve_opponent_dir(
                rank,
                roster,
                extracted_root=extracted_root,
                precheck_root=precheck_root,
                rank16_build_root=rank16_build_root,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            mismatches.append(f"rank{rank:02d}: {exc}")
            continue
        if not d.exists():
            mismatches.append(f"rank{rank:02d}: opponent dir missing {d}")
            continue
        if rank in PYTHON_RANKS:
            strategy = strategies.get(rank)
            if strategy is None:
                mismatches.append(f"rank{rank:02d}: roster strategy missing")
            else:
                mismatches.extend(verify_python_strategy_hashes(strategy, extracted_root, archives_root))
        else:
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


def _run_resume(r, resume_sid):
    """Resume an existing session. VERIFY FIRST (read-only); only if ALL checks
    pass, call resume() + write. Verification failure leaves session untouched."""
    global LOG
    sd = SESSION_ROOT / resume_sid
    if not sd.exists():
        print(f"FATAL: session not found: {sd}", file=sys.stderr)
        return 2
    # READ-ONLY verification — load protocol + roster for full identity
    v3 = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    ba = v3["frozen_identities"]["build_artifacts_win64_mingw"]
    ok, errs = verify_session_for_resume(
        sd,
        protocol_sha=PROTOCOL_SHA,
        code_files={
            "matrix": str(REPO / "src/agentbench_frame/games/miracle/matrix.py"),
            "matrix_runner": str(REPO / "src/agentbench_frame/games/miracle/matrix_runner.py"),
            "match_runner": str(REPO / "src/agentbench_frame/games/miracle/match_runner.py"),
            "vendor_run_match": str(VENDOR),
        },
        expected_plan=make_attempt_plan(),
        expected_ifelse_sha=v3["frozen_identities"]["evaluated_agent"]["sha256"],
        expected_judge_sha=v3["frozen_identities"]["judge"]["main_py_sha256"],
        expected_opponent_shas={s["rank"]: s["archive_sha256"] for s in roster["strategies"]},
        expected_build_shas={int(k.replace("rank", "")): v for k, v in ba.items()
                             if k.startswith("rank") and isinstance(v, str) and "_" not in k},
        expected_platform=platform.platform(),
    )
    if not ok:
        print("FATAL: resume verification failed:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2
    # only now: open session + write
    r.resume(resume_sid)
    LOG = open(r.session_dir / "matrix.full.log", "a", encoding="utf-8")
    log(f"RESUME session: {resume_sid} (verified)")
    result = r.execute()
    log(f"resume result: {json.dumps({k: v for k, v in result.items() if k != 'record'}, ensure_ascii=False)}")
    r.write_run_compatible_output()
    agg = r.aggregate_from_events()
    log(f"aggregate: total={agg['total_attempts']} valid={agg['valid_games']} invalid={agg['invalid_games']} win_rate={agg['win_rate']}")
    return 0 if result.get("completed") else 1


def main() -> int:
    global LOG
    dry = "--dry-run" in sys.argv
    # parse --resume <session_id>
    resume_sid = None
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        resume_sid = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not resume_sid:
            print("FATAL: --resume requires a session_id", file=sys.stderr)
            return 2
    if dry and resume_sid:
        print("FATAL: --resume and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        resume_sid = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not resume_sid:
            print("FATAL: --resume requires a session_id", file=sys.stderr)
            return 2
        r = MatrixRunner(
            session_root=SESSION_ROOT, judge_dir=JUDGE, ifelse_dir=IFELSE,
            opponent_dir_of=opponent_dir_of, vendor_script=VENDOR, framework_src=FW_SRC,
            timeout=8.0, wrapper_timeout_s=180.0, protocol_sha=PROTOCOL_SHA,
            python=sys.executable, evaluated_agent="miracle_ifelse", auth_text=AUTH_TEXT,
        )
        return _run_resume(r, resume_sid)
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

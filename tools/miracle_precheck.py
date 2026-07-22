#!/usr/bin/env python3
"""24_miracle 阶段9A 预检：隔离编译 13 个 C++ 策略 + 无对局启动预检。

绝不启动 Judge / server / 真实对局。对每个 C++ 策略：
  1. 从受保护 extracted 源目录复制到唯一全新 session 目录（不动原策略）。
  2. 用策略自带 makefile 原样编译（不改逻辑/优化）。
  3. 记录源哈希、编译命令、完整 stdout/stderr/exit、产物类型/大小/SHA256、PE/DLL 静态检查。
  4. 用 entry.resolve_ai_command 构造命令（与正式 runner 一致），短时 Popen（stdin=DEVNULL，
     不接 Judge），按 PID+create_time 精确清理。
  5. 分类：COMPILE_PASS / OS_SPAWN_PASS / COMMAND_RESOLUTION_PASS / PROTOCOL_NOT_VALIDATED。

无 Judge 启动 → 协议/比赛可用性未验证（PROTOCOL_NOT_VALIDATED），不得表述为"策略已成功完成比赛"。
rank03 同样编译记录，但保留"运行时崩溃风险"（编译/启动成功 ≠ 比赛可用）。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.entry import resolve_ai_command  # noqa: E402
from agentbench_frame.games.miracle.proctree import ProcessTreeManager  # noqa: E402
from agentbench_frame.games.miracle.smoke_audit import (  # noqa: E402
    ensure_fresh_session, make_session_id, write_manifest_atomic,
)

from agentbench_frame.games.miracle.paths import extracted_dir
EXTRACTED = extracted_dir()
ROSTER = REPO / "docs" / "games" / "24_miracle_roster_manifest.json"
PRECHECK_ROOT = REPO / ".smoke" / "precheck"
LOG = None


def log(msg=""):
    print(msg, flush=True)
    if LOG:
        LOG.write(str(msg) + "\n"); LOG.flush()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_pe(p: Path) -> bool:
    try:
        return p.read_bytes()[:2] == b"MZ"
    except OSError:
        return False


def dll_deps(p: Path):
    """Static DLL-dependency check via objdump (no execution of the AI)."""
    try:
        r = subprocess.run(["objdump", "-p", str(p)], capture_output=True, text=True, timeout=15)
        deps = [ln.split(":", 1)[1].strip() for ln in r.stdout.splitlines()
                if ln.strip().startswith("DLL Name")]
        return deps
    except Exception as e:
        return [f"<objdump failed: {e}>"]


def find_binary(d: Path):
    for name in ("main.exe", "main"):
        if (d / name).exists():
            return d / name
    return None


def cpp_strategies():
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    out = []
    for s in roster["strategies"]:
        if s.get("type") == "cpp_source":
            rank = s["rank"]
            cand = list(EXTRACTED.glob(f"rank{rank:02d}__*"))
            out.append({"rank": rank, "name": s["entity"], "extracted": cand[0] if cand else None,
                        "archive_sha256": s["archive_sha256"], "invalid_now": s.get("invalid_now", False)})
    return out


def compile_one(copy_dir: Path):
    """Run the strategy's own makefile. Record everything."""
    t0 = time.time()
    cmd = ["make"]
    try:
        r = subprocess.run(cmd, cwd=str(copy_dir), capture_output=True, text=True, timeout=180)
        return {"command": cmd, "stdout": r.stdout, "stderr": r.stderr,
                "returncode": r.returncode, "duration_s": round(time.time() - t0, 2),
                "exception": None}
    except subprocess.TimeoutExpired as e:
        return {"command": cmd, "stdout": e.stdout or "", "stderr": e.stderr or "",
                "returncode": None, "duration_s": round(time.time() - t0, 2),
                "exception": "timeout"}
    except Exception as e:
        return {"command": cmd, "stdout": "", "stderr": str(e),
                "returncode": None, "duration_s": round(time.time() - t0, 2),
                "exception": repr(e)}


def spawn_check(copy_dir: Path, timeout_s=2.0):
    """Short isolated process-spawn check. NO Judge. stdin=DEVNULL so the AI
    sees EOF and exits fast; we only verify the OS can create the process."""
    result = {"command_resolution": None, "command_resolution_pass": False,
              "os_spawn_pass": None, "returncode": None, "stderr_head": None,
              "cleanup_all_succeeded": None, "exception": None, "protocol_validated": False}
    # 1. command resolution (same path the real runner uses)
    try:
        cmd = resolve_ai_command(copy_dir)
        result["command_resolution"] = cmd
        result["command_resolution_pass"] = True
    except FileNotFoundError as e:
        result["command_resolution"] = f"FileNotFoundError: {e}"
        result["command_resolution_pass"] = False
        return result
    # 2. OS spawn (NO judge; stdin closed)
    mgr = ProcessTreeManager()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(cmd, cwd=str(copy_dir), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=creationflags)
        mgr.register_popen(proc, f"ai_spawn")
        result["os_spawn_pass"] = True
        try:
            proc.communicate(timeout=timeout_s)
            result["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            result["returncode"] = "timeout"
    except (OSError, FileNotFoundError) as e:
        result["os_spawn_pass"] = False
        result["exception"] = repr(e)
        return result
    finally:
        # precise PID + create_time cleanup
        procs = mgr.cleanup_all("spawn-check")
        result["cleanup_all_succeeded"] = all(p.cleanup_succeeded for p in procs)
        result["cleanup_detail"] = mgr.status()  # list[dict], JSON-serializable
    # capture a short stderr head if any
    return result


def main() -> int:
    global LOG
    session_id = make_session_id()
    session_dir = ensure_fresh_session(PRECHECK_ROOT, session_id)
    LOG = open(session_dir / "precheck.full.log", "w", encoding="utf-8")
    log(f"precheck session: {session_dir}")
    log(f"compiler: g++ {subprocess.run(['g++','--version'],capture_output=True,text=True).stdout.splitlines()[0]}")
    log(f"make: {subprocess.run(['make','--version'],capture_output=True,text=True).stdout.splitlines()[0]}")
    log("NOTE: 无 Judge / 无对局启动；仅编译 + 短时进程创建预检。PROTOCOL_NOT_VALIDATED。")

    targets = cpp_strategies()
    reports = []
    for t in targets:
        rank = t["rank"]; name = t["name"]
        log(f"\n===== rank{rank:02d} ({name}) =====")
        if t["extracted"] is None:
            log(f"  EXTRACTED_DIR_MISSING"); reports.append({"rank": rank, "status": "MISSING_SOURCE"}); continue
        copy_dir = session_dir / "strategies" / f"rank{rank:02d}"
        shutil.copytree(t["extracted"], copy_dir)
        # source hashes (record, do not modify)
        src_files = sorted(p for p in copy_dir.iterdir() if p.suffix in (".cpp", ".c", ".h", ".hpp", ".json") or p.name == "makefile")
        source_hashes = {p.name: sha256_file(p) for p in src_files}
        log(f"  copied {len(src_files)} source files from {t['extracted'].name}")
        # compile
        comp = compile_one(copy_dir)
        binary = find_binary(copy_dir)
        compile_pass = (comp["returncode"] == 0 and binary is not None)
        log(f"  compile: returncode={comp['returncode']} binary={'main.exe' if binary and binary.name=='main.exe' else binary.name if binary else None} -> {'COMPILE_PASS' if compile_pass else 'COMPILE_FAIL'}")
        if comp["stderr"]:
            log(f"  [stderr tail] {comp['stderr'][-300:]!r}")
        bin_info = None
        if binary:
            bin_info = {"name": binary.name, "size": binary.stat().st_size,
                        "sha256": sha256_file(binary), "is_pe": is_pe(binary),
                        "dll_deps": dll_deps(binary)}
            log(f"  binary: size={bin_info['size']} pe={bin_info['is_pe']} sha256={bin_info['sha256'][:16]}… dlls={bin_info['dll_deps']}")
        # spawn pre-check (only if compiled)
        spawn = None
        if compile_pass:
            spawn = spawn_check(copy_dir)
            log(f"  spawn: cmd_res={spawn['command_resolution_pass']} os_spawn={spawn['os_spawn_pass']} rc={spawn['returncode']} cleanup={spawn['cleanup_all_succeeded']}")
        report = {
            "rank": rank, "name": name, "invalid_now": t["invalid_now"],
            "source_dir_original": str(t["extracted"]),
            "source_dir_copy": str(copy_dir),
            "archive_sha256": t["archive_sha256"],
            "source_hashes": source_hashes,
            "compile": comp, "binary": bin_info,
            "compile_pass": compile_pass,
            "spawn_precheck": spawn,
            "classification": {
                "COMPILE_PASS": compile_pass,
                "OS_SPAWN_PASS": bool(spawn and spawn["os_spawn_pass"]),
                "COMMAND_RESOLUTION_PASS": bool(spawn and spawn["command_resolution_pass"]),
                "PROTOCOL_NOT_VALIDATED": True,
            },
            "verification_note": "仅静态编译 + 短时进程创建预检；未启动 Judge/对局，不证明协议或比赛可用。"
                                 + (" rank03 历史运行时崩溃风险保留，编译/启动成功不等于比赛可用。" if rank == 3 else ""),
        }
        reports.append(report)
        (session_dir / "reports" ).mkdir(exist_ok=True)
        (session_dir / "reports" / f"rank{rank:02d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "session_id": session_id, "kind": "stage9A_precheck",
        "created_unix": time.time(), "python": sys.executable,
        "python_version": platform.python_version(),
        "compiler": "g++ 15.2.0 (MinGW)", "make": "GNU Make 4.4.1",
        "no_judge_no_match": True,
        "strategy_count": len(reports),
        "summary": {
            "compile_pass": sum(1 for r in reports if r.get("compile_pass")),
            "compile_fail": sum(1 for r in reports if not r.get("compile_pass") and r.get("status") != "MISSING_SOURCE"),
            "missing_source": sum(1 for r in reports if r.get("status") == "MISSING_SOURCE"),
        },
    }
    write_manifest_atomic(session_dir, manifest)
    (session_dir / "summary.json").write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n===== SUMMARY =====")
    log(f"compile_pass={manifest['summary']['compile_pass']}/{len(reports)}  compile_fail={manifest['summary']['compile_fail']}  missing={manifest['summary']['missing_source']}")
    for r in reports:
        c = r.get("classification", {})
        log(f"  rank{r['rank']:02d}: compile={c.get('COMPILE_PASS')} os_spawn={c.get('OS_SPAWN_PASS')} cmd_res={c.get('COMMAND_RESOLUTION_PASS')} protocol=NOT_VALIDATED{' [invalid_now]' if r.get('invalid_now') else ''}")
    log(f"\nsession dir: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

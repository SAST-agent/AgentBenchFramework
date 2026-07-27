#!/usr/bin/env python3
"""阶段9B：rank16 隔离构建修复 + 无对局启动预检。

用户决定：允许只在 rank16 的隔离构建副本中预先创建空 ``build/`` 目录（构建环境
准备，非策略修复）。不改源码 / Makefile / 编译参数 / 优化。

步骤：
  1. 全新唯一 session；从受保护 extracted 源复制 rank16。
  2. 记录编译前全部源文件 + Makefile SHA256。
  3. 仅在副本中创建 Makefile 预期的空 build/。
  4. 用原 Makefile 编译一次。
  5. 编译后再次核对源文件 + Makefile 哈希 == 编译前（必须一致）。
  6. 记录新增文件清单（仅构建产物 + 空目录准备）。
  7. 若成功：PE/大小/SHA256/依赖 + 绝对路径命令解析 + 短时无 Judge 启动 + PID+create_time 清理。
不删除此前 rank16 失败证据。不启动 Judge / 真实对局。
"""
from __future__ import annotations

import hashlib
import json
import os
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
RANK16_SRC = next(EXTRACTED.glob("rank16__*"))
SESSION_ROOT = REPO / ".smoke" / "rank16build"
LOG = None

SOURCE_SUFFIXES = (".cpp", ".c", ".h", ".hpp", ".json")


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
    try:
        r = subprocess.run(["objdump", "-p", str(p)], capture_output=True, text=True, timeout=15)
        return [ln.split(":", 1)[1].strip() for ln in r.stdout.splitlines()
                if ln.strip().startswith("DLL Name")]
    except Exception as e:
        return [f"<objdump failed: {e}>"]


def source_hashes(d: Path):
    """Hash strategy SOURCE files only (not build/ artifacts, not main exe)."""
    out = {}
    for p in sorted(d.iterdir()):
        if p.is_dir():
            continue
        if p.suffix in SOURCE_SUFFIXES or p.name == "makefile":
            out[p.name] = sha256_file(p)
    return out


def spawn_check(copy_dir: Path, timeout_s=2.0):
    res = {"command_resolution": None, "command_resolution_pass": False,
           "os_spawn_pass": None, "returncode": None, "cleanup_all_succeeded": None,
           "exception": None}
    try:
        cmd = resolve_ai_command(copy_dir)
        res["command_resolution"] = cmd
        res["command_resolution_pass"] = True
    except FileNotFoundError as e:
        res["command_resolution"] = f"FileNotFoundError: {e}"
        return res
    mgr = ProcessTreeManager()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(cmd, cwd=str(copy_dir), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=creationflags)
        mgr.register_popen(proc, "rank16_spawn")
        res["os_spawn_pass"] = True
        try:
            proc.communicate(timeout=timeout_s)
            res["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            res["returncode"] = "timeout"
    except (OSError, FileNotFoundError) as e:
        res["os_spawn_pass"] = False
        res["exception"] = repr(e)
        return res
    finally:
        procs = mgr.cleanup_all("spawn-check")
        res["cleanup_all_succeeded"] = all(p.cleanup_succeeded for p in procs)
        res["cleanup_detail"] = mgr.status()
    return res


def main() -> int:
    global LOG
    session_id = make_session_id()
    session_dir = ensure_fresh_session(SESSION_ROOT, session_id)
    LOG = open(session_dir / "rank16_build.full.log", "w", encoding="utf-8")
    log(f"rank16 build session: {session_dir}")
    log(f"source (protected, read-only): {RANK16_SRC}")
    log("用户决定：仅在隔离副本创建空 build/（构建环境准备，非策略修复）；不改源码/Makefile。")

    copy_dir = session_dir / "rank16_copy"
    shutil.copytree(RANK16_SRC, copy_dir)
    log(f"copied to: {copy_dir}")

    hashes_before = source_hashes(copy_dir)
    log(f"source files before: {len(hashes_before)} files hashed")

    # build-env prep ONLY in the copy: empty build/ dir the Makefile assumes
    (copy_dir / "build").mkdir(exist_ok=False)
    log("created empty build/ in copy (Makefile assumes it exists)")

    files_before_make = {p.name for p in copy_dir.iterdir() if p.is_file()} | {"build"}

    # compile with the ORIGINAL makefile (no parameter/optimization change)
    cmd = ["make"]
    t0 = time.time()
    comp = subprocess.run(cmd, cwd=str(copy_dir), capture_output=True, text=True, timeout=180)
    comp_rec = {"command": cmd, "stdout": comp.stdout, "stderr": comp.stderr,
                "returncode": comp.returncode, "duration_s": round(time.time() - t0, 2)}

    hashes_after = source_hashes(copy_dir)
    source_integrity_ok = (hashes_before == hashes_after)

    # new files after make (build artifacts + exe only)
    files_after = set()
    for p in copy_dir.rglob("*"):
        if p.is_file():
            files_after.add(str(p.relative_to(copy_dir)).replace("\\", "/"))
    files_before_set = set()
    for p in copy_dir.rglob("*"):
        pass
    # compute new files = files present now that are NOT original source files
    original_source_names = set(hashes_before.keys()) | {"Data.json"} if "Data.json" in hashes_before else set(hashes_before.keys())
    new_files = sorted(f for f in files_after if Path(f).name not in hashes_before)

    binary = None
    for name in ("main.exe", "main"):
        if (copy_dir / name).exists():
            binary = copy_dir / name
            break
    compile_pass = (comp_rec["returncode"] == 0 and binary is not None)

    log(f"compile: returncode={comp_rec['returncode']} binary={binary.name if binary else None} -> {'COMPILE_PASS' if compile_pass else 'COMPILE_FAIL'}")
    log(f"source_integrity (before==after): {source_integrity_ok}")
    log(f"new files after make: {new_files}")
    if comp_rec["stderr"]:
        log(f"[stderr tail] {comp_rec['stderr'][-400:]!r}")

    binary_info = None
    spawn = None
    if binary:
        binary_info = {"name": binary.name, "size": binary.stat().st_size,
                       "sha256": sha256_file(binary), "is_pe": is_pe(binary),
                       "dll_deps": dll_deps(binary)}
        log(f"binary: size={binary_info['size']} pe={binary_info['is_pe']} sha256={binary_info['sha256']} dlls={binary_info['dll_deps']}")
    if compile_pass:
        spawn = spawn_check(copy_dir)
        log(f"spawn: cmd_res={spawn['command_resolution_pass']} os_spawn={spawn['os_spawn_pass']} rc={spawn['returncode']} cleanup={spawn['cleanup_all_succeeded']}")

    report = {
        "session_id": session_id,
        "rank": 16,
        "source_dir_original": str(RANK16_SRC),
        "source_dir_copy": str(copy_dir),
        "build_env_prep": "created empty build/ in copy ONLY (Makefile assumes it); no source/Makefile/param/optimization change",
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_integrity_ok": source_integrity_ok,
        "new_files_after_make": new_files,
        "make": comp_rec,
        "binary": binary_info,
        "compile_pass": compile_pass,
        "spawn_precheck": spawn,
        "classification": {
            "COMPILE_PASS": compile_pass,
            "OS_SPAWN_PASS": bool(spawn and spawn["os_spawn_pass"]),
            "COMMAND_RESOLUTION_PASS": bool(spawn and spawn["command_resolution_pass"]),
            "PROTOCOL_NOT_VALIDATED": True,
        },
        "verification_note": "仅隔离编译（副本内建空 build/）+ 短时无 Judge 进程创建预检；未启动 Judge/对局，不证明协议或比赛可用。",
        "prior_failure_evidence_preserved": ".smoke/precheck/20260721-184652_da899d/ (rank16 COMPILE_FAIL 历史保留)",
    }
    (session_dir / "rank16_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "session_id": session_id, "kind": "stage9B_rank16_isolated_build",
        "created_unix": time.time(), "python": sys.executable,
        "compiler": subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "make": subprocess.run(["make", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "compile_pass": compile_pass,
        "source_integrity_ok": source_integrity_ok,
        "no_source_or_makefile_modification": source_integrity_ok,
    }
    write_manifest_atomic(session_dir, manifest)
    log(f"\nRESULT: compile_pass={compile_pass} source_integrity_ok={source_integrity_ok}")
    log(f"session dir: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

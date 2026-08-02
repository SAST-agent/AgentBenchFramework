# Plan: HL loop timeout-reaping + probe hard-wall timeout

## Problem
Real 10-act run `hl-run-0730` hung and was terminated at act 2:
1. **`ClaudeCodeRunner`** hit the 600s `--claude-timeout`; `subprocess.run` killed
   the direct child but **leaked a grandchild `claude.exe`** (PID 41580 survived
   and had to be `taskkill`'d by hand). The tree was not reaped.
2. **`ReferenceProbe.probe_one`** — the FIRST real probe (act 1 skips it because
   `version_before is None`, controller.py:230) — hung for ~11 min with zero
   progress. Root cause: `probe_one`'s read loop is deadline-bounded, but the two
   `_write_frame` calls (probe.py:169 in `_start`, :215 in `probe_one`) do
   **blocking `stream.flush()` with no timeout guard**. If the candidate doesn't
   drain stdin, `flush()` blocks forever, past the deadline, with no watchdog.

## Fix 1 — probe hard-wall timeout (`hl/probe.py`)
Wrap `probe_one`'s body in a worker thread joined with a hard `self.timeout`:
- `probe_one(sample)` spawns a daemon worker running `_probe_one_impl(sample)`
  (the current body), `worker.join(self.timeout)`.
- If the worker finishes → return its result (happy path; echo/random candidates
  reply in <1s → zero overhead).
- If still alive → **force-kill** the candidate process (`proc.kill()`), which
  unblocks the worker's stuck I/O (write `flush` → BrokenPipe, read → EOF), then
  `worker.join(_KILL_GRACE=5.0)` to guarantee the worker exits before
  `probe_one` returns. Return `None` (missing emission — already the contract
  for an unresponsive candidate).
- Rationale for worker-thread over a fire-and-forget watchdog: each sample's
  worker is fully joined before the next `probe_one` runs, so there is **no race
  on `self._proc` across samples** (a lingering watchdog could otherwise kill the
  next sample's freshly-spawned process). The candidate is a single Python
  process, so `proc.kill()` suffices (no tree-kill needed here, unlike claude).
- Bounds every sample at `self.timeout + _KILL_GRACE` regardless of WHICH call
  blocks (write flush, read, close) — robust to the exact hang path.

## Fix 2 — `ClaudeCodeRunner` reaps the whole process tree (`hl/runner.py`)
Switch `run()` from `subprocess.run` to `Popen` + `communicate(timeout=...)` so
the child PID is available on timeout (needed to kill the tree):
- Launch with `start_new_session=True` (Unix → own process group) /
  `CREATE_NEW_PROCESS_GROUP` (Windows).
- On `TimeoutExpired`: `_kill_tree(proc)` — Windows `taskkill /F /T /PID`,
  fallback `proc.kill()`; Unix `os.killpg(getpgid, SIGKILL)` — then a final
  `communicate(timeout=10)` to drain pipes + reap, then return
  `failure_reason="claude CLI timed out"` (unchanged stable string).
- All other paths (FileNotFoundError, nonzero exit, success + parse `result`
  event for usage/session_id) behavior-preserving.

## Tests (TDD — written first)
- `tests/hl/test_probe.py`: add `test_probe_hard_timeout_on_write_blocked_candidate`
  — a candidate that spawns but never reads stdin, with a >4KB observation so
  `_write_frame`'s flush blocks past the read deadline. Assert `probe_one`
  returns `None` within a hard wall-clock bound (<10s), not infinity. This is
  the direct reproduction of the act-2 production hang.
- `tests/hl/test_runner_claude.py`: update the existing `subprocess.run`
  monkeypatches to `subprocess.Popen` (the runner now uses `Popen`). Convert
  `_FakeProc` to expose `communicate(timeout)`, `pid`, `kill`, `returncode`.
  Add `test_timeout_kills_process_tree` asserting `kill()` is invoked on
  timeout (the leak fix) AND `failure_reason == "claude CLI timed out"`.

## Verification
- `pytest tests/hl/test_probe.py tests/hl/test_runner_claude.py -q`
- `pytest tests/hl -q` (whole HL suite — ensure the worker-thread refactor
  didn't regress the echo/random/dead/out-of-support probe contracts).
- Re-launch the 10-act run (`run_hl_round.sh` shape) and confirm it survives
  past act 2 (the previous hang point) without a stuck probe / leaked claude.

## Scope / notes
- Two files: `src/agentbench_frame/hl/probe.py`, `src/agentbench_frame/hl/runner.py`
  + their two test files. No spec/contract change (failure_reason strings,
  event schemas, return values all preserved).
- Branch `liuzhuo/lostspace` only. Commit after tests green.
- Lightweight OpenSpec: this is a bugfix to unblock the run the operator asked
  for; full propose/spec-delta ceremony skipped per the direct "implement the
  fixes" instruction. TDD (tests first) is preserved per the workflow's core.

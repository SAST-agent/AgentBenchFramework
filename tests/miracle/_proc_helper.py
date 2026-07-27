"""Harmless helper process for the process-tree tests. NOT the real Judge.

Modes (argv[1]):
  stay <seconds>          sleep <seconds> then exit 0 (or until killed)
  stay_forever            sleep until killed
  spawn_child <c_secs> <p_secs>
                          spawn `stay <c_secs>` as a detached child, print
                          "READY spawn_child parent=<pid> child=<pid>" (flush),
                          then sleep <p_secs> and exit

The tests use these to build real parent/child trees without touching the
Miracle Judge or any AI binary.
"""
import os
import subprocess
import sys
import time


def _main() -> int:
    mode = sys.argv[1]
    py = sys.executable
    me = os.path.abspath(__file__)

    if mode == "stay":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        print(f"READY stay pid={os.getpid()} secs={secs}", flush=True)
        time.sleep(secs)
        return 0

    if mode == "stay_forever":
        print(f"READY stay_forever pid={os.getpid()}", flush=True)
        while True:
            time.sleep(3600)
        return 0

    if mode == "spawn_child":
        child_secs = float(sys.argv[2])
        parent_secs = float(sys.argv[3])
        child = subprocess.Popen(
            [py, me, "stay", str(child_secs)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(
            f"READY spawn_child parent={os.getpid()} child={child.pid}",
            flush=True,
        )
        time.sleep(parent_secs)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())

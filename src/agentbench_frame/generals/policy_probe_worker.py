"""One-shot subprocess worker for a frozen Generals ``main.agent``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from .engine import OfficialGeneralsEngine


def _import_policy(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "agentbench_frozen_generals_policy_main",
        source / "main.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen policy from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise RuntimeError("frozen main.py does not define callable agent")
    return module


def _run(request: dict[str, Any]) -> dict[str, Any]:
    source = Path(request["source"]).resolve()
    engine_root = Path(request["engine_root"]).resolve()
    sdk_root = Path(request["sdk_root"]).resolve()
    for path in (engine_root, sdk_root, source):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    replay_path = (
        Path(tempfile.mkdtemp(prefix="generals-policy-probe-"))
        / "official.jsonl"
    )
    engine = OfficialGeneralsEngine.from_measurement_state(
        engine_root,
        request["measurement_state"],
        replay_path,
    )
    policy = _import_policy(source)
    raw = policy.agent(
        int(request["round"]),
        int(request["seat"]),
        engine.state,
    )
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("frozen agent must return a command sequence")
    commands = []
    for command in raw:
        if (
            not isinstance(command, (list, tuple))
            or not command
            or not all(type(item) is int for item in command)
        ):
            raise RuntimeError(
                "frozen agent returned a malformed command"
            )
        commands.append([int(item) for item in command])
    if commands and commands[-1] == [8]:
        raise RuntimeError(
            "frozen SDK agent unexpectedly returned its own end marker"
        )
    commands.append([8])
    return {"status": "complete", "commands": commands}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        response = _run(request)
    except Exception as exc:
        response = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    sys.stdout.write(
        json.dumps(response, sort_keys=True, ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

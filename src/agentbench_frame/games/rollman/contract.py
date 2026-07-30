"""Machine-checkable contract extracted from the frozen Rollman backend."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import operator
import subprocess
from pathlib import Path
from typing import Any


BACKEND_RELATIVE = Path(
    "backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic"
)


def asset_path(name: str) -> Path:
    return Path(__file__).with_name("assets") / name


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _LiteralEnvironment:
    _BINARY = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.evaluate(element) for element in node.elts]
        if isinstance(node, ast.Name):
            return self.values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            return self._BINARY[type(node.op)](
                self.evaluate(node.left), self.evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.evaluate(node.operand)
        raise ValueError(f"unsupported contract expression: {ast.dump(node)}")

    def load(self, source: Path) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        enums: dict[str, dict[str, int]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        self.values[target.id] = self.evaluate(node.value)
                    except (KeyError, ValueError, TypeError):
                        continue
            elif isinstance(node, ast.ClassDef):
                members: dict[str, int] = {}
                for child in node.body:
                    if (
                        isinstance(child, ast.Assign)
                        and len(child.targets) == 1
                        and isinstance(child.targets[0], ast.Name)
                    ):
                        try:
                            value = self.evaluate(child.value)
                        except (KeyError, ValueError, TypeError):
                            continue
                        if isinstance(value, int):
                            members[child.targets[0].id] = value
                if members:
                    enums[node.name] = members
        return self.values, enums


@dataclasses.dataclass(frozen=True)
class RollmanContract:
    backend_root: Path
    map_sizes: tuple[int, int, int]
    max_rounds: tuple[int, int, int]
    skill_durations: tuple[int, int, int, int, int]
    portal_activation_rounds: tuple[int, int]
    directions: dict[str, int]
    spaces: dict[str, int]
    events: dict[str, int]
    score_constants: dict[str, float]

    @classmethod
    def from_agentbench(cls, agentbench_root: str | Path) -> "RollmanContract":
        backend = Path(agentbench_root) / BACKEND_RELATIVE
        source = backend / "core" / "gamedata.py"
        if not source.is_file():
            raise FileNotFoundError(source)
        values, enums = _LiteralEnvironment().load(source)
        score_names = (
            "PACMAN_HUGE_BONUS_THRESHOLD",
            "PACMAN_HUGE_BONUS",
            "EATEN_BY_GHOST",
            "EAT_ALL_BEANS",
            "ROUND_BONUS_GAMMA",
            "GHOST_HUGE_BONUS_THRESHOLD",
            "GHOST_HUGE_BONUS",
            "PREVENT_PACMAN_EAT_ALL_BEANS",
            "EAT_PACMAN",
            "DESTORY_PACMAN_SHIELD",
        )
        return cls(
            backend_root=backend,
            map_sizes=tuple(values["INITIAL_BOARD_SIZE"][1:4]),
            max_rounds=tuple(values["MAX_ROUND"][1:4]),
            skill_durations=tuple(values["DEFAULT_SKILL_TIME"]),
            portal_activation_rounds=tuple(values["PORTAL_AVAILABLE"][1:3]),
            directions=dict(enums["Direction"]),
            spaces=dict(enums["Space"]),
            events=dict(enums["Event"]),
            score_constants={name: values[name] for name in score_names},
        )


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _seed_is_applied(main_path: Path) -> bool:
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "seed":
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id in {"random", "np"}:
            return True
        if (
            isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "np"
            and node.func.value.attr == "random"
        ):
            return True
    return False


def audit_sources(
    agentbench_root: str | Path,
    official_logic_root: str | Path,
) -> dict[str, Any]:
    agentbench = Path(agentbench_root)
    official = Path(official_logic_root)
    backend = agentbench / BACKEND_RELATIVE
    manifest = json.loads(asset_path("source_manifest.json").read_text(encoding="utf-8"))
    comparisons = {}
    for relative in manifest["core_files"]:
        frozen = backend / relative
        upstream = official / relative
        comparisons[relative] = {
            "frozen_sha256": sha256_file(frozen),
            "official_sha256": sha256_file(upstream),
            "match": sha256_file(frozen) == sha256_file(upstream),
        }
    seed_applied = _seed_is_applied(backend / "main.py")
    return {
        "schema_version": "1.0",
        "frozen_backend_commit": _git_head(agentbench),
        "official_logic_commit": _git_head(official),
        "official_core_commit": _git_head(official / "core"),
        "core_files": comparisons,
        "all_core_files_match": all(value["match"] for value in comparisons.values()),
        "random_seed_is_applied": seed_applied,
        "adapter_seed_injection_required": not seed_applied,
    }

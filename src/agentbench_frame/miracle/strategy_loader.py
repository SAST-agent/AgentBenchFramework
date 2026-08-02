"""不可变策略源码快照及 CandidateAgent 加载。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .agent_bridge import MiracleAgent


class StrategyValidationError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


def save_source(path: Path | str, source: str) -> None:
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") == source:
            return
        raise StrategyValidationError("snapshot", f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def load_candidate(path: Path | str, module_key: str) -> MiracleAgent:
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    try:
        compile(source, str(path), "exec")
    except (SyntaxError, ValueError) as exc:
        raise StrategyValidationError("compile", str(exc)) from exc
    try:
        spec = importlib.util.spec_from_file_location(module_key, path)
        if spec is None or spec.loader is None:
            raise ImportError("could not create module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_key, None)
        raise StrategyValidationError("import", repr(exc)) from exc
    candidate_class = getattr(module, "CandidateAgent", None)
    if not isinstance(candidate_class, type) or not issubclass(candidate_class, MiracleAgent):
        raise StrategyValidationError(
            "class", "source must define CandidateAgent(MiracleAgent)",
        )
    try:
        return candidate_class()
    except Exception as exc:
        raise StrategyValidationError("instantiate", repr(exc)) from exc


def validate_candidate(agent: MiracleAgent) -> None:
    try:
        cards = agent.choose_cards(0)
    except Exception as exc:
        raise StrategyValidationError("choose_cards", repr(exc)) from exc
    if not isinstance(cards, dict):
        raise StrategyValidationError("choose_cards", "result must be an object")
    artifacts = cards.get("artifacts")
    creatures = cards.get("creatures")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise StrategyValidationError("choose_cards", "exactly one artifact is required")
    if not isinstance(creatures, list) or len(creatures) != 3:
        raise StrategyValidationError("choose_cards", "exactly three creatures are required")
    obs = {
        "camp": 0,
        "round": 1,
        "map": {"units": [], "miracles": [30, 30], "barracks": [2, 2, 2, 2]},
        "players": [[[], 0, 1, [], []], [[], 0, 1, [], []]],
    }
    try:
        action = agent.act(obs)
    except Exception as exc:
        raise StrategyValidationError("action_shape", repr(exc)) from exc
    if not isinstance(action, dict):
        raise StrategyValidationError("action_shape", "action must be an object")
    if not isinstance(action.get("operation_type"), str):
        raise StrategyValidationError("action_shape", "operation_type must be a string")
    if not isinstance(action.get("operation_parameters"), dict):
        raise StrategyValidationError("action_shape", "operation_parameters must be an object")

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentbench_frame.env._antwar2.backend.engine import DEFAULT_MOVEMENT_POLICY
from agentbench_frame.env._antwar2.backend.state import BackendState, PythonBackendState


class EngineBackend(Protocol):
    name: str

    def initial_state(
        self,
        seed: int = 0,
        movement_policy: str = DEFAULT_MOVEMENT_POLICY,
        cold_handle_rule_illegal: bool = False,
    ) -> BackendState: ...


@dataclass(slots=True)
class PythonBackend:
    name: str = "python"

    def initial_state(
        self,
        seed: int = 0,
        movement_policy: str = DEFAULT_MOVEMENT_POLICY,
        cold_handle_rule_illegal: bool = False,
    ) -> BackendState:
        return PythonBackendState.initial(
            seed=seed,
            movement_policy=movement_policy,
            cold_handle_rule_illegal=cold_handle_rule_illegal,
        )


class NativeBackendUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class NativeBackend:
    module: object
    name: str = "native"

    def initial_state(
        self,
        seed: int = 0,
        movement_policy: str = DEFAULT_MOVEMENT_POLICY,
        cold_handle_rule_illegal: bool = False,
    ) -> BackendState:
        from agentbench_frame.env._antwar2.native_adapter import NativeGameStateAdapter

        return NativeGameStateAdapter.initial(
            seed,
            movement_policy=movement_policy,
            cold_handle_rule_illegal=cold_handle_rule_illegal,
        )


def load_backend(prefer_native: bool = False) -> EngineBackend:
    if not prefer_native:
        return PythonBackend()
    try:
        from agentbench_frame.env._antwar2 import native_antwar  # type: ignore
    except Exception as exc:  # pragma: no cover - optional acceleration path
        raise NativeBackendUnavailable(str(exc)) from exc
    return NativeBackend(native_antwar)

"""
TrackedEnv and TimedAgent wrappers for transparent observability.

- TrackedEnv wraps env.step() to emit StepRecord on every step.
- TimedAgent wraps agent.act() to time decision latency.

Neither wrapper changes the original API — they are drop-in replacements
that emit tracking events as a side effect.
"""

import time
from typing import Any, Dict, Optional, Tuple

from agentbench_frame.tracking.records import StepRecord


class TrackedEnv:
    """Wraps an environment, emitting StepRecord on each step().

    Usage:
        env = GeneralsEnv()
        tracked = TrackedEnv(env, on_step=writer.write)
        obs, reward, done, info = tracked.step(action)
    """

    def __init__(self, env, on_step=None):
        self._env = env
        self._on_step = on_step
        self._episode = 0
        self._step = 0

    def reset(self, *args, **kwargs):
        self._step = 0
        return self._env.reset(*args, **kwargs)

    def step(self, action: Any):
        start = time.time()
        result = self._env.step(action)
        wall_ms = (time.time() - start) * 1000
        self._step += 1

        obs, reward, done, info = result
        if self._on_step is not None:
            record = StepRecord(
                timestamp=time.time(),
                episode=self._episode,
                step=self._step,
                action=action,
                reward=float(reward),
                done=done,
                info=info or {},
                wall_time_ms=wall_ms,
                agent_time_ms=0.0,
            )
            try:
                self._on_step(record)
            except Exception:
                pass

        if done:
            self._episode += 1
            self._step = 0

        return result

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def episode(self) -> int:
        return self._episode


class TimedAgent:
    """Wraps an agent, timing act() calls.

    Usage:
        agent = RuleBasedAgent()
        timed = TimedAgent(agent, on_act=writer.write)
        action = timed.act(obs)
    """

    def __init__(self, agent, on_act=None):
        self._agent = agent
        self._on_act = on_act
        self._total_calls = 0
        self._total_time_ms = 0.0

    def act(self, observation: Any) -> Any:
        start = time.time()
        action = self._agent.act(observation)
        elapsed_ms = (time.time() - start) * 1000
        self._total_calls += 1
        self._total_time_ms += elapsed_ms

        if self._on_act is not None:
            try:
                self._on_act({
                    "event": "agent_act",
                    "timestamp": time.time(),
                    "agent_name": getattr(self._agent, "name", "unknown"),
                    "time_ms": elapsed_ms,
                })
            except Exception:
                pass

        return action

    def __getattr__(self, name):
        return getattr(self._agent, name)

    @property
    def avg_time_ms(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_time_ms / self._total_calls

    def reset(self):
        self._agent.reset()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self.avg_time_ms,
        }

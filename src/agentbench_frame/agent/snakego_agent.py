"""SnakeGo agent adapters for the framework."""

import os, sys

from agentbench_frame.agent.base import BaseAgent

_SNAKEGO_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _SNAKEGO_DIR not in sys.path:
    sys.path.insert(0, _SNAKEGO_DIR)

from snakego.strategy_core import Weights, make_decide, decide as _decide


class SnakeGoAgent(BaseAgent):
    """In-process SnakeGo agent wrapping make_decide(weights).

    Reads the live Engine object from the observation dict and calls
    decide() to produce an op code 1-6.
    """

    def __init__(self, name="SnakeGoAgent", weights=None, decide_fn=None):
        super().__init__(name=name)
        if decide_fn is not None:
            self._decide = decide_fn
        elif weights is not None:
            self._decide = make_decide(weights)
        else:
            self._decide = make_decide(Weights())
        self.metadata["weights_class"] = type(weights).__name__ if weights else "default"

    def act(self, observation):
        eng = observation.get("state", {}).get("engine")
        if eng is None:
            return 1
        my_id = observation.get("player_id", eng.current_player)
        if not eng.alive_player():
            return 1
        op = int(self._decide(eng, my_id))
        if not (1 <= op <= 6):
            op = 1
        return op

    def reset(self):
        pass


class SnakeGoSubprocessAgent(BaseAgent):
    """Agent that reads ops from a subprocess (compiled human AI).

    The env manages the socket lifecycle (init/ack/relay). This agent
    only reads 5-byte ops from the socket and returns the op type.
    """

    def __init__(self, name, socket_player):
        super().__init__(name=name)
        self.sp = socket_player

    def act(self, observation):
        data = self.sp.read_op(timeout=12.0)
        return data[4]

    def reset(self):
        pass

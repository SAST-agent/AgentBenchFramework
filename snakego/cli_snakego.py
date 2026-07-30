# CLI wrapper that registers SnakeGo agents before calling agentbench CLI.
# Usage:
#   python cli_snakego.py arena --game 26_snakego --agents snakego_rule,snakego_random --n-games 2
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FW_SRC = os.path.join(_HERE, "AgentBenchFramework", "src")
_DESKTOP_FW = r"C:\Users\53125\Desktop\snakego\AgentBenchFramework\src"
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _FW_SRC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SNAKEGO_ROOT", _HERE)

from agentbench_frame.agent.registry import AgentRegistry
from agentbench_frame.agent.base import RandomAgent

try:
    from agentbench_frame.agent.snakego_agent import SnakeGoAgent
    from snakego.strategy_core import Weights
    AgentRegistry.register("snakego_rule", SnakeGoAgent)
    AgentRegistry.register("snakego_rule_v0", lambda **kw: SnakeGoAgent(
        name="rule_v0", weights=Weights()))
    AgentRegistry.register("snakego_random", lambda **kw: SnakeGoAgent(
        name="snakego_random", decide_fn=lambda eng, pid: __import__("random").randint(1, 4)))
except Exception as e:
    print("Warning: SnakeGoAgent registration failed:", e, file=sys.stderr)

from agentbench_frame.cli import main
main()

"""24_miracle（神迹之战）harness：官方逻辑零改动运行 + 规则策略对战 + 迭代闭环。

包结构：
- ``official_logic/``：官方对战逻辑原样移植（零改动，子进程运行）
- ``protocol.py`` / ``logic_runner.py``：官方线协议与子进程运行器
- ``host.py`` / ``match.py``：评测机（对局驱动、trace 记录）
- ``agent_bridge.py``：Agent 接口与内置策略
- ``decision_space.py`` / ``ig.py``：决策空间与信息增益
- ``replay.py``：官方二进制回放解析
- ``loop.py`` / ``versions.py`` / ``curves.py``：迭代闭环编排
"""

from .agent_bridge import EndRoundAgent, MiracleAgent, SampleAgent
from .host import MatchResult, MiracleHost
from .match import run_match

__version__ = "0.1.0"

__all__ = [
    "MiracleAgent",
    "EndRoundAgent",
    "SampleAgent",
    "MiracleHost",
    "MatchResult",
    "run_match",
]

"""
MCP (Model Context Protocol) Tools for AgentBench.

Provides MCP-compatible tools that can be called by LLM agents or
external systems to interact with the AgentBench framework.

Tools follow the MCP specification:
- name, description, inputSchema
- call() method that returns structured results

Integration with skills:
- Skills can be exposed as MCP tools via `as_mcp_tool_definition()`
- The MCPServer can serve both built-in tools and registered skills
"""

from agentbench_frame.mcp.base import MCPTool, MCPToolRegistry
from agentbench_frame.mcp.server import MCPServer, create_default_server
from agentbench_frame.mcp.replay_tool import ReadReplayTool
from agentbench_frame.mcp.analyzer_tool import AnalyzeGameTool
from agentbench_frame.mcp.history_tool import QueryHistoryTool

__all__ = [
    "MCPTool",
    "MCPToolRegistry",
    "MCPServer",
    "create_default_server",
    "ReadReplayTool",
    "AnalyzeGameTool",
    "QueryHistoryTool",
]

"""
MCP Server for AgentBench.

Provides a stdio-based MCP server that exposes framework tools to LLM agents.
Follows the MCP (Model Context Protocol) specification for tool listing and calling.

Usage (as subprocess):
    python -m agentbench_frame.mcp.server

The server reads JSON-RPC style messages from stdin and responds on stdout.
This allows LLM hosts (like Claude Desktop, Codex, etc.) to discover and call
AgentBench tools directly.
"""

import sys
import json
from typing import Any, Dict, List, Optional

from agentbench_frame.mcp.base import MCPToolRegistry
from agentbench_frame.mcp.replay_tool import ReadReplayTool
from agentbench_frame.mcp.analyzer_tool import AnalyzeGameTool
from agentbench_frame.mcp.history_tool import QueryHistoryTool
from agentbench_frame.skills.registry import SkillRegistry
from agentbench_frame.skills.replay_reader import ReplayReaderSkill
from agentbench_frame.skills.opponent_modeler import OpponentModelerSkill
from agentbench_frame.skills.map_analyzer import MapAnalyzerSkill


class MCPServer:
    """
    Stdio-based MCP server for AgentBench tools.

    Implements the MCP protocol:
    - initialize: server capabilities
    - tools/list: list available tools
    - tools/call: call a specific tool

    Messages are JSON-RPC style over stdin/stdout.
    """

    def __init__(self,
                 tool_registry: Optional[MCPToolRegistry] = None,
                 skill_registry: Optional[SkillRegistry] = None,
                 name: str = "agentbench-mcp",
                 version: str = "0.1.0"):
        self.name = name
        self.version = version

        # Setup registries
        self.tool_registry = tool_registry or MCPToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()

        # Integrate skills as MCP tools
        self.tool_registry.integrate_with_skills(self.skill_registry)

        # Register built-in tools if not already present
        self._register_defaults()

    def _register_defaults(self):
        """Register the default set of tools."""
        defaults = [
            ReadReplayTool(),
            AnalyzeGameTool(),
            QueryHistoryTool(),
        ]
        for tool in defaults:
            if tool.name not in self.tool_registry:
                self.tool_registry.register(tool)

        # Register default skills if none present
        if len(self.skill_registry) == 0:
            self.skill_registry.register(ReplayReaderSkill())
            self.skill_registry.register(OpponentModelerSkill())
            self.skill_registry.register(MapAnalyzerSkill())
            # Re-integrate to pick up new skills
            self.tool_registry.integrate_with_skills(self.skill_registry)

    def run(self):
        """
        Run the MCP server main loop.

        Reads JSON messages from stdin, processes them, writes responses to stdout.
        """
        print(f"[MCPServer] {self.name} v{self.version} starting", file=sys.stderr)
        print(f"[MCPServer] {len(self.tool_registry)} tools registered", file=sys.stderr)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self._handle_request(request)
                self._send(response)
            except json.JSONDecodeError:
                self._send({"error": "Invalid JSON"})
            except Exception as e:
                self._send({"error": str(e)})

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single MCP request."""
        method = request.get("method", "")
        req_id = request.get("id", 0)
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "0.1.0",
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.tool_registry.list_tools(),
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self.tool_registry.call(tool_name, **arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }
                    ],
                },
            }

        elif method == "skills/list":
            # Extension: list skills
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "skills": self.skill_registry.list_skills(),
                },
            }

        elif method == "skills/stats":
            # Extension: get skill stats
            skill_name = params.get("name")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "stats": self.skill_registry.get_stats(skill_name),
                },
            }

        elif method == "skills/ab_report":
            # Extension: get A/B test report
            skill_name = params.get("name", "")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": self.skill_registry.get_ab_report(skill_name),
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    def _send(self, response: Dict[str, Any]):
        """Send a response to stdout."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    # ---- Convenience methods for programmatic use ----

    def handle_message(self, message: str) -> str:
        """Handle a single message and return the response string."""
        try:
            request = json.loads(message)
            response = self._handle_request(request)
            return json.dumps(response)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_tool_catalog_json(self) -> str:
        """Get the tool catalog as a JSON string."""
        return json.dumps(self.tool_registry.get_catalog(), indent=2)


def create_default_server() -> MCPServer:
    """Create an MCPServer with all default tools and skills registered."""
    skill_registry = SkillRegistry()
    skill_registry.register(ReplayReaderSkill())
    skill_registry.register(OpponentModelerSkill())
    skill_registry.register(MapAnalyzerSkill())

    tool_registry = MCPToolRegistry()
    tool_registry.register(ReadReplayTool())
    tool_registry.register(AnalyzeGameTool())
    tool_registry.register(QueryHistoryTool())
    tool_registry.integrate_with_skills(skill_registry)

    return MCPServer(tool_registry=tool_registry, skill_registry=skill_registry)


# Standalone entry point
if __name__ == "__main__":
    server = create_default_server()
    server.run()

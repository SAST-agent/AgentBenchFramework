"""
MCP Tool base classes.

Defines the interface for MCP-compatible tools that can be:
- Called by LLM agents
- Served via MCP server
- Integrated with skills
- Discovered through tool catalogs
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import json


class MCPTool(ABC):
    """
    Base class for MCP-compatible tools.

    Each tool has:
    - Metadata (name, description)
    - An input schema (JSON Schema)
    - A call() method

    Tools can wrap:
    - Game operations (read replay, analyze state)
    - Framework functions (list agents, query history)
    - Skills (exposed as MCP tools)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for the tool's input parameters."""
        ...

    @abstractmethod
    def call(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.

        Returns a structured result dict.
        """
        ...

    def get_definition(self) -> Dict[str, Any]:
        """Return the full MCP tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.get_input_schema(),
        }

    def __repr__(self) -> str:
        return f"MCPTool({self.name})"


class MCPToolRegistry:
    """
    Registry for MCP tools.

    Supports:
    - Registration and discovery
    - Catalog export
    - Tool calling
    - Integration with SkillRegistry
    """

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._skill_registry = None  # set by integrate_with_skills()

    def register(self, tool: MCPTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Tool not found: {name}", "available": list(self._tools.keys())}
        try:
            return tool.call(**kwargs)
        except Exception as e:
            return {"error": str(e), "tool": name}

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their definitions."""
        return [t.get_definition() for t in self._tools.values()]

    def get_catalog(self) -> Dict[str, Any]:
        """Get the full tool catalog (MCP list format)."""
        return {"tools": self.list_tools()}

    def integrate_with_skills(self, skill_registry):
        """
        Integrate with a SkillRegistry to expose skills as MCP tools.

        When called, skill MCP tools use the skill's call_as_mcp() method.
        """
        self._skill_registry = skill_registry

        # Create wrapper tools for each skill
        for skill_info in skill_registry.list_skills():
            name = skill_info["name"]
            skill = skill_registry.get(name)
            if skill:
                wrapper = SkillMCPTool(skill)
                self.register(wrapper)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"MCPToolRegistry({len(self._tools)} tools)"


class SkillMCPTool(MCPTool):
    """
    Wraps a Skill as an MCP tool.

    Delegates to the skill's call_as_mcp() method.
    """

    def __init__(self, skill):
        self.skill = skill
        super().__init__(
            name=skill.meta.name,
            description=f"[Skill v{skill.meta.version}] {skill.meta.description}",
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return self.skill.as_mcp_tool_definition().get("inputSchema", {
            "type": "object",
            "properties": {},
        })

    def call(self, **kwargs) -> Dict[str, Any]:
        return self.skill.call_as_mcp(**kwargs)

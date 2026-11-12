"""The agent's view of the MCP server: tool schemas for the model, and a way to call the tools.

Tools are loaded from the MCP server with langchain-mcp-adapters over one long-lived stdio session.
The schemas shown to the model leave out the policy parameters; the agent fills those in itself.
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, Protocol

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from citewise.config import server_environment
from citewise.mcp_server import POLICY_PARAMS


class Toolbox(Protocol):
    def specs(self) -> list[dict[str, Any]]:
        """Tool definitions for the model (Anthropic format), without policy parameters."""

    async def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool and return its text output."""


def model_facing_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    for param in POLICY_PARAMS:
        schema.get("properties", {}).pop(param, None)
    if "required" in schema:
        schema["required"] = [p for p in schema["required"] if p not in POLICY_PARAMS]
    return schema


def content_text(content: Any) -> str:
    """Normalise a tool result (string, content blocks or a list of them) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, tuple):  # (content, artifact)
        return content_text(content[0])
    if isinstance(content, list):
        return "".join(content_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text", "")) if content.get("type", "text") == "text" else ""
    text = getattr(content, "text", None)
    return text if isinstance(text, str) else json.dumps(content, default=str)


class MCPToolbox:
    """Starts the Citewise MCP server as a subprocess and keeps one session open."""

    def __init__(
        self,
        command: str = sys.executable,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.connection = {
            "transport": "stdio",
            "command": command,
            "args": args or ["-m", "citewise.mcp_server"],
            "env": server_environment(env),
        }
        self.tools: dict[str, Any] = {}

    async def __aenter__(self) -> MCPToolbox:
        self._client = MultiServerMCPClient({"citewise": self.connection})
        self._session_cm = self._client.session("citewise")
        session = await self._session_cm.__aenter__()
        self.tools = {tool.name: tool for tool in await load_mcp_tools(session)}
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._session_cm.__aexit__(*exc)

    def specs(self) -> list[dict[str, Any]]:
        specs = []
        for tool in self.tools.values():
            schema = tool.args_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            specs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": model_facing_schema(schema),
                }
            )
        return specs

    async def call(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            return json.dumps({"error": f"unknown tool {name}"})
        return content_text(await self.tools[name].ainvoke(args))

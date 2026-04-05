"""MCP (Model Context Protocol) server adapter for HATEOAS resources.

Exposes a HATEOAS state machine as an MCP tool server with dynamic tool lists
that update after each state transition.

Usage::

    from hateoas_agent.mcp_server import serve

    serve(my_state_machine, name="my-server")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _check_mcp_available() -> None:
    """Raise ImportError with install instructions if mcp is not available."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required for MCP server support. "
            "Install it with: pip install 'hateoas-agent[mcp]' "
            "or: pip install mcp"
        ) from None


def _build_tool_schemas(registry: Any) -> list:
    """Convert Registry tool schemas to MCP Tool objects."""
    from mcp.types import Tool

    schemas = registry.get_current_tool_schemas()
    tools = []
    for schema in schemas:
        tools.append(
            Tool(
                name=schema["name"],
                description=schema.get("description", ""),
                inputSchema=schema.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            )
        )
    return tools


def _handle_call_tool(registry: Any, name: str, arguments: dict) -> tuple:
    """Execute a tool call and return (result_text, state_changed).

    Returns:
        Tuple of (result_text, state_changed) where state_changed indicates
        if the tool list should be refreshed.
    """
    old_state = registry._last_state
    result = registry.handle_tool_call(name, arguments or {})
    new_state = registry._last_state
    state_changed = old_state != new_state
    return result, state_changed


def serve(
    resource: Any,
    *,
    name: str = "hateoas-agent",
    transport: str = "stdio",
) -> None:
    """Start an MCP server exposing a HATEOAS resource.

    The server exposes the resource's gateway and dynamic actions as MCP tools.
    After each tool call that changes state, a ``tools/list_changed``
    notification is sent so the client refreshes its tool list.

    Args:
        resource: A StateMachine or Resource instance (anything implementing HasHateoas).
        name: Server name for MCP identification.
        transport: Transport type. Currently supports ``"stdio"`` (default).

    Raises:
        ImportError: If ``mcp`` package is not installed.
        ValueError: If transport is not supported.
    """
    _check_mcp_available()

    if transport != "stdio":
        raise ValueError(f"Unsupported transport: {transport!r}. Use 'stdio'.")

    import asyncio

    from mcp.server.lowlevel import NotificationOptions, Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, TextContent

    from .registry import Registry

    if hasattr(resource, "validate"):
        resource.validate()
    registry = Registry(resource)
    server = Server(name)

    @server.list_tools()
    async def list_tools():
        return _build_tool_schemas(registry)

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict | None = None):
        try:
            result_text, state_changed = _handle_call_tool(
                registry, tool_name, arguments or {}
            )
        except Exception as exc:
            logger.exception("Tool call '%s' failed", tool_name)
            return CallToolResult(
                content=[TextContent(type="text", text="An internal error occurred.")],
                isError=True,
            )

        if state_changed:
            await server.request_context.session.send_tool_list_changed()

        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
        )

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(
                    notification_options=NotificationOptions(
                        tools_changed=True,
                    ),
                ),
            )

    asyncio.run(_run())

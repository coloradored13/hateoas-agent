"""Tests for MCP server adapter (logic tests without running server)."""

import pytest

from hateoas_agent import StateMachine
from hateoas_agent.registry import Registry

# Check if mcp package is available
try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _make_sm():
    sm = StateMachine("test", gateway_name="gw")
    sm.gateway(description="GW", params={"id": "string"})
    sm.action("approve", description="Approve", from_states=["pending"],
              to_state="approved", params={"id": "string"})
    sm.action("ship", description="Ship", from_states=["approved"],
              to_state="shipped", params={"id": "string"})

    @sm.on_gateway
    def gw(id=""):
        return {"order": {"id": id}, "_state": "pending"}

    @sm.on_action("approve")
    def approve(id=""):
        return {"approved": True, "_state": "approved"}

    @sm.on_action("ship")
    def ship(id=""):
        return {"shipped": True, "_state": "shipped"}

    return sm


class TestMCPImportGuard:
    """Test that missing mcp package gives clear error."""

    def test_check_mcp_available(self):
        from hateoas_agent.mcp_server import _check_mcp_available

        if HAS_MCP:
            _check_mcp_available()  # should not raise
        else:
            with pytest.raises(ImportError, match="pip install"):
                _check_mcp_available()


class TestMCPHandlerLogic:
    """Test the handler logic extracted into testable functions."""

    def test_handle_call_tool_gateway(self):
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = _make_sm()
        reg = Registry(sm)

        result, changed = _handle_call_tool(reg, "gw", {"id": "1"})
        assert changed  # None -> pending
        assert "order" in result

    def test_handle_call_tool_action(self):
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = _make_sm()
        reg = Registry(sm)

        _handle_call_tool(reg, "gw", {"id": "1"})
        result, changed = _handle_call_tool(reg, "approve", {"id": "1"})
        assert changed  # pending -> approved
        assert "approved" in result

    def test_handle_call_tool_no_state_change(self):
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action("noop", description="Noop", from_states="*", params={})

        @sm.on_gateway
        def gw():
            return {"_state": "active"}

        @sm.on_action("noop")
        def noop():
            return {"ok": True, "_state": "active"}

        reg = Registry(sm)
        _handle_call_tool(reg, "gw", {})
        _, changed = _handle_call_tool(reg, "noop", {})
        assert not changed

    def test_handle_call_tool_empty_arguments(self):
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw():
            return {"_state": "ready"}

        reg = Registry(sm)
        result, changed = _handle_call_tool(reg, "gw", {})
        assert changed

    def test_handle_call_tool_raises_on_invalid_action(self):
        """_handle_call_tool propagates errors for the serve() wrapper to catch."""
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = _make_sm()
        reg = Registry(sm)

        # Call gateway first to set state
        _handle_call_tool(reg, "gw", {"id": "1"})

        # Try to call 'ship' from 'pending' state — should raise
        from hateoas_agent.errors import InvalidActionError
        with pytest.raises(InvalidActionError):
            _handle_call_tool(reg, "ship", {"id": "1"})

    def test_handle_call_tool_raises_on_unknown_tool(self):
        """Unknown tool name raises InvalidActionError (not a valid action)."""
        from hateoas_agent.mcp_server import _handle_call_tool

        sm = _make_sm()
        reg = Registry(sm)

        _handle_call_tool(reg, "gw", {"id": "1"})

        from hateoas_agent.errors import InvalidActionError
        with pytest.raises(InvalidActionError):
            _handle_call_tool(reg, "nonexistent", {})


@pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")
class TestMCPBuildToolSchemas:
    """Test tool schema conversion (requires mcp package)."""

    def test_build_schemas(self):
        from hateoas_agent.mcp_server import _build_tool_schemas

        sm = _make_sm()
        reg = Registry(sm)

        tools = _build_tool_schemas(reg)
        assert len(tools) >= 1  # at least gateway
        assert tools[0].name == "gw"
        assert tools[0].description == "GW"

    def test_build_schemas_after_state_change(self):
        from hateoas_agent.mcp_server import _build_tool_schemas, _handle_call_tool

        sm = _make_sm()
        reg = Registry(sm)

        # Before gateway
        tools_before = _build_tool_schemas(reg)
        names_before = [t.name for t in tools_before]
        assert "approve" not in names_before

        # After gateway
        _handle_call_tool(reg, "gw", {"id": "1"})
        tools_after = _build_tool_schemas(reg)
        names_after = [t.name for t in tools_after]
        assert "approve" in names_after


@pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")
class TestMCPServeValidation:
    """Test serve() input validation (requires mcp package)."""

    def test_invalid_transport(self):
        from hateoas_agent.mcp_server import serve

        sm = _make_sm()
        with pytest.raises(ValueError, match="Unsupported transport"):
            serve(sm, transport="http")

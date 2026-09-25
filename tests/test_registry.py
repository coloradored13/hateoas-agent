"""Tests for the Registry (routing + validation)."""

import pytest

from hateoas_agent import InvalidActionError, Registry


def test_gateway_tool_schema(order_machine):
    reg = Registry(order_machine)
    schema = reg.get_gateway_tool_schema()
    assert schema["name"] == "query_orders"
    assert "properties" in schema["input_schema"]


def test_is_gateway(order_machine):
    reg = Registry(order_machine)
    assert reg.is_gateway("query_orders") is True
    assert reg.is_gateway("approve_order") is False


def test_gateway_call_returns_actions(order_machine):
    reg = Registry(order_machine)
    result = reg.handle_tool_call("query_orders", {"order_id": "100"})
    assert "pending" in result  # order status
    assert "approve_order" in result  # advertised action
    assert "cancel_order" in result  # advertised action


def test_valid_action_after_gateway(order_machine):
    reg = Registry(order_machine)
    # First, gateway sets state to "pending"
    reg.handle_tool_call("query_orders", {"order_id": "100"})
    # Then, approve (valid for pending)
    result = reg.handle_tool_call("approve_order", {"order_id": "100"})
    assert "success" in result
    # After approval, should advertise "approved" state actions
    assert "ship_order" in result


def test_invalid_action_raises_error(order_machine):
    reg = Registry(order_machine)
    # Set state to "pending"
    reg.handle_tool_call("query_orders", {"order_id": "100"})
    # Try to ship (not valid for pending) — should raise InvalidActionError
    with pytest.raises(InvalidActionError) as exc_info:
        reg.handle_tool_call("ship_order", {"order_id": "100"})
    assert exc_info.value.action == "ship_order"
    assert exc_info.value.state == "pending"


def test_state_transition_chain(order_machine):
    reg = Registry(order_machine)
    # pending -> approved -> shipped
    reg.handle_tool_call("query_orders", {"order_id": "100"})
    reg.handle_tool_call("approve_order", {"order_id": "100"})
    result = reg.handle_tool_call("ship_order", {"order_id": "100", "tracking": "1Z999"})
    assert "success" in result
    assert "track_shipment" in result


def test_resource_class_works_with_registry(order_resource):
    reg = Registry(order_resource)
    result = reg.handle_tool_call("query_orders", {"order_id": "100"})
    assert "approve_order" in result

    result = reg.handle_tool_call("approve_order", {"order_id": "100"})
    assert "success" in result
    assert "ship_order" in result

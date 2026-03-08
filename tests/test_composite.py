"""Tests for multi-resource composition (CompositeRegistry)."""

import pytest

from hateoas_agent import StateMachine
from hateoas_agent.composite import CompositeRegistry, ToolNameConflictError


def _make_orders_sm():
    sm = StateMachine("orders", gateway_name="query_orders")
    sm.gateway(description="Query orders", params={"id": "string"})
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


def _make_inventory_sm():
    sm = StateMachine("inventory", gateway_name="query_inventory")
    sm.gateway(description="Query inventory", params={"sku": "string"})
    sm.action("restock", description="Restock", from_states=["low"],
              to_state="in_stock", params={"sku": "string", "qty": "number"})
    sm.action("reserve", description="Reserve", from_states=["in_stock"],
              to_state="reserved", params={"sku": "string", "qty": "number"})

    @sm.on_gateway
    def gw(sku=""):
        return {"item": {"sku": sku, "qty": 5}, "_state": "in_stock"}

    @sm.on_action("restock")
    def restock(sku="", qty=0):
        return {"restocked": True, "_state": "in_stock"}

    @sm.on_action("reserve")
    def reserve(sku="", qty=0):
        return {"reserved": True, "_state": "reserved"}

    return sm


class TestCompositeRegistry:
    """Tests for CompositeRegistry basic operations."""

    def test_separate_registries(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        assert "query_orders" in composite.gateway_names
        assert "query_inventory" in composite.gateway_names

    def test_tool_schemas_include_all(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        schemas = composite.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "query_orders" in names
        assert "query_inventory" in names

    def test_routing_to_correct_registry(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        # Call orders gateway
        composite.handle_tool_call("query_orders", {"id": "1"})
        # Call inventory gateway
        composite.handle_tool_call("query_inventory", {"sku": "ABC"})

        # Now actions should be available
        schemas = composite.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "approve" in names  # from orders (state=pending)
        assert "reserve" in names  # from inventory (state=in_stock)

    def test_independent_state(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        # Only call orders gateway
        composite.handle_tool_call("query_orders", {"id": "1"})
        composite.handle_tool_call("approve", {"id": "1"})

        # Orders should be in "approved" state, inventory has no state yet
        schemas = composite.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "ship" in names  # orders is now approved
        assert "query_inventory" in names  # inventory gateway always available

    def test_is_gateway(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        assert composite.is_gateway("query_orders")
        assert composite.is_gateway("query_inventory")
        assert not composite.is_gateway("approve")

    def test_is_known_action(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        assert composite.is_known_action("approve")
        assert composite.is_known_action("restock")
        assert not composite.is_known_action("nonexistent")

    def test_last_state_from_active_registry(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        assert composite._last_state is None

        composite.handle_tool_call("query_orders", {"id": "1"})
        assert composite._last_state == "pending"

        composite.handle_tool_call("query_inventory", {"sku": "A"})
        assert composite._last_state == "in_stock"


class TestCompositeConflicts:
    """Tests for tool name conflict detection."""

    def test_gateway_name_conflict(self):
        sm1 = StateMachine("a", gateway_name="query")
        sm1.gateway(description="A", params={})
        sm2 = StateMachine("b", gateway_name="query")
        sm2.gateway(description="B", params={})

        with pytest.raises(ToolNameConflictError, match="query"):
            CompositeRegistry([sm1, sm2])

    def test_action_name_conflict(self):
        sm1 = StateMachine("a", gateway_name="gw1")
        sm1.gateway(description="A", params={})
        sm1.action("do_thing", description="Do", from_states=["s"], params={})

        sm2 = StateMachine("b", gateway_name="gw2")
        sm2.gateway(description="B", params={})
        sm2.action("do_thing", description="Do", from_states=["s"], params={})

        with pytest.raises(ToolNameConflictError, match="do_thing"):
            CompositeRegistry([sm1, sm2])


class TestCompositeDiscovery:
    """Tests for merged discovery reports."""

    def test_merged_report(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        composite.handle_tool_call("query_orders", {"id": "1"})
        composite.handle_tool_call("approve", {"id": "1"})
        composite.handle_tool_call("query_inventory", {"sku": "A"})
        composite.handle_tool_call("reserve", {"sku": "A", "qty": 2})

        report = composite.get_discovery_report()
        assert len(report.transitions) == 2
        actions = [t.action for t in report.transitions]
        assert "approve" in actions
        assert "reserve" in actions

    def test_merged_report_sorted_by_timestamp(self):
        orders = _make_orders_sm()
        inventory = _make_inventory_sm()
        composite = CompositeRegistry([orders, inventory])

        composite.handle_tool_call("query_orders", {"id": "1"})
        composite.handle_tool_call("approve", {"id": "1"})
        composite.handle_tool_call("query_inventory", {"sku": "A"})
        composite.handle_tool_call("reserve", {"sku": "A", "qty": 2})

        report = composite.get_discovery_report()
        timestamps = [t.timestamp for t in report.transitions]
        assert timestamps == sorted(timestamps)

    def test_empty_report(self):
        orders = _make_orders_sm()
        composite = CompositeRegistry([orders])
        report = composite.get_discovery_report()
        assert len(report.transitions) == 0

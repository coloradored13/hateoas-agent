"""Tests for the declarative StateMachine API."""

import pytest

from hateoas_agent import StateMachine


def test_gateway_definition(order_machine):
    gw = order_machine.get_gateway()
    assert gw is not None
    assert gw.name == "query_orders"
    assert gw.description == "Search and retrieve orders"
    assert "order_id" in gw.params


def test_state_actions(order_machine):
    actions = order_machine.get_actions_for_state("pending")
    names = [a.name for a in actions]
    assert "approve_order" in names
    assert "cancel_order" in names


def test_unknown_state_returns_empty(order_machine):
    actions = order_machine.get_actions_for_state("nonexistent")
    assert actions == []


def test_all_action_names(order_machine):
    names = order_machine.get_all_action_names()
    assert "approve_order" in names
    assert "ship_order" in names
    assert "track_shipment" in names
    assert "reopen_order" in names


def test_handlers_attached(order_machine):
    handler = order_machine.get_handler("approve_order")
    assert handler is not None
    result = handler(order_id="100")
    assert result["success"] is True
    assert result["_state"] == "approved"


def test_gateway_handler(order_machine):
    gw = order_machine.get_gateway()
    result = gw.handler(order_id="100")
    assert "order" in result
    assert result["_state"] == "pending"


def test_gateway_handler_not_found(order_machine):
    gw = order_machine.get_gateway()
    result = gw.handler(order_id="999")
    assert "message" in result


class TestValidate:
    """Tests for StateMachine.validate()."""

    def test_valid_machine_passes(self, order_machine):
        order_machine.validate()  # should not raise

    def test_no_gateway_raises(self):
        sm = StateMachine("test", gateway_name="gw")
        with pytest.raises(ValueError, match="no gateway defined"):
            sm.validate()

    def test_no_gateway_handler_raises(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        with pytest.raises(ValueError, match="no gateway handler"):
            sm.validate()

    def test_missing_action_handler_raises(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action("do_thing", description="Do", from_states=["active"], params={})

        @sm.on_gateway
        def gw():
            return {"_state": "active"}

        with pytest.raises(ValueError, match="actions without handlers.*do_thing"):
            sm.validate()

    def test_state_centric_missing_handler_raises(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state("active", actions=[
            {"name": "do_thing", "description": "Do", "params": {}},
        ])

        @sm.on_gateway
        def gw():
            return {"_state": "active"}

        with pytest.raises(ValueError, match="do_thing"):
            sm.validate()

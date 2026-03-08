"""Tests for the Resource (handler-based) API."""

import pytest

from hateoas_agent import Resource, action, gateway


def test_gateway_definition(order_resource):
    gw = order_resource.get_gateway()
    assert gw is not None
    assert gw.name == "query_orders"
    assert gw.handler is not None


def test_state_actions(order_resource):
    actions = order_resource.get_actions_for_state("pending")
    names = [a.name for a in actions]
    assert "approve_order" in names
    assert "cancel_order" in names
    # ship_order should NOT be available in pending state
    assert "ship_order" not in names


def test_approved_state_actions(order_resource):
    actions = order_resource.get_actions_for_state("approved")
    names = [a.name for a in actions]
    assert "ship_order" in names
    assert "cancel_order" in names  # cancel is valid in pending + approved


def test_all_action_names(order_resource):
    names = order_resource.get_all_action_names()
    assert "approve_order" in names
    assert "cancel_order" in names
    assert "ship_order" in names


def test_handler_execution(order_resource):
    handler = order_resource.get_handler("approve_order")
    assert handler is not None
    result = handler(order_id="100")
    assert result["success"] is True


def test_gateway_handler_execution(order_resource):
    gw = order_resource.get_gateway()
    result = gw.handler(order_id="100")
    assert "order" in result
    assert result["_state"] == "pending"


def test_stateless_action_in_all_states(order_resource):
    """Actions without @state decorator should appear in all states."""

    class TestResource(Resource):
        name = "test"

        @gateway(name="lookup", description="Lookup", params={"id": "string"})
        def lookup(self, id=None):
            return {"found": True, "_state": "active"}

        @action(name="do_thing", description="Do a thing", params={"id": "string"})
        def do_thing(self, id):
            return {"done": True, "_state": "active"}

    r = TestResource()
    # Should appear in any state since no @state restriction
    actions = r.get_actions_for_state("active")
    assert any(a.name == "do_thing" for a in actions)

    actions = r.get_actions_for_state("random_state")
    assert any(a.name == "do_thing" for a in actions)


class TestResourceValidate:
    """Tests for Resource.validate() parity with StateMachine.validate()."""

    def test_validate_passes_for_complete_resource(self):
        class GoodResource(Resource):
            name = "good"

            @gateway(name="start", description="Start", params={})
            def start(self):
                return {"ok": True, "_state": "active"}

            @action(name="do_thing", description="Do", params={})
            def do_thing(self):
                return {"done": True, "_state": "active"}

        r = GoodResource()
        r.validate()  # should not raise

    def test_validate_raises_for_missing_gateway(self):
        class NoGateway(Resource):
            name = "bad"

            @action(name="do_thing", description="Do", params={})
            def do_thing(self):
                return {"done": True, "_state": "active"}

        r = NoGateway()
        with pytest.raises(ValueError, match="no gateway defined"):
            r.validate()

    def test_runner_calls_validate_on_resource(self):
        """Runner.__init__ should call validate() on Resource instances."""
        from unittest.mock import MagicMock

        from hateoas_agent import Runner

        class IncompleteResource(Resource):
            name = "incomplete"
            # No gateway defined

            @action(name="do_thing", description="Do", params={})
            def do_thing(self):
                return {"done": True, "_state": "active"}

        r = IncompleteResource()
        with pytest.raises(ValueError, match="no gateway defined"):
            Runner(r, client=MagicMock())

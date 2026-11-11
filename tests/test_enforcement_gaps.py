"""Close enforcement gaps found by mutation testing (mutmut 3.8).

Each test here was checked two ways: it passes on the current code, and it
fails when the guarded line is broken. Before these tests existed, every one of
those broken versions passed the full suite.
"""

import pytest

from hateoas_agent import StateMachine
from hateoas_agent.composite import CompositeRegistry, ToolNameConflictError
from hateoas_agent.errors import StateTransitionError
from hateoas_agent.registry import Registry


def _orders(calls):
    sm = StateMachine("orders", gateway_name="query_orders")
    sm.gateway(description="Look up an order", params={"order_id": "string"}, required=["order_id"])
    sm.action(
        "ship",
        description="Ship an approved order",
        from_states=["approved"],
        to_state="shipped",
        params={"order_id": "string", "carrier": "string"},
        required=["order_id", "carrier"],
    )

    @sm.on_gateway
    def query(order_id=None):
        return {"order": order_id, "_state": "approved"}

    @sm.on_action("ship")
    def ship(order_id=None, carrier=None):
        calls.append((order_id, carrier))
        return {"shipped": True, "_state": "shipped"}

    return sm


class TestRequiredParamsAreEnforced:
    """Existing tests check that `required` is advertised in schemas.
    These check that a call missing a required param is actually refused."""

    def test_action_missing_required_param_is_rejected(self):
        calls = []
        reg = Registry(_orders(calls))
        reg.handle_tool_call("query_orders", {"order_id": "1"})

        out = reg.handle_tool_call("ship", {"order_id": "1"})  # no carrier

        assert "Missing required parameter(s) for 'ship': carrier" in out
        assert calls == []  # handler never ran
        assert reg._last_state == "approved"  # state did not move

    def test_gateway_missing_required_param_is_rejected(self):
        reg = Registry(_orders([]))

        out = reg.handle_tool_call("query_orders", {})

        assert "Missing required parameter(s) for 'query_orders': order_id" in out


class TestCompositeSecurityDefaults:
    """CompositeRegistry declares its own strict_transitions default and passes
    it down explicitly, so Registry's pinned default does not cover it."""

    def test_composite_rejects_lying_handler_by_default(self):
        sm = StateMachine("orders", gateway_name="query_orders")
        sm.gateway(description="q", params={})
        sm.action("approve", description="a", from_states=["pending"], to_state="approved")

        @sm.on_gateway
        def query():
            return {"_state": "pending"}

        @sm.on_action("approve")
        def approve():
            return {"_state": "refunded"}  # contradicts declared to_state

        composite = CompositeRegistry([sm])
        composite.handle_tool_call("query_orders", {})

        with pytest.raises(StateTransitionError):
            composite.handle_tool_call("approve", {})


class TestInputValidation:
    def test_from_states_rejects_non_string_entries(self):
        sm = StateMachine("orders", gateway_name="q")
        with pytest.raises(ValueError, match="from_states entries must be strings"):
            sm.action("approve", description="a", from_states=["pending", None])

    def test_conflict_error_names_both_resources(self):
        sm1 = StateMachine("orders", gateway_name="query")
        sm1.gateway(description="A", params={})
        sm2 = StateMachine("inventory", gateway_name="query")
        sm2.gateway(description="B", params={})

        with pytest.raises(ToolNameConflictError) as exc:
            CompositeRegistry([sm1, sm2])

        assert sorted(exc.value.resources) == ["inventory", "orders"]

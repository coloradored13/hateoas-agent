"""
Integration tests for hateoas-agent against the real Claude API.

These tests cost real money — run manually or via `pytest -m integration`.

Usage:
    ANTHROPIC_API_KEY=sk-... pytest tests/test_integration.py -m integration -v
"""

import os

import pytest

from hateoas_agent import Runner, StateMachine

# Skip entire module if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]

MODEL = "claude-haiku-4-5-20251001"


# ── Helpers ──────────────────────────────────────────────────────────


def make_order_machine(db=None):
    """Build a fresh order management StateMachine with its own in-memory DB."""
    if db is None:
        db = {
            "123": {
                "id": "123",
                "customer": "Alice",
                "items": ["Widget"],
                "total": 49.99,
                "status": "pending",
                "notes": [],
            },
        }

    sm = StateMachine("orders", gateway_name="query_orders")
    sm.gateway(
        description="Search and retrieve orders. Starting point for all order operations.",
        params={"order_id": "string"},
        required=["order_id"],
    )

    sm.state(
        "pending",
        actions=[
            {
                "name": "approve_order",
                "description": "Approve this order for fulfillment",
                "params": {"order_id": "string"},
                "required": ["order_id"],
            },
            {
                "name": "cancel_order",
                "description": "Cancel this order",
                "params": {"order_id": "string", "reason": "string"},
                "required": ["order_id"],
            },
        ],
    )

    sm.state(
        "approved",
        actions=[
            {
                "name": "ship_order",
                "description": "Mark order as shipped with tracking info",
                "params": {"order_id": "string", "tracking_number": "string"},
                "required": ["order_id", "tracking_number"],
            },
        ],
    )

    sm.state(
        "shipped",
        actions=[
            {
                "name": "track_shipment",
                "description": "Get tracking details for this shipment",
                "params": {"order_id": "string"},
                "required": ["order_id"],
            },
        ],
    )

    sm.state("cancelled", actions=[])

    # ── Handlers ──

    @sm.on_gateway
    def handle_query(order_id=None):
        if order_id and order_id in db:
            order = db[order_id]
            return {"order": order.copy(), "_state": order["status"]}
        return {"message": f"No order found with id '{order_id}'"}

    @sm.on_action("approve_order")
    def handle_approve(order_id):
        order = db[order_id]
        order["status"] = "approved"
        return {
            "success": True,
            "message": f"Order {order_id} approved",
            "order": order.copy(),
            "_state": "approved",
        }

    @sm.on_action("cancel_order")
    def handle_cancel(order_id, reason="No reason given"):
        order = db[order_id]
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        return {
            "success": True,
            "message": f"Order {order_id} cancelled",
            "order": order.copy(),
            "_state": "cancelled",
        }

    @sm.on_action("ship_order")
    def handle_ship(order_id, tracking_number=""):
        order = db[order_id]
        order["status"] = "shipped"
        order["tracking"] = tracking_number
        return {
            "success": True,
            "message": f"Order {order_id} shipped",
            "order": order.copy(),
            "_state": "shipped",
        }

    @sm.on_action("track_shipment")
    def handle_track(order_id):
        order = db[order_id]
        return {
            "tracking_number": order.get("tracking", "N/A"),
            "status": "In Transit",
            "order": order.copy(),
            "_state": order["status"],
        }

    return sm, db


# ── Tests ────────────────────────────────────────────────────────────


class TestHappyPathSingleAction:
    """Scenario 1: Look up order 123 and approve it."""

    def test_lookup_and_approve(self):
        sm, db = make_order_machine()
        tool_log = []

        runner = Runner(
            sm,
            model=MODEL,
            on_tool_call=lambda name, inp: tool_log.append((name, inp)),
        )

        result = runner.run("Look up order 123 and approve it.")

        # Claude should have produced a final text response
        assert result.text, "Expected non-empty final text"
        assert not result.truncated, "Agent loop should not have been truncated"

        # Should have called gateway then approve_order
        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "query_orders" in tool_names, f"Expected gateway call, got: {tool_names}"
        assert "approve_order" in tool_names, f"Expected approve_order call, got: {tool_names}"

        # State transition: pending -> approved
        assert db["123"]["status"] == "approved"

        # RunResult properties
        assert result.gateway_calls >= 1
        assert result.dynamic_calls >= 1
        assert "query_orders" in result.unique_tools
        assert "approve_order" in result.unique_tools

        print(f"\n  Tool calls: {tool_names}")
        print(f"  Gateway: {result.gateway_calls}, Dynamic: {result.dynamic_calls}")
        print(f"  Final DB status: {db['123']['status']}")


class TestHappyPathMultiStep:
    """Scenario 2: Look up, approve, then ship with tracking."""

    def test_lookup_approve_ship(self):
        sm, db = make_order_machine()

        runner = Runner(
            sm,
            model=MODEL,
            max_turns=15,
        )

        result = runner.run(
            "Look up order 123, approve it, then ship it with tracking number 1Z999."
        )

        assert result.text, "Expected non-empty final text"
        assert not result.truncated

        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "query_orders" in tool_names
        assert "approve_order" in tool_names
        assert "ship_order" in tool_names

        # Full chain: pending -> approved -> shipped
        assert db["123"]["status"] == "shipped"
        assert db["123"].get("tracking") == "1Z999"

        # Should have at least 3 tool calls (gateway + approve + ship)
        assert len(result.tool_calls) >= 3, f"Expected >=3 tool calls, got {len(result.tool_calls)}"

        print(f"\n  Tool calls: {tool_names}")
        print(f"  Gateway: {result.gateway_calls}, Dynamic: {result.dynamic_calls}")
        print(f"  Final status: {db['123']['status']}, tracking: {db['123'].get('tracking')}")


class TestGatewayOnly:
    """Scenario 3: Just look up an order, no action needed."""

    def test_status_query_only(self):
        sm, db = make_order_machine()

        runner = Runner(sm, model=MODEL)

        result = runner.run("What's the status of order 123?")

        assert result.text, "Expected non-empty final text"
        assert not result.truncated

        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "query_orders" in tool_names

        # Should NOT have called any action — just the gateway
        assert result.dynamic_calls == 0, (
            f"Expected 0 dynamic calls for a status query, got {result.dynamic_calls}: {tool_names}"
        )

        # DB unchanged
        assert db["123"]["status"] == "pending"

        print(f"\n  Tool calls: {tool_names}")
        print(f"  Response excerpt: {result.text[:200]}")


class TestSelfCorrectionOnInvalidAction:
    """Scenario 4: Claude tries an action not available in current state and recovers."""

    def test_recovery_from_invalid_action(self):
        # Start with a cancelled order — no actions available
        db = {
            "123": {
                "id": "123",
                "customer": "Alice",
                "items": ["Widget"],
                "total": 49.99,
                "status": "cancelled",
                "notes": [],
            },
        }
        sm, db = make_order_machine(db)
        invalid_attempts = []

        runner = Runner(
            sm,
            model=MODEL,
            max_turns=10,
            on_invalid_action=lambda name, inp, st: invalid_attempts.append((name, st)),
        )

        # Ask to approve a cancelled order — there are no available actions
        result = runner.run("Look up order 123 and try to approve it.")

        assert result.text, "Expected non-empty final text"

        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "query_orders" in tool_names

        # Claude should either:
        # 1. Recognize no approve action is available and respond without calling it, OR
        # 2. Try approve_order, get an error, and self-correct by responding with text
        # Either is acceptable HATEOAS behavior
        if "approve_order" in tool_names:
            # Claude tried it — verify it got an error and recovered
            assert len(invalid_attempts) > 0, "Expected invalid action callback"
            print("\n  Claude attempted invalid action and self-corrected")
            print(f"  Invalid attempts: {invalid_attempts}")
        else:
            print("\n  Claude correctly recognized no approve action was available")

        # Order should still be cancelled
        assert db["123"]["status"] == "cancelled"

        print(f"  Tool calls: {tool_names}")
        print(f"  Final text excerpt: {result.text[:200]}")


class TestMultiTurnConversation:
    """Scenario 5: Two-turn conversation using run_multi."""

    def test_multi_turn_state_persistence(self):
        sm, db = make_order_machine()

        runner = Runner(sm, model=MODEL)

        result = runner.run_multi(
            [
                "Look up order 123.",
                "Now approve it.",
            ]
        )

        assert result.text, "Expected non-empty final text"

        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "query_orders" in tool_names
        assert "approve_order" in tool_names

        # State should have persisted across turns
        assert db["123"]["status"] == "approved"

        # Should have tool calls from both turns
        assert len(result.tool_calls) >= 2

        print(f"\n  Tool calls across turns: {tool_names}")
        print(f"  Gateway: {result.gateway_calls}, Dynamic: {result.dynamic_calls}")
        print(f"  Final status: {db['123']['status']}")


class TestRunResultProperties:
    """Scenario 6: Verify RunResult computed properties."""

    def test_result_properties(self):
        sm, db = make_order_machine()

        runner = Runner(sm, model=MODEL, max_turns=15)

        result = runner.run("Look up order 123, approve it, then ship it with tracking 1Z999.")

        # Verify properties are correct
        assert isinstance(result.gateway_calls, int)
        assert isinstance(result.dynamic_calls, int)
        assert isinstance(result.unique_tools, set)
        assert isinstance(result.truncated, bool)

        # gateway_calls + dynamic_calls == total
        assert result.gateway_calls + result.dynamic_calls == len(result.tool_calls)

        # unique_tools should match what we see in tool_calls
        actual_unique = {tc["tool"] for tc in result.tool_calls}
        assert result.unique_tools == actual_unique

        # Not truncated
        assert not result.truncated

        print(f"\n  gateway_calls: {result.gateway_calls}")
        print(f"  dynamic_calls: {result.dynamic_calls}")
        print(f"  unique_tools: {result.unique_tools}")
        print(f"  truncated: {result.truncated}")
        print(f"  total tool_calls: {len(result.tool_calls)}")

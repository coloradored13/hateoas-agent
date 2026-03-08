"""Shared fixtures for hateoas-agent tests."""

import pytest

from hateoas_agent import Resource, StateMachine, action, gateway, state


@pytest.fixture
def order_machine():
    """A StateMachine for order management (matching the POC)."""
    sm = StateMachine("orders", gateway_name="query_orders")
    sm.gateway(
        description="Search and retrieve orders",
        params={"order_id": "string"},
    )
    sm.state(
        "pending",
        actions=[
            {"name": "approve_order", "description": "Approve", "params": {"order_id": "string"}},
            {
                "name": "cancel_order",
                "description": "Cancel",
                "params": {"order_id": "string", "reason": "string"},
            },
        ],
    )
    sm.state(
        "approved",
        actions=[
            {
                "name": "ship_order",
                "description": "Ship",
                "params": {"order_id": "string", "tracking": "string"},
            },
        ],
    )
    sm.state(
        "shipped",
        actions=[
            {"name": "track_shipment", "description": "Track", "params": {"order_id": "string"}},
        ],
    )
    sm.state(
        "cancelled",
        actions=[
            {"name": "reopen_order", "description": "Reopen", "params": {"order_id": "string"}},
        ],
    )

    db = {
        "100": {"id": "100", "status": "pending", "customer": "Alice"},
        "200": {"id": "200", "status": "shipped", "customer": "Bob"},
    }

    @sm.on_gateway
    def query(order_id=None, **kw):
        if order_id and order_id in db:
            return {"order": db[order_id], "_state": db[order_id]["status"]}
        return {"message": "Not found"}

    @sm.on_action("approve_order")
    def approve(order_id):
        db[order_id]["status"] = "approved"
        return {"success": True, "_state": "approved"}

    @sm.on_action("cancel_order")
    def cancel(order_id, reason=""):
        db[order_id]["status"] = "cancelled"
        return {"success": True, "_state": "cancelled"}

    @sm.on_action("ship_order")
    def ship(order_id, tracking=""):
        db[order_id]["status"] = "shipped"
        return {"success": True, "_state": "shipped"}

    @sm.on_action("track_shipment")
    def track(order_id):
        return {"tracking": "1Z999", "_state": "shipped"}

    @sm.on_action("reopen_order")
    def reopen(order_id):
        db[order_id]["status"] = "pending"
        return {"success": True, "_state": "pending"}

    return sm


@pytest.fixture
def order_resource():
    """A Resource class for order management."""
    db = {
        "100": {"id": "100", "status": "pending", "customer": "Alice"},
        "200": {"id": "200", "status": "shipped", "customer": "Bob"},
    }

    class OrderResource(Resource):
        name = "orders"

        @gateway(name="query_orders", description="Search orders", params={"order_id": "string"})
        def query(self, order_id=None, **kw):
            if order_id and order_id in db:
                return {"order": db[order_id], "_state": db[order_id]["status"]}
            return {"message": "Not found"}

        @action(name="approve_order", description="Approve", params={"order_id": "string"})
        @state("pending")
        def approve(self, order_id):
            db[order_id]["status"] = "approved"
            return {"success": True, "_state": "approved"}

        @action(
            name="cancel_order",
            description="Cancel",
            params={"order_id": "string", "reason": "string"},
        )
        @state("pending", "approved")
        def cancel(self, order_id, reason=""):
            db[order_id]["status"] = "cancelled"
            return {"success": True, "_state": "cancelled"}

        @action(
            name="ship_order",
            description="Ship",
            params={"order_id": "string", "tracking": "string"},
        )
        @state("approved")
        def ship(self, order_id, tracking=""):
            db[order_id]["status"] = "shipped"
            return {"success": True, "_state": "shipped"}

    return OrderResource()

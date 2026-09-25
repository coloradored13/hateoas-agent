"""HATEOAS Agent — MCP Server example.

Exposes the orders state machine as an MCP tool server.
The tool list dynamically updates after each state transition.

Usage:
    pip install 'hateoas-agent[mcp]'
    python examples/orders_mcp.py

Then configure your MCP client to connect via stdio.
"""

from hateoas_agent import StateMachine

# ============================================================
# Define the order management state machine
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders")

orders.gateway(
    description="Search and retrieve orders. Starting point for all order operations.",
    params={"order_id": "string", "status": "string"},
)

orders.action(
    "approve_order",
    description="Approve this order for fulfillment",
    from_states=["pending"],
    to_state="approved",
    params={"order_id": "string"},
    required=["order_id"],
)

orders.action(
    "cancel_order",
    description="Cancel this order",
    from_states=["pending", "approved"],
    to_state="cancelled",
    params={"order_id": "string", "reason": "string"},
    required=["order_id"],
)

orders.action(
    "ship_order",
    description="Mark order as shipped",
    from_states=["approved"],
    to_state="shipped",
    params={"order_id": "string", "tracking_number": "string"},
    required=["order_id"],
)

orders.action(
    "add_note",
    description="Add an internal note to this order",
    from_states="*",
    params={"order_id": "string", "note": "string"},
    required=["order_id"],
)

# ============================================================
# In-memory database
# ============================================================

ORDERS = {
    "1001": {"id": "1001", "customer": "Jane Smith", "total": 89.99, "status": "pending"},
    "1002": {"id": "1002", "customer": "Bob Jones", "total": 249.99, "status": "approved"},
}


@orders.on_gateway
def handle_query(order_id=None, status=None):
    matches = []
    for o in ORDERS.values():
        if order_id and o["id"] == order_id:
            matches.append(o)
        elif status and o["status"] == status:
            matches.append(o)
    if not matches and not order_id and not status:
        matches = list(ORDERS.values())
    if len(matches) == 1:
        return {"order": matches[0].copy(), "_state": matches[0]["status"]}
    elif matches:
        return {"orders": [o.copy() for o in matches]}
    return {"message": "No orders found"}


@orders.on_action("approve_order")
def handle_approve(order_id):
    ORDERS[order_id]["status"] = "approved"
    return {"success": True, "order": ORDERS[order_id].copy(), "_state": "approved"}


@orders.on_action("cancel_order")
def handle_cancel(order_id, reason=""):
    ORDERS[order_id]["status"] = "cancelled"
    return {"success": True, "order": ORDERS[order_id].copy(), "_state": "cancelled"}


@orders.on_action("ship_order")
def handle_ship(order_id, tracking_number=""):
    ORDERS[order_id]["status"] = "shipped"
    ORDERS[order_id]["tracking"] = tracking_number
    return {"success": True, "order": ORDERS[order_id].copy(), "_state": "shipped"}


@orders.on_action("add_note")
def handle_note(order_id, note=""):
    ORDERS[order_id].setdefault("notes", []).append(note)
    return {"success": True, "order": ORDERS[order_id].copy(),
            "_state": ORDERS[order_id]["status"]}


# ============================================================
# Start MCP server
# ============================================================

if __name__ == "__main__":
    from hateoas_agent.mcp_server import serve

    print("Starting MCP server for orders...")
    serve(orders, name="orders-mcp")

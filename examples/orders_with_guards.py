"""HATEOAS Agent — Guards example.

Demonstrates conditional action availability using guard functions.
Actions with guards are only shown when their condition is met.

In this example:
  - refund_order is only available for orders under $500
  - expedite_order is only available when items are in stock
"""

from hateoas_agent import Registry, StateMachine

# ============================================================
# Define order state machine with guards
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders")

orders.gateway(
    description="Search and retrieve orders.",
    params={"order_id": "string"},
)

orders.action(
    "approve_order",
    description="Approve this order for fulfillment",
    from_states=["pending"],
    to_state="approved",
    params={"order_id": "string"},
)

orders.action(
    "refund_order",
    description="Issue a refund for this order",
    from_states=["shipped"],
    params={"order_id": "string"},
    guard=lambda ctx: ctx.get("order", {}).get("total", 0) < 500,
)

orders.action(
    "expedite_order",
    description="Expedite shipping for this order",
    from_states=["approved"],
    params={"order_id": "string"},
    guard=lambda ctx: ctx.get("order", {}).get("in_stock", False),
)

# ============================================================
# In-memory database
# ============================================================

ORDERS = {
    "100": {"id": "100", "total": 49.99, "status": "pending", "in_stock": True},
    "200": {"id": "200", "total": 999.99, "status": "shipped", "in_stock": True},
    "300": {"id": "300", "total": 25.00, "status": "shipped", "in_stock": False},
}


@orders.on_gateway
def handle_query(order_id=None):
    if order_id and order_id in ORDERS:
        order = ORDERS[order_id]
        return {"order": order.copy(), "_state": order["status"]}
    return {"message": "Order not found"}


@orders.on_action("approve_order")
def handle_approve(order_id=""):
    order = ORDERS[order_id]
    order["status"] = "approved"
    return {"success": True, "order": order.copy(), "_state": "approved"}


@orders.on_action("refund_order")
def handle_refund(order_id=""):
    order = ORDERS[order_id]
    return {"success": True, "refund_amount": order["total"], "order": order.copy(),
            "_state": "refunded"}


@orders.on_action("expedite_order")
def handle_expedite(order_id=""):
    order = ORDERS[order_id]
    return {"success": True, "order": order.copy(), "_state": "expedited"}


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    import json

    reg = Registry(orders)

    print("=" * 60)
    print("SCENARIO 1: Small order (total=$25) — refund IS available")
    print("=" * 60)
    reg.handle_tool_call("query_orders", {"order_id": "300"})
    schemas = reg.get_current_tool_schemas()
    tool_names = [s["name"] for s in schemas]
    print(f"Available tools: {tool_names}")
    assert "refund_order" in tool_names, "refund should be available for cheap orders"

    print()
    print("=" * 60)
    print("SCENARIO 2: Expensive order (total=$999) — refund NOT available")
    print("=" * 60)
    reg2 = Registry(orders)
    reg2.handle_tool_call("query_orders", {"order_id": "200"})
    schemas2 = reg2.get_current_tool_schemas()
    tool_names2 = [s["name"] for s in schemas2]
    print(f"Available tools: {tool_names2}")
    assert "refund_order" not in tool_names2, "refund should NOT be available for expensive orders"

    print()
    print("=" * 60)
    print("SCENARIO 3: Pending order, approve then check expedite")
    print("=" * 60)
    reg3 = Registry(orders)
    reg3.handle_tool_call("query_orders", {"order_id": "100"})
    reg3.handle_tool_call("approve_order", {"order_id": "100"})
    schemas3 = reg3.get_current_tool_schemas()
    tool_names3 = [s["name"] for s in schemas3]
    print(f"Available tools: {tool_names3}")
    assert "expedite_order" in tool_names3, "expedite should be available (in_stock=True)"

    print("\nAll guard scenarios passed!")

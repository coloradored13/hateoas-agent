"""HATEOAS Agent — Discovery mode example.

Demonstrates how to run with no state constraints, observe transitions,
and auto-generate a state machine definition from the observed behavior.

In discovery mode:
  1. All actions are available in every state (no constraints)
  2. Every state transition is logged
  3. After running, you get a report with suggested .action() code
"""

from hateoas_agent import StateMachine, Runner

# ============================================================
# Define actions WITHOUT from_states (discovery mode)
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders", mode="discover")

orders.gateway(
    description="Search and retrieve orders. Starting point for all order operations.",
    params={"order_id": "string", "customer_name": "string", "status": "string"},
)

# In discovery mode, from_states is optional — all actions are always available
orders.action("approve_order", description="Approve this order for fulfillment", params={"order_id": "string"})
orders.action("cancel_order", description="Cancel this order", params={"order_id": "string", "reason": "string"})
orders.action("modify_order", description="Modify items or details on this order", params={"order_id": "string", "changes": "string"})
orders.action("ship_order", description="Mark order as shipped and add tracking", params={"order_id": "string", "tracking_number": "string", "carrier": "string"})
orders.action("track_shipment", description="Get tracking details for this shipment", params={"order_id": "string"})
orders.action("initiate_return", description="Start a return process for this order", params={"order_id": "string", "reason": "string"})
orders.action("reopen_order", description="Reopen this cancelled order", params={"order_id": "string"})
orders.action("add_note", description="Add an internal note to this order", params={"order_id": "string", "note": "string"})

# ============================================================
# In-memory database
# ============================================================

ORDERS = {
    "4521": {"id": "4521", "customer": "Jane Smith", "items": ["Widget Pro", "Gadget Mini"], "total": 89.99, "status": "pending", "notes": []},
    "4522": {"id": "4522", "customer": "Bob Jones", "items": ["Mega Bundle"], "total": 249.99, "status": "shipped", "tracking": "1Z999AA10123456784", "notes": ["Expedited shipping requested"]},
    "4523": {"id": "4523", "customer": "Alice Chen", "items": ["Basic Plan - Annual"], "total": 599.00, "status": "cancelled", "cancel_reason": "Found cheaper alternative", "notes": []},
}

# ============================================================
# Handlers — return _state as usual, framework infers transitions
# ============================================================


@orders.on_gateway
def handle_query(order_id=None, customer_name=None, status=None):
    matches = []
    for o in ORDERS.values():
        if order_id and o["id"] == order_id:
            matches.append(o)
        elif customer_name and customer_name.lower() in o["customer"].lower():
            matches.append(o)
        elif status and o["status"] == status:
            matches.append(o)
    if not matches and not order_id and not customer_name and not status:
        matches = list(ORDERS.values())

    if len(matches) == 1:
        return {"order": matches[0].copy(), "_state": matches[0]["status"]}
    elif matches:
        return {"orders": [o.copy() for o in matches]}
    else:
        return {"message": "No orders found matching your query"}


@orders.on_action("approve_order")
def handle_approve(order_id):
    order = ORDERS[order_id]
    order["status"] = "approved"
    return {"success": True, "message": f"Order {order_id} approved", "order": order.copy(), "_state": "approved"}


@orders.on_action("cancel_order")
def handle_cancel(order_id, reason="No reason given"):
    order = ORDERS[order_id]
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    return {"success": True, "message": f"Order {order_id} cancelled", "order": order.copy(), "_state": "cancelled"}


@orders.on_action("modify_order")
def handle_modify(order_id, changes=""):
    order = ORDERS[order_id]
    order["notes"].append(f"Modified: {changes}")
    return {"success": True, "message": f"Order {order_id} modified: {changes}", "order": order.copy(), "_state": order["status"]}


@orders.on_action("add_note")
def handle_add_note(order_id, note=""):
    order = ORDERS[order_id]
    order["notes"].append(note)
    return {"success": True, "message": f"Note added to order {order_id}", "order": order.copy(), "_state": order["status"]}


@orders.on_action("ship_order")
def handle_ship(order_id, tracking_number="", carrier=""):
    order = ORDERS[order_id]
    order["status"] = "shipped"
    order["tracking"] = tracking_number
    order["carrier"] = carrier
    return {"success": True, "message": f"Order {order_id} shipped", "order": order.copy(), "_state": "shipped"}


@orders.on_action("track_shipment")
def handle_track(order_id):
    order = ORDERS[order_id]
    return {
        "tracking_number": order.get("tracking", "N/A"),
        "carrier": "UPS",
        "status": "In Transit",
        "estimated_delivery": "2026-03-07",
        "order": order.copy(),
        "_state": order["status"],
    }


@orders.on_action("initiate_return")
def handle_return(order_id, reason="", items=""):
    order = ORDERS[order_id]
    order["status"] = "return_initiated"
    order["return_reason"] = reason
    return {
        "success": True,
        "message": f"Return initiated for order {order_id}",
        "return_label_url": f"https://returns.example.com/label/RET-{order_id}-001",
        "order": order.copy(),
        "_state": "return_initiated",
    }


@orders.on_action("reopen_order")
def handle_reopen(order_id):
    order = ORDERS[order_id]
    order["status"] = "pending"
    order.pop("cancel_reason", None)
    return {"success": True, "message": f"Order {order_id} reopened", "order": order.copy(), "_state": "pending"}


# ============================================================
# Run scenarios and discover the state machine
# ============================================================

if __name__ == "__main__":
    import json

    def on_tool(name, inp):
        print(f"  -> {name}({json.dumps(inp)[:120]})")

    def on_text(text):
        print(f"  Agent: {text[:300]}")

    def on_transition(old, action, new):
        print(f"  [transition] {old} --{action}--> {new}")

    runner = Runner(
        orders,
        model="claude-sonnet-4-20250514",
        system="You are an order management assistant. Help staff manage orders.",
        on_tool_call=on_tool,
        on_text=on_text,
        on_transition=on_transition,
    )

    # Run several scenarios to exercise different paths
    scenarios = [
        "Look up order 4521 and approve it.",
        "Ship order 4521 with tracking number 1Z123 via UPS.",
        "Look up order 4523 and reopen it, then cancel it again with reason 'changed mind'.",
        "Look up order 4522 and initiate a return because the item was damaged.",
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {i}: {scenario}")
        print("=" * 60)
        result = runner.run(scenario)
        print(f"Tool calls: {len(result.tool_calls)}")

    # ============================================================
    # Print discovery results
    # ============================================================
    report = runner.get_discovery_report()

    print("\n" + "=" * 60)
    print("DISCOVERY REPORT")
    print("=" * 60)

    print(f"\nObserved {len(report.transitions)} transitions:\n")
    for t in report.transitions:
        print(f"  {t.state_before} --{t.action}--> {t.state_after}")

    print(f"\nState map (state -> available actions):")
    for state, actions in report.to_state_map().items():
        print(f"  {state}: {actions}")

    print(f"\nAction map (action -> transitions):")
    for action, meta in report.to_action_map().items():
        print(f"  {action}: from={meta['from_states']} to={meta['to_state']}")

    print(f"\n{'=' * 60}")
    print("GENERATED .action() CODE")
    print("=" * 60)
    print()
    print(report.to_python("orders"))

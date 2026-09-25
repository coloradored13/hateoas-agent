"""HATEOAS Agent — Multi-Resource example.

Demonstrates composing multiple state machines into a single agent
that can operate across both an Orders and an Inventory resource.
"""

from hateoas_agent import StateMachine
from hateoas_agent.composite import CompositeRegistry

# ============================================================
# Orders resource
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders")
orders.gateway(description="Search and retrieve orders", params={"order_id": "string"})
orders.action("approve_order", description="Approve this order",
              from_states=["pending"], to_state="approved",
              params={"order_id": "string"})
orders.action("ship_order", description="Ship this order",
              from_states=["approved"], to_state="shipped",
              params={"order_id": "string"})

ORDERS = {
    "1": {"id": "1", "status": "pending", "items": ["Widget"]},
}


@orders.on_gateway
def query_orders(order_id=None):
    if order_id and order_id in ORDERS:
        o = ORDERS[order_id]
        return {"order": o.copy(), "_state": o["status"]}
    return {"orders": list(ORDERS.values())}


@orders.on_action("approve_order")
def approve(order_id=""):
    ORDERS[order_id]["status"] = "approved"
    return {"approved": True, "order": ORDERS[order_id].copy(), "_state": "approved"}


@orders.on_action("ship_order")
def ship(order_id=""):
    ORDERS[order_id]["status"] = "shipped"
    return {"shipped": True, "order": ORDERS[order_id].copy(), "_state": "shipped"}


# ============================================================
# Inventory resource
# ============================================================

inventory = StateMachine("inventory", gateway_name="query_inventory")
inventory.gateway(description="Check inventory levels", params={"sku": "string"})
inventory.action("restock", description="Restock item",
                 from_states=["low", "out_of_stock"], to_state="in_stock",
                 params={"sku": "string", "qty": "number"})
inventory.action("reserve", description="Reserve stock for an order",
                 from_states=["in_stock"], to_state="reserved",
                 params={"sku": "string", "qty": "number"})

INVENTORY = {
    "WIDGET-001": {"sku": "WIDGET-001", "qty": 50, "status": "in_stock"},
    "GADGET-002": {"sku": "GADGET-002", "qty": 0, "status": "out_of_stock"},
}


@inventory.on_gateway
def query_inventory(sku=None):
    if sku and sku in INVENTORY:
        item = INVENTORY[sku]
        return {"item": item.copy(), "_state": item["status"]}
    return {"inventory": list(INVENTORY.values())}


@inventory.on_action("restock")
def restock(sku="", qty=0):
    INVENTORY[sku]["qty"] += int(qty)
    INVENTORY[sku]["status"] = "in_stock"
    return {"restocked": True, "item": INVENTORY[sku].copy(), "_state": "in_stock"}


@inventory.on_action("reserve")
def reserve(sku="", qty=0):
    INVENTORY[sku]["qty"] -= int(qty)
    INVENTORY[sku]["status"] = "reserved"
    return {"reserved": True, "item": INVENTORY[sku].copy(), "_state": "reserved"}


# ============================================================
# Compose and use
# ============================================================

if __name__ == "__main__":
    composite = CompositeRegistry([orders, inventory])

    print("Gateway tools:", composite.gateway_names)

    print("\n" + "=" * 60)
    print("Initial tool schemas:")
    print("=" * 60)
    for schema in composite.get_current_tool_schemas():
        print(f"  {schema['name']}: {schema['description']}")

    print("\n" + "=" * 60)
    print("After querying an order:")
    print("=" * 60)
    composite.handle_tool_call("query_orders", {"order_id": "1"})
    for schema in composite.get_current_tool_schemas():
        print(f"  {schema['name']}: {schema['description']}")

    print("\n" + "=" * 60)
    print("After also querying inventory:")
    print("=" * 60)
    composite.handle_tool_call("query_inventory", {"sku": "WIDGET-001"})
    for schema in composite.get_current_tool_schemas():
        print(f"  {schema['name']}: {schema['description']}")

    print("\n" + "=" * 60)
    print("Discovery report after some operations:")
    print("=" * 60)
    composite.handle_tool_call("approve_order", {"order_id": "1"})
    composite.handle_tool_call("reserve", {"sku": "WIDGET-001", "qty": 5})

    report = composite.get_discovery_report()
    for t in report.transitions:
        print(f"  {t.state_before} --{t.action}--> {t.state_after}")

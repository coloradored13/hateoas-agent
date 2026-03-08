"""HATEOAS Agent — Visualization example.

Demonstrates generating Mermaid state diagrams from:
  1. A StateMachine definition (design-time view)
  2. A DiscoveryReport (runtime observed view)
"""

from hateoas_agent import StateMachine
from hateoas_agent.types import DiscoveryReport, TransitionRecord

# ============================================================
# 1. Visualize a StateMachine definition
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders")

orders.action(
    "approve_order",
    description="Approve this order for fulfillment",
    from_states=["pending"],
    to_state="approved",
    params={"order_id": "string"},
)
orders.action(
    "cancel_order",
    description="Cancel this order",
    from_states=["pending", "approved"],
    to_state="cancelled",
    params={"order_id": "string", "reason": "string"},
)
orders.action(
    "ship_order",
    description="Ship this order",
    from_states=["approved"],
    to_state="shipped",
    params={"order_id": "string"},
)
orders.action(
    "add_note",
    description="Add an internal note",
    from_states="*",
    params={"order_id": "string", "note": "string"},
)

print("=" * 60)
print("DESIGN-TIME DIAGRAM (from StateMachine definition)")
print("=" * 60)
print()
print(orders.to_mermaid())

# ============================================================
# 2. Visualize a DiscoveryReport (observed transitions)
# ============================================================

report = DiscoveryReport(transitions=[
    TransitionRecord("pending", "approve_order", "approved", 1.0),
    TransitionRecord("approved", "ship_order", "shipped", 2.0),
    TransitionRecord("pending", "cancel_order", "cancelled", 3.0),
    TransitionRecord("approved", "cancel_order", "cancelled", 4.0),
    TransitionRecord("shipped", "add_note", "shipped", 5.0),
])

print()
print("=" * 60)
print("RUNTIME DIAGRAM (from observed transitions)")
print("=" * 60)
print()
print(report.to_mermaid())

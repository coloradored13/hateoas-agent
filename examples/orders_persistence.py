"""HATEOAS Agent — Persistence example.

Demonstrates saving and restoring Registry state (checkpoints).
Useful for long-running conversations or session recovery.
"""

import json

from hateoas_agent import Registry, StateMachine
from hateoas_agent.persistence import (
    load_registry_checkpoint,
    save_registry_checkpoint,
)
from hateoas_agent.types import DiscoveryReport

# ============================================================
# Define a simple order state machine
# ============================================================

orders = StateMachine("orders", gateway_name="query_orders")
orders.gateway(description="Query orders", params={"order_id": "string"})
orders.action("approve", description="Approve", from_states=["pending"],
              to_state="approved", params={"order_id": "string"})
orders.action("ship", description="Ship", from_states=["approved"],
              to_state="shipped", params={"order_id": "string"})


@orders.on_gateway
def query(order_id=None):
    return {"order": {"id": order_id, "status": "pending"}, "_state": "pending"}


@orders.on_action("approve")
def approve(order_id=""):
    return {"approved": True, "_state": "approved"}


@orders.on_action("ship")
def ship(order_id=""):
    return {"shipped": True, "_state": "shipped"}


# ============================================================
# Demo: Save and restore
# ============================================================

if __name__ == "__main__":
    # Session 1: Process some orders
    print("=" * 60)
    print("SESSION 1: Process orders")
    print("=" * 60)

    reg = Registry(orders)
    reg.handle_tool_call("query_orders", {"order_id": "42"})
    print(f"State after query: {reg._last_state}")

    reg.handle_tool_call("approve", {"order_id": "42"})
    print(f"State after approve: {reg._last_state}")

    # Save checkpoint
    checkpoint = save_registry_checkpoint(reg)
    checkpoint_json = json.dumps(checkpoint, indent=2)
    print(f"\nSaved checkpoint ({len(checkpoint_json)} bytes):")
    print(checkpoint_json[:200] + "...")

    # Also save discovery report
    report = reg.get_discovery_report()
    report_json = report.to_json()
    print(f"\nDiscovery report: {report_json}")

    # Session 2: Restore and continue
    print("\n" + "=" * 60)
    print("SESSION 2: Restore and continue")
    print("=" * 60)

    reg2 = Registry(orders)
    print(f"Fresh registry state: {reg2._last_state}")

    load_registry_checkpoint(reg2, json.loads(checkpoint_json))
    print(f"Restored state: {reg2._last_state}")
    print(f"Restored transitions: {len(reg2._transition_log)}")

    # Continue from restored state
    reg2.handle_tool_call("ship", {"order_id": "42"})
    print(f"State after ship: {reg2._last_state}")

    # Verify full history
    full_report = reg2.get_discovery_report()
    print(f"\nFull transition history ({len(full_report.transitions)} transitions):")
    for t in full_report.transitions:
        print(f"  {t.state_before} --{t.action}--> {t.state_after}")

    # Restore discovery report
    restored_report = DiscoveryReport.from_json(report_json)
    print(f"\nRestored report state map: {restored_report.to_state_map()}")

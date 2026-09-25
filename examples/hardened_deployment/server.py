"""Hardened deployment — the gate process.

This process is the ONLY thing that holds the resource's credentials. It runs
the hateoas-agent state gate over MCP stdio. The agent (the MCP client) connects
to it and can only send tool calls; it has no direct path to the resource.

In a real deployment, run this as a dedicated OS user (see run_hardened.sh) whose
environment carries the DB password / file path / API key. The agent process is
started WITHOUT those credentials, so "action valid for the current state" is
enforced by the OS boundary, not merely by the Registry contract.

See ../../SECURITY.md for the full rationale.

Usage:
    pip install 'hateoas-agent[mcp]'
    python examples/hardened_deployment/server.py
"""

import os

from hateoas_agent import StateMachine

# The resource credential lives ONLY in this process's environment. The agent
# process must not have it. (Here it's a stand-in; in production it would be a
# DB DSN, an API key, etc.)
RESOURCE_CREDENTIAL = os.environ.get("RESOURCE_DB_DSN")

orders = StateMachine("orders", gateway_name="query_orders")

orders.gateway(
    description="Search and retrieve orders. Starting point for all order operations.",
    params={"order_id": "string"},
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
    "ship_order",
    description="Mark order as shipped",
    from_states=["approved"],
    to_state="shipped",
    params={"order_id": "string"},
    required=["order_id"],
)


def _db():
    """The resource. Reachable only with RESOURCE_CREDENTIAL, held by this process."""
    if not RESOURCE_CREDENTIAL:
        raise RuntimeError(
            "This gate process has no resource credential. In a hardened "
            "deployment RESOURCE_DB_DSN is set for the gate user only."
        )
    # A real handler would connect using RESOURCE_CREDENTIAL here. The handler is
    # also the right place to re-verify the entity's real state against the
    # source of truth (defense in depth) — the machine only gates reachability.
    return {"1001": {"id": "1001", "status": "pending"}}


@orders.on_gateway
def handle_query(order_id=None):
    row = _db().get(order_id or "1001")
    return {"order": row, "_state": row["status"]}


@orders.on_action("approve_order")
def handle_approve(order_id):
    return {"success": True, "_state": "approved"}


@orders.on_action("ship_order")
def handle_ship(order_id):
    return {"success": True, "_state": "shipped"}


if __name__ == "__main__":
    from hateoas_agent.mcp_server import serve

    # strict_transitions defaults to True (v0.3); the gateway/actions above are
    # all gated. The agent on the other end of stdio has no other path in.
    serve(orders, name="orders-hardened")

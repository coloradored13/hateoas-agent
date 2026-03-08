# hateoas-agent

**Dynamic tool discovery for AI agents. Start with one tool. Let the API tell your agent what to do next.**

Other frameworks give your LLM a flat list of tools and hope it picks the right one. With 10 tools, that works. With 50, accuracy drops. With 100+, your agent hallucinates tool names, calls things out of order, and you're left writing defensive prompts begging it to behave.

hateoas-agent applies the same principle that makes the web work — [HATEOAS](https://en.wikipedia.org/wiki/HATEOAS) (Hypermedia As The Engine Of Application State). Your agent starts with a single gateway tool. Each response tells it exactly what actions are available *right now*, based on the current state. No guessing. No searching. No hallucinating.

> **Other frameworks make the LLM better at choosing from too many tools. hateoas-agent makes the wrong choice impossible.**

## How it compares

| | Flat tools | Tool RAG | Tool Search | **hateoas-agent** |
|---|---|---|---|---|
| Tool selection | LLM picks from all | Semantic retrieval | LLM writes query | **Server declares what's valid** |
| Correctness | Hope-based | Probabilistic | Probabilistic | **Deterministic** |
| Security | Prompt-based | None | None | **Server-side enforcement** |
| Tools visible per turn | All (100+) | ~10-20 | ~5-10 | **3-5 (only what's valid)** |
| Model dependency | Any | Any | Provider-specific | **Any** |

## Install

```bash
pip install hateoas-agent                  # core library (no LLM dependency)
pip install 'hateoas-agent[anthropic]'     # with Claude Runner support
pip install 'hateoas-agent[mcp]'           # with MCP server support
```

## Quick start

```python
from hateoas_agent import StateMachine, Runner

# In-memory database
db = {
    "4521": {"id": "4521", "customer": "Jane Smith", "total": 89.99, "status": "pending"},
    "4522": {"id": "4522", "customer": "Bob Jones", "total": 249.99, "status": "shipped"},
}

orders = StateMachine("orders", gateway_name="query_orders")

orders.gateway(
    description="Search and retrieve orders",
    params={"order_id": "string"},
)

orders.action("approve_order",
    description="Approve this order",
    from_states=["pending"],
    to_state="approved",
    params={"order_id": "string"},
)

orders.action("cancel_order",
    description="Cancel this order",
    from_states=["pending", "approved"],
    to_state="cancelled",
    params={"order_id": "string", "reason": "string"},
)

orders.action("ship_order",
    description="Ship this order",
    from_states=["approved"],
    to_state="shipped",
    params={"order_id": "string", "tracking": "string"},
)

orders.action("add_note",
    description="Add an internal note",
    from_states="*",  # available in every state
    params={"order_id": "string", "note": "string"},
)

@orders.on_gateway
def handle_query(order_id=None):
    order = db[order_id]
    return {"order": order, "_state": order["status"]}

@orders.on_action("approve_order")
def handle_approve(order_id):
    db[order_id]["status"] = "approved"
    return {"success": True, "_state": "approved"}

@orders.on_action("cancel_order")
def handle_cancel(order_id, reason=""):
    db[order_id]["status"] = "cancelled"
    return {"success": True, "_state": "cancelled"}

@orders.on_action("ship_order")
def handle_ship(order_id, tracking=""):
    db[order_id]["status"] = "shipped"
    return {"success": True, "_state": "shipped"}

@orders.on_action("add_note")
def handle_note(order_id, note=""):
    return {"noted": True, "_state": db[order_id]["status"]}

runner = Runner(orders, model="claude-sonnet-4-20250514")
result = runner.run("Look up order 4521 and approve it.")
```

The agent can't call `ship_order` on a cancelled order — not because you told it not to, but because it **never sees the option**.

## Three ways to define your state machine

### Action-centric (recommended)

Define actions with their transition rules. The framework builds the state graph:

```python
sm.action("approve", from_states=["pending"], to_state="approved", ...)
sm.action("ship",    from_states=["approved"], to_state="shipped", ...)
sm.action("cancel",  from_states=["pending", "approved"], to_state="cancelled", ...)
sm.action("add_note", from_states="*", ...)  # available everywhere
```

### State-centric

Define states with their available actions (original API, still fully supported):

```python
sm.state("pending", actions=[
    {"name": "approve", "description": "Approve", "params": {"order_id": "string"}},
    {"name": "cancel", "description": "Cancel", "params": {"order_id": "string"}},
])
sm.state("approved", actions=[
    {"name": "ship", "description": "Ship", "params": {"order_id": "string"}},
])
```

### Class-based (Resource API)

Use decorators for an object-oriented style:

```python
class OrderResource(Resource):
    @gateway(name="query_orders", description="Search orders", params={...})
    def query(self, order_id=None): ...

    @action(name="approve", description="Approve", params={...})
    @state("pending")
    def approve(self, order_id): ...
```

All three styles can be mixed and produce identical runtime behavior.

## Discovery mode

Don't know your state graph yet? Run wide open, observe what happens, then lock it down:

```python
# Start with no constraints
orders = StateMachine("orders", gateway_name="query_orders", mode="discover")

# Define actions without from_states — all are available everywhere
orders.action("approve_order", description="Approve", params={"order_id": "string"})
orders.action("ship_order", description="Ship", params={"order_id": "string"})
orders.action("cancel_order", description="Cancel", params={"order_id": "string"})

# ... define handlers that return _state as usual ...

# Run your agent against real scenarios
runner = Runner(orders, model="claude-sonnet-4-20250514")
runner.run("Approve order 123 and ship it.")
runner.run("Cancel order 456.")

# Auto-generate the state machine from observed transitions
report = runner.get_discovery_report()

print(report.to_state_map())
# {'approved': ['ship_order'], 'pending': ['approve_order', 'cancel_order']}

print(report.to_python("orders"))
# orders.action("approve_order",
#     description="...",
#     from_states=['pending'],
#     to_state="approved",
# )
# ...ready-to-use .action() code
```

The progression: **discover** (zero config) → **observe** (auto-suggest constraints) → **lock down** (strict mode).

## How `_state` drives everything

The `_state` key in your handler's return dict is how the framework knows what state you're in. It controls which actions get advertised to the LLM on the next turn:

```python
@orders.on_action("approve_order")
def handle_approve(order_id):
    db[order_id]["status"] = "approved"
    return {"success": True, "_state": "approved"}  # ← this drives the next set of actions
```

**Key rules:**
- Every handler should return a dict with `_state` set to the current state name
- `_state` is stripped before results reach the LLM — it's internal plumbing, not user-visible data
- If you omit `_state`, the state stays unchanged and a warning is logged
- `_state` must be a string — non-string values raise `TypeError`

If you're using the action-centric API with `to_state` metadata, the framework will also warn if your handler returns a `_state` that doesn't match the declared `to_state` — useful for catching bugs early.

## Security

The framework uses server-side state validation — not prompts — to enforce correctness. Every action is checked against the current state before execution.

**Defense layers:**

- **Server-side state validation** — actions called in the wrong state or before any gateway call are rejected.
- **Phantom tool detection** — if the agent fabricates a tool name, it's caught and rejected.
- **Defensive system prompt** — instructs the LLM to only use actions from the most recent tool result.
- **Parameter filtering** — handlers only receive declared parameters. Extra keys are stripped.
- **State type validation** — `_state` must be a string. Non-string values raise `TypeError`.

**Callbacks and strict mode:**

```python
runner = Runner(
    orders,
    on_phantom_tool=lambda name, inp, state: log.warning(f"Phantom: {name}"),
    on_invalid_action=lambda name, inp, state: log.info(f"Invalid: {name} in {state}"),
    on_transition=lambda old, action, new: log.info(f"{old} -> {new}"),
    strict=True,  # raise PhantomToolError instead of returning error to LLM
)
```

- `on_phantom_tool` — agent called a tool that doesn't exist in any state (hallucination or injection).
- `on_invalid_action` — agent called a real action that isn't valid for the current state.
- `on_transition` — fires on every state change for logging/auditing.

## When to use it

- You have a domain with **stateful workflows** (orders, tickets, deployments, approvals)
- Your tool count is **growing beyond what an LLM can reliably choose from**
- You need **server-side guarantees** that the agent can't take invalid actions
- You want to **prototype fast** (discovery mode) and **lock down later** (strict mode)

## Examples

See `examples/` for complete working examples:

- `orders_action_centric.py` — action-centric API with transition rules
- `orders_declarative.py` — state-centric API (original style)
- `orders_resource.py` — class-based Resource API
- `orders_with_guards.py` — conditional actions with guard functions
- `orders_discovery.py` — discovery mode with auto-generated state machine
- `orders_visualization.py` — Mermaid diagram generation
- `orders_persistence.py` — checkpoint and restore state
- `orders_mcp.py` — MCP server integration
- `multi_resource.py` — composing multiple resources
- `comparison_flat_vs_hateoas.py` — flat tools vs HATEOAS side-by-side
- `database_admin.py` — SQLite admin tool with state-driven navigation
- `database_admin_api.py` — database admin with action-centric API

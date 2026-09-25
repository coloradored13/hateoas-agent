# Security model

hateoas-agent gates which actions an agent may invoke based on the current
state. This document is precise about **what that gate does and does not
guarantee**, so integrators don't over-trust it.

## What the gate guarantees

Given a single `Registry` (or `CompositeRegistry`) as the only path between the
agent and the resource:

- An action is executed only if it is valid for the current state. Wrong-state
  known actions raise `InvalidActionError`; unknown tool names are phantom calls.
  This is enforced **server-side** in `Registry._handle_action` — not by hiding
  tools from the model. Even if the model ignores the advertised tool list and
  emits an arbitrary tool call, the gate rejects it.
- The model cannot smuggle a `_state` into tool input to flip the gate:
  undeclared params are stripped before the handler runs.
- With `strict_transitions=True` (the default as of v0.3), a handler that returns
  a `_state` other than its declared `to_state` is rejected and the bad state is
  not committed. With `enforce_known_states=True`, a returned state outside the
  resource's declared vocabulary is rejected as well.
- The `Orchestrator` gateway cannot be used to teleport into a later phase or to
  inject context that a transition guard reads (v0.3).

## What the gate does NOT guarantee

**In-process gating is the agent-facing contract, not a security boundary.**

The gate constrains an agent that reaches the resource *only* through the
Registry's tool calls. It cannot constrain code that shares the same process,
shell, and credentials as the resource. If the agent can:

- import the resource module and call a handler (or `_get_handler`) directly,
- read or write the underlying files / database with the same credentials, or
- shell out to tools that touch the resource,

then it is *below* the handler layer and the state gate does not apply. This is
not a bug that a code change can fix — it is a property of running the gate and
the resource in the same trust domain.

The corollary: **the gate is hard only across a process/credential boundary.**

## Recommended hard-gate deployment

Run the gate as a separate process/OS user that holds the *sole* credentials to
the resource, and give the agent only the MCP channel to it:

```
┌────────────────┐     MCP stdio (tool calls only)     ┌───────────────────────┐
│  Agent process │ ──────────────────────────────────▶ │  hateoas-agent server │
│  (no DB/FS creds)│                                    │  runs as `gate` user   │
└────────────────┘                                     │  holds sole DB/FS creds │
                                                        └───────────┬───────────┘
                                                                    │ credentials
                                                                    ▼
                                                            ┌───────────────┐
                                                            │   Resource     │
                                                            │ (DB / files)   │
                                                            └───────────────┘
```

- The agent process runs with **no** direct access to the resource — no DB
  password, no filesystem path, no API key. Its only capability is to send tool
  calls over MCP stdio.
- The server process (`hateoas_agent.mcp_server.serve`) runs as a dedicated user
  that owns the resource credentials. Every tool call passes through the gate.
- Because the agent has no other path to the resource, "valid action for the
  current state" becomes an actual boundary, not just a contract.

See `examples/hardened_deployment/` for a runnable sketch of this split.

## Defense in depth: verify entity state in the handler

The machine gates **reachability** (is this action allowed from this state);
the handler should still **verify the entity's real state against the source of
truth** before performing a privileged effect. `_last_state` is one in-memory
value per Registry and can be stale under server restarts or concurrent clients;
the database constraint (or equivalent invariant) is what actually protects the
resource. Treat the machine gate as the agent-facing contract and the per-entity
check as the invariant.

## Reporting

Please report security issues via the repository's Issues page, or privately to
the maintainer if the issue is sensitive.

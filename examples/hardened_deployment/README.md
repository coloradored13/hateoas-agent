# Hardened deployment — the gate across a process/credential boundary

hateoas-agent's state gate is only a *hard* boundary when the agent cannot reach
the resource except through the gate. That means running the gate as a **separate
process/OS user that holds the sole credentials** to the resource, and giving the
agent only the MCP channel. See `../../SECURITY.md` for the full rationale.

This directory sketches that split:

- `server.py` — the gate. Runs `hateoas_agent.mcp_server.serve()` over stdio.
  In a real deployment this process runs as a dedicated user (e.g. `gate`) that
  owns the database password / filesystem path / API key. Nothing else does.
- `run_hardened.sh` — launches the server as a locked-down user with the resource
  credentials in *its* environment only, illustrating the credential split.

## The security property

```
Agent process (no creds)  ──MCP stdio──▶  gate process (sole creds)  ──▶  resource
```

- The agent process is started with **no** resource credentials in its
  environment and **no** direct path (no DB DSN, no file path). Its only
  capability is to send tool calls to the gate over stdio.
- The gate process validates every call against the current state before it ever
  touches the resource with its credentials.

Because the agent has no second path to the resource, "action valid for the
current state" is enforced by the OS boundary, not merely by the Registry
contract. An agent that tries to bypass the gate has nothing to bypass it *with*
— it holds no credentials.

## What breaks the property (anti-patterns)

- Importing the resource module into the agent process and calling handlers
  directly (or `_get_handler`) — now below the gate.
- Handing the agent process the same DB/FS credentials "for convenience".
- Letting the agent shell out to tools that carry the resource credentials.

If the agent and the resource share a trust domain, the gate is an advisory
contract, not a boundary — this is a deployment property, not something a code
change can enforce.

## Run the sketch

```bash
# The gate process gets the credentials; the agent (your MCP client) does not.
./run_hardened.sh
```

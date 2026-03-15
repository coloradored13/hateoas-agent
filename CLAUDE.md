# hateoas-agent

Dynamic tool discovery for AI agents using HATEOAS-style action advertisement.

## Quick reference

- **Source**: `src/hateoas_agent/` (13 modules)
- **Tests**: `tests/` — run with `python3 -m pytest tests/ -q`
- **Lint**: `python3 -m ruff check src/ tests/`
- **Python**: >=3.10
- **Dependencies**: zero runtime deps. Optional: `anthropic`, `mcp`, `pytest-asyncio`

## Architecture

Two API surfaces that both implement the `HasHateoas` protocol:

| API | Entry point | Use case |
|-----|------------|----------|
| **Declarative** | `StateMachine` | Define states/actions with `.state()` / `.action()` |
| **Handler-based** | `Resource` | Define with `@gateway` / `@action` / `@state` decorators |
| **Orchestration** | `Orchestrator` | Multi-agent workflows — phases as states, transitions with guards |

All three plug into `Registry` (tool routing), `Runner` (Claude API loop), MCP server, persistence, and visualization.

## v0.2 Orchestration modules

- `agent_slot.py` — `AgentStatus`, `AgentSlot`, `AgentResult`
- `orchestrator.py` — `Orchestrator` implementing `HasHateoas`. Phases, transitions, guards, `run_agent()`, `run_agents_parallel()`
- `async_runner.py` — `AsyncRunner` drives orchestrator to completion with async handler support
- `conditions.py` — Composable guard factories: `all_converged()`, `belief_above()`, `exit_gate_passed()`, `gap_count_below()`, `round_limit()`. Compose with `&` `|` `~`
- `orchestrator_persistence.py` — `OrchestratorCheckpoint`, `save_orchestrator_checkpoint()`, `load_orchestrator_checkpoint()`
- `orchestrator_visualization.py` — `orchestrator_to_mermaid()` with guard labels + agent annotations

## Key design decisions

- Orchestrator IS a HasHateoas state machine (not a separate abstraction)
- Agents are `AgentSlot` dataclasses, NOT their own state machines
- Self-loop transitions (e.g., challenge → challenge) work through Registry
- Guards are `Callable[[dict], bool]` — use `conditions.py` factories or plain lambdas
- Zero breaking changes from v0.1

## Testing

```
python3 -m pytest tests/ -q          # all tests
python3 -m pytest tests/ -k async    # async tests only
python3 -m pytest tests/ -k orch     # orchestration tests only
```

Adversarial tests (`test_adversarial.py`) are excluded via pyproject.toml — they require a live API key.

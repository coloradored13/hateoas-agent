# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Opt-in transition enforcement**: `Registry(resource, strict_transitions=True)`
  and `Runner(resource, strict_transitions=True)`. When enabled, a handler that
  returns a `_state` other than its action's declared `to_state` raises the new
  `StateTransitionError` and the mismatched state is **not** committed (the
  resource stays in its prior state). Default behavior is unchanged — the
  mismatch is logged as a warning and applied — so this is fully back-compatible.
- `StateTransitionError` exception (exported from the package).
- `tests/test_state_integrity.py` — formalizes the state-bypass investigation:
  state injection via tool input is blocked, author-supplied params piped into
  `_state` are pinned as a known footgun, and `to_state` enforcement is covered
  in both default and strict modes.
- PyPI release workflow (trusted publishing) triggered by `v*` tags.
- Nightly adversarial red-team workflow against the live Claude API
  (skips when no `ANTHROPIC_API_KEY` secret is configured).
- `CHANGELOG.md` and `CONTRIBUTING.md`.

### Changed
- Default `Runner` model updated from the deprecated `claude-sonnet-4-20250514`
  (retires June 15, 2026) to `claude-opus-4-8` across the runner, README, and
  all examples.

### Fixed
- CI installs the `anthropic` extra so the READ-297 regression tests run;
  those tests now skip gracefully when the optional SDK is absent.

## [0.2.0] - 2026-05-03

### Added
- **Multi-agent orchestration**: `Orchestrator` implementing the same
  `HasHateoas` protocol as `StateMachine` — phases as states, guarded
  transitions, self-loop rounds, `run_agent()` / `run_agents_parallel()`.
- `AgentSlot`, `AgentStatus`, `AgentResult` dataclasses for agent management,
  including `join_phase` for late-joining agents.
- `AsyncRunner` — drives an orchestrator to completion with async handler
  support.
- Composable guard factories in `conditions.py`: `all_converged()`,
  `belief_above()`, `exit_gate_passed()`, `gap_count_below()`,
  `round_limit()` — compose with `&` `|` `~`.
- Orchestrator persistence (`save_orchestrator_checkpoint()` /
  `load_orchestrator_checkpoint()`) and Mermaid visualization
  (`orchestrator_to_mermaid()`).

### Changed
- Deprecated the state-centric `StateMachine.state()` API in favor of the
  action-centric `.action(name, from_states=[...], to_state=...)` API
  (still fully supported; emits `DeprecationWarning`).

### Fixed
- Security hardening: context injection, parameter filtering, and
  discovery-mode warnings.
- Orchestrator guards seeing the wrong context shape under `Registry`.
- `Runner` API failures after SDK retries now raise `RunnerAPIError` with the
  conversation attached so callers can resume instead of starting over.
- Async phase handlers that were silently dropped now raise `TypeError`.

## [0.1.0] - 2026-04

### Added
- `StateMachine` declarative API with gateway, actions, states, and
  discovery mode.
- Handler-based `Resource` API with `@gateway` / `@action` / `@state`
  decorators.
- `Registry` tool routing with server-side state validation, phantom-tool
  detection, and parameter filtering.
- `Runner` — Claude API agent loop with security callbacks and strict mode.
- MCP server adapter (`hateoas_agent.mcp_server.serve()`) with
  `tools/list_changed` notifications.
- Runner persistence (checkpoint/restore), Mermaid visualization,
  `CompositeRegistry` for multi-resource composition.
- Adversarial red-team test suite.

[Unreleased]: https://github.com/coloradored13/hateoas-agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/coloradored13/hateoas-agent/releases/tag/v0.2.0
[0.1.0]: https://github.com/coloradored13/hateoas-agent/commits/main

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — Gating hardening

This release closes several ways an agent (or an integrator) could reach an
action the current state should forbid. Some changes are **breaking** — the gate
is now safe-by-default. See "Breaking changes" below for migration.

### Breaking changes
- **`get_handler` is now private (`_get_handler`).** A public `get_handler` on
  `StateMachine`/`Resource`/`Orchestrator` returned a raw handler callable that,
  invoked directly, skipped the Registry's state gate entirely — the first
  integrator reached for it and silently lost enforcement. Application code must
  now go through `Registry.handle_tool_call` or the new gated `invoke()`.
- **`strict_transitions` now defaults to `True`** on `Registry`, `Runner`, and
  `CompositeRegistry`. A handler that returns a `_state` other than its action's
  declared `to_state` now raises `StateTransitionError` and the bad state is not
  committed. Pass `strict_transitions=False` for the old warn-and-apply behavior.
- **The `Orchestrator` gateway no longer accepts LLM-supplied `phase` or
  `context`.** Previously an agent could call the start gateway with
  `phase="synthesis"` to teleport past guarded transitions, and inject `context`
  that a later transition guard reads. The gateway now always starts at
  `start_phase` (or the first phase). Set the starting phase/context via the
  trusted `start()` / `AsyncRunner` APIs. An explicit `allow_phase_targets`
  allowlist re-enables a validated `phase` param for advanced use.
- **`mode="discover"` now requires an explicit opt-in** (`allow_discover=True`
  or `HATEOAS_ALLOW_DISCOVER=1`), because discover mode disables state gating
  entirely (all actions in all states).

### Added
- **Gated `invoke()`** on `StateMachine`/`Resource`/`Orchestrator` (via
  `GatedInvokeMixin`): a single-call entry point that routes through an internal
  Registry so the gateway establishes state that later action calls are gated
  against. The supported replacement for reaching into a handler directly.
- **`preserves_state=True`** on `.action()` / `@action`: the Registry ignores any
  `_state` such an action's handler returns and keeps the current state. Use for
  read-only / universal actions so a stray `_state` can't silently re-gate a
  session (the failure mode behind sigma-mem's `f89aaf6` state-flip bug).
- **`enforce_known_states=True`** (opt-in) on `Registry`/`Runner`: additionally
  rejects a returned `_state` outside the resource's declared vocabulary
  (`get_known_states()`), for maximum-strictness gates.
- **`CompositeRegistry.get_current_actions()`** — previously missing, which
  crashed the multi-resource error-recovery path with `AttributeError` on any
  phantom/invalid tool call.
- **`Runner(strict=True)` now halts on wrong-state known actions**, at parity
  with phantom tools (previously only phantom tools halted under `strict`).
- `SECURITY.md` documenting the process/credential boundary (in-process gating is
  the agent-facing contract, not a security boundary) plus a hardened deployment
  recipe, and `examples/hardened_deployment/`.
- `tests/test_gating_hardening.py` — probes for every hardening item above,
  including the phase-teleport and `get_handler`-bypass reproductions.

## [Unreleased]

### Added
- **LLM-friendly recoverable errors**: when the model calls a wrong-state action
  or a phantom tool, the error response now inlines the currently-valid actions
  (via the previously-unused `format_error_with_actions`) so the model can
  self-correct in the same turn. This applies to both the `Runner` and the MCP
  server. Previously the MCP server masked these recoverable errors as a generic
  "An internal error occurred." — now only genuine handler crashes are generic,
  and strict-mode `StateTransitionError` still propagates as a developer bug.
- `Registry.get_current_actions()` — public accessor for the guard-filtered
  actions valid in the current state (empty before the gateway runs).
- `tests/test_error_responses.py` — covers friendly error responses across the
  Runner and MCP paths.
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

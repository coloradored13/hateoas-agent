"""Async workflow runner for orchestrated multi-agent workflows.

Drives an Orchestrator from initial phase to terminal, supporting
async phase handlers and parallel agent execution.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

from .orchestrator import Orchestrator, OrchestratorState

logger = logging.getLogger(__name__)


class AsyncRunner:
    """Runs an orchestrated workflow asynchronously to completion.

    Supports both sync and async phase handlers. When a phase handler
    is a coroutine function, it is awaited; otherwise it is called
    synchronously.

    Usage::

        review = Orchestrator(name="review", agents=[...])
        # ... define phases, transitions, handlers ...

        runner = AsyncRunner(review)
        final_state = await runner.run_orchestrated(
            context={"task": "Evaluate API design"}
        )
        assert final_state.is_terminal
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        max_iterations: int = 50,
    ):
        self._orch = orchestrator
        self._max_iterations = max_iterations

    @property
    def orchestrator(self) -> Orchestrator:
        """Return the wrapped orchestrator."""
        return self._orch

    async def run_orchestrated(
        self,
        *,
        phase: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorState:
        """Run the full workflow from initial phase to terminal.

        Starts the orchestrator in the given phase, then repeatedly
        evaluates guards and advances until a terminal phase is reached
        or no guard matches (workflow stalls).

        Args:
            phase: Starting phase. Defaults to the first defined phase.
            context: Initial context dict.

        Returns:
            Final OrchestratorState.
        """
        state = await self._async_start(phase, context=context)

        for _ in range(self._max_iterations):
            if state.is_terminal:
                break
            prev_len = len(state.phase_history)
            state = await self._async_advance()
            if len(state.phase_history) == prev_len:
                # No transition happened — all guards failed
                break

        return state

    async def _async_start(
        self,
        phase: Optional[str] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorState:
        """Start the orchestrator, supporting async phase handlers."""
        orch = self._orch
        target = phase or orch._first_phase
        if target is None:
            raise ValueError("No phase specified and no phases defined")
        if target not in orch._phases:
            raise ValueError(f"Unknown phase: {target!r}")

        orch._current_phase = target
        orch._context = dict(context or {})
        orch._phase_history = [target]

        await self._async_execute_phase(target)
        return orch._make_state()

    async def _async_advance(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorState:
        """Evaluate guards and advance, supporting async phase handlers."""
        orch = self._orch
        if orch._current_phase is None:
            raise ValueError("Orchestrator not started.")

        if context:
            orch._context.update(context)

        outgoing = [t for t in orch._transitions if t.from_phase == orch._current_phase]

        for trans in outgoing:
            if trans.guard is None or orch._eval_guard(trans.guard):
                orch._current_phase = trans.to_phase
                orch._phase_history.append(trans.to_phase)
                await self._async_execute_phase(trans.to_phase)
                return orch._make_state()

        return orch._make_state()

    async def _async_execute_phase(self, phase: str) -> None:
        """Run a phase handler, awaiting it if it's a coroutine function."""
        orch = self._orch
        handler = orch._phase_handlers.get(phase)
        if handler is None:
            return

        agents = orch.get_agents_for_phase(phase)

        if inspect.iscoroutinefunction(handler):
            result = await handler(orch, agents, orch._context)
        else:
            result = handler(orch, agents, orch._context)

        if isinstance(result, dict):
            orch._context.update(result)

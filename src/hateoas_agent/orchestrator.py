"""Multi-agent workflow orchestrator implementing HasHateoas.

The Orchestrator IS a state machine: review phases are states,
transitions are guarded by computed conditions. Because it implements
HasHateoas, it gets Registry, MCP server, persistence, and
visualization for free.

Agents are NOT their own state machines. They are AgentSlot dataclasses
managed by the orchestrator. Meaningful state transitions happen at
the workflow level, not per-agent.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .agent_slot import AgentResult, AgentSlot, AgentStatus
from .types import ActionDef, GatewayDef

logger = logging.getLogger(__name__)


@dataclass
class PhaseDef:
    """Internal definition of a workflow phase."""

    name: str
    description: str = ""
    parallel: bool = False
    agent_filter: Union[str, List[str]] = "*"


@dataclass
class TransitionDef:
    """Internal definition of a phase transition."""

    name: str
    from_phase: str
    to_phase: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class OrchestratorState:
    """Snapshot of the orchestrator's current runtime state."""

    name: str
    current_phase: str
    context: Dict[str, Any] = field(default_factory=dict)
    phase_history: List[str] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class PhaseResult:
    """Result from executing a workflow phase."""

    phase: str
    context: Dict[str, Any] = field(default_factory=dict)
    agent_results: List[AgentResult] = field(default_factory=list)


class Orchestrator:
    """Multi-agent workflow orchestrator implementing HasHateoas.

    Phases are states, transitions are guarded by conditions. Because it
    implements HasHateoas, an Orchestrator can be wrapped in a Registry
    for tool routing, exposed via MCP server, persisted, and visualized
    — all for free.

    Usage::

        review = Orchestrator(
            name="sigma-review",
            agents=[
                AgentSlot("tech-architect", role="Architecture analysis"),
                AgentSlot("product-strategist", role="Market viability"),
                AgentSlot("devils-advocate", role="Adversarial challenge",
                          join_phase="challenge"),
            ],
        )

        review.phase("research", parallel=True, agents="*")
        review.phase("challenge", parallel=True, agents="*")
        review.phase("synthesis", parallel=False, agents=["lead"])

        review.transition("research", "challenge",
            guard=lambda ctx: ctx.get("converged", False))
        review.transition("challenge", "synthesis",
            guard=lambda ctx: ctx.get("exit_gate") == "PASS")
        review.transition("challenge", "challenge",
            guard=lambda ctx: ctx.get("exit_gate") == "FAIL")

        @review.on_phase("research")
        def run_research(orchestrator, agents, context):
            for agent in agents:
                orchestrator.run_agent(agent, task=context["task"])
            return {"converged": True}

        state = review.start("research", context={"task": "analyze X"})
        state = review.advance()  # evaluates guards, transitions
    """

    def __init__(
        self,
        name: str,
        *,
        agents: Optional[List[AgentSlot]] = None,
        gateway_name: str = "start_workflow",
    ):
        self.name = name
        self._gateway_name = gateway_name

        # Agent registry
        self._agents: Dict[str, AgentSlot] = {}
        for agent in agents or []:
            self._agents[agent.name] = agent

        # Phase and transition definitions
        self._phases: Dict[str, PhaseDef] = {}
        self._transitions: List[TransitionDef] = []
        self._phase_handlers: Dict[str, Callable] = {}
        self._terminal_phases: set[str] = set()

        # Runtime state
        self._current_phase: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._phase_history: List[str] = []

        # Agent execution
        self._default_executor: Optional[Callable] = None
        self._agent_executors: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Definition API
    # ------------------------------------------------------------------

    def phase(
        self,
        name: str,
        *,
        description: str = "",
        parallel: bool = False,
        agents: Union[str, List[str]] = "*",
        terminal: bool = False,
    ) -> None:
        """Define a workflow phase.

        Args:
            name: Unique phase name (becomes a HATEOAS state).
            description: Human-readable description.
            parallel: Hint that agents in this phase can run in parallel
                (Phase 2 async_runner respects this; Phase 1 ignores it).
            agents: ``"*"`` for all agents, or a list of agent names.
            terminal: If True, the workflow ends when this phase completes.
        """
        self._phases[name] = PhaseDef(
            name=name,
            description=description,
            parallel=parallel,
            agent_filter=agents,
        )
        if terminal:
            self._terminal_phases.add(name)

    def transition(
        self,
        from_phase: str,
        to_phase: str,
        *,
        guard: Optional[Callable[[Dict[str, Any]], bool]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Define a phase transition with an optional guard condition.

        Guards receive the current context dict and return True to allow
        the transition. Transitions are evaluated in definition order;
        the first matching guard wins.

        Args:
            from_phase: Source phase name.
            to_phase: Target phase name.
            guard: ``Callable[[dict], bool]`` or None (always passes).
            name: Optional explicit name. Auto-generated if omitted.
        """
        if name is None:
            base = f"{from_phase}_to_{to_phase}"
            existing = [t for t in self._transitions if t.name.startswith(base)]
            name = f"{base}_{len(existing)}" if existing else base

        self._transitions.append(
            TransitionDef(
                name=name,
                from_phase=from_phase,
                to_phase=to_phase,
                guard=guard,
            )
        )

    def on_phase(self, phase_name: str) -> Callable:
        """Decorator to register a handler for a phase.

        The handler signature is ``(orchestrator, agents, context) -> dict``.
        The returned dict is merged into the orchestrator's context.
        """

        def decorator(fn: Callable) -> Callable:
            self._phase_handlers[phase_name] = fn
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, agent: AgentSlot) -> None:
        """Add an agent to the orchestrator."""
        self._agents[agent.name] = agent

    def remove_agent(self, name: str) -> AgentSlot:
        """Remove and return an agent by name."""
        return self._agents.pop(name)

    def get_agent(self, name: str) -> AgentSlot:
        """Return an agent by name."""
        return self._agents[name]

    @property
    def agents(self) -> Dict[str, AgentSlot]:
        """Return all registered agents."""
        return dict(self._agents)

    def get_agents_for_phase(self, phase: str) -> List[AgentSlot]:
        """Return agents that participate in a given phase.

        Filters by both ``join_phase`` (agents join from that phase onward)
        and the phase's ``agent_filter`` setting.
        """
        phase_def = self._phases.get(phase)
        if not phase_def:
            return []

        # Filter by join_phase: include if no join_phase or current >= join
        active: List[AgentSlot] = []
        for agent in self._agents.values():
            if agent.join_phase is None:
                active.append(agent)
            elif self._phase_is_at_or_after(phase, agent.join_phase):
                active.append(agent)

        # Filter by phase's agent_filter
        if phase_def.agent_filter == "*":
            return active
        if isinstance(phase_def.agent_filter, list):
            return [a for a in active if a.name in phase_def.agent_filter]
        return active

    def _phase_is_at_or_after(self, current: str, join: str) -> bool:
        """Check if current phase is at or after join phase in definition order."""
        phase_order = list(self._phases.keys())
        if current not in phase_order or join not in phase_order:
            return True  # unknown phase: include agent
        return phase_order.index(current) >= phase_order.index(join)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start(
        self,
        phase: Optional[str] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorState:
        """Initialize the orchestrator in a phase and run its handler.

        Args:
            phase: Starting phase. Defaults to the first defined phase.
            context: Initial context dict.

        Returns:
            OrchestratorState snapshot after the phase handler runs.
        """
        target = phase or self._first_phase
        if target is None:
            raise ValueError("No phase specified and no phases defined")
        if target not in self._phases:
            raise ValueError(f"Unknown phase: {target!r}")

        self._current_phase = target
        self._context = dict(context or {})
        self._phase_history = [target]

        self._execute_phase(target)
        return self._make_state()

    def advance(self, *, context: Optional[Dict[str, Any]] = None) -> OrchestratorState:
        """Evaluate guards from current phase and transition to the next.

        Transitions are evaluated in definition order. The first guard
        that returns True (or the first unguarded transition) wins.
        If no guard matches, the orchestrator stays in the current phase.

        Args:
            context: Optional context updates merged before guard evaluation.

        Returns:
            OrchestratorState snapshot after transition (or same state if
            no guard matched).
        """
        if self._current_phase is None:
            raise ValueError("Orchestrator not started. Call start() first.")

        if context:
            self._context.update(context)

        outgoing = [t for t in self._transitions if t.from_phase == self._current_phase]

        for trans in outgoing:
            if trans.guard is None or self._eval_guard(trans.guard):
                self._current_phase = trans.to_phase
                self._phase_history.append(trans.to_phase)
                self._execute_phase(trans.to_phase)
                return self._make_state()

        # No guard matched — stay in current phase
        return self._make_state()

    def run_agent(
        self,
        agent: AgentSlot,
        *,
        task: Any = None,
        executor: Optional[Callable] = None,
    ) -> AgentResult:
        """Execute a single agent sequentially.

        The executor callable signature is ``(agent, task) -> output``.
        Uses the agent-specific executor, then the default executor,
        then the provided executor argument.

        Args:
            agent: The agent slot to execute.
            task: Task data passed to the executor.
            executor: Override executor for this call.
        """
        exec_fn = executor or self._agent_executors.get(agent.name) or self._default_executor
        if exec_fn is None:
            raise ValueError(
                f"No executor for agent '{agent.name}'. Provide an executor or call set_executor()."
            )

        agent.status = AgentStatus.RUNNING
        try:
            output = exec_fn(agent, task)
            agent.status = AgentStatus.CONVERGED
            return AgentResult(
                agent_name=agent.name,
                output=output,
                status=AgentStatus.CONVERGED,
            )
        except Exception as e:
            agent.status = AgentStatus.ERROR
            return AgentResult(
                agent_name=agent.name,
                status=AgentStatus.ERROR,
                error=str(e),
            )

    async def run_agents_parallel(
        self,
        agents: List[AgentSlot],
        *,
        task: Any = None,
        executor: Optional[Callable] = None,
        timeout: Optional[float] = None,
    ) -> List[AgentResult]:
        """Execute multiple agents in parallel via asyncio.gather.

        Error policy: if one agent errors, partial results are collected,
        the errored agent is marked ERROR, but the phase doesn't fail.

        Args:
            agents: List of agent slots to execute.
            task: Task data passed to each executor.
            executor: Override executor for all agents in this call.
            timeout: Per-agent timeout in seconds. None means no timeout.

        Returns:
            List of AgentResult in the same order as agents.
        """
        import asyncio

        async def _run_one(agent: AgentSlot) -> AgentResult:
            exec_fn = executor or self._agent_executors.get(agent.name) or self._default_executor
            if exec_fn is None:
                return AgentResult(
                    agent_name=agent.name,
                    status=AgentStatus.ERROR,
                    error=f"No executor for agent '{agent.name}'",
                )

            agent.status = AgentStatus.RUNNING
            try:
                if inspect.iscoroutinefunction(exec_fn):
                    coro = exec_fn(agent, task)
                else:
                    coro = asyncio.to_thread(exec_fn, agent, task)

                if timeout is not None:
                    output = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    output = await coro

                agent.status = AgentStatus.CONVERGED
                return AgentResult(
                    agent_name=agent.name,
                    output=output,
                    status=AgentStatus.CONVERGED,
                )
            except asyncio.TimeoutError:
                agent.status = AgentStatus.ERROR
                return AgentResult(
                    agent_name=agent.name,
                    status=AgentStatus.ERROR,
                    error=f"Timeout after {timeout}s",
                )
            except Exception as e:
                agent.status = AgentStatus.ERROR
                return AgentResult(
                    agent_name=agent.name,
                    status=AgentStatus.ERROR,
                    error=str(e),
                )

        coros = [_run_one(agent) for agent in agents]
        return list(await asyncio.gather(*coros))

    def set_executor(
        self,
        fn: Callable,
        *,
        agent_name: Optional[str] = None,
    ) -> None:
        """Register an executor function for running agents.

        Args:
            fn: Callable with signature ``(agent, task) -> output``.
            agent_name: If provided, registers for that agent only.
                Otherwise sets the default executor.
        """
        if agent_name:
            self._agent_executors[agent_name] = fn
        else:
            self._default_executor = fn

    # ------------------------------------------------------------------
    # HasHateoas protocol
    # ------------------------------------------------------------------

    def get_gateway(self) -> Optional[GatewayDef]:
        """Return a gateway tool that starts the orchestrator workflow."""

        def handler(**kwargs: Any) -> Dict[str, Any]:
            phase = kwargs.get("phase") or self._first_phase
            ctx = self._parse_tool_context(kwargs.get("context"))
            state = self.start(phase, context=ctx)
            return {
                "phase": state.current_phase,
                "context": state.context,
                "agents": [a.name for a in self.get_agents_for_phase(state.current_phase)],
                "_state": state.current_phase,
            }

        return GatewayDef(
            name=self._gateway_name,
            description=f"Start the {self.name} workflow",
            params={"phase": "string", "context": "string"},
            handler=handler,
        )

    def get_actions_for_state(self, state: str) -> List[ActionDef]:
        """Return available transitions from a phase as HATEOAS actions."""
        outgoing = [t for t in self._transitions if t.from_phase == state]
        actions: List[ActionDef] = []

        for trans in outgoing:
            actions.append(
                ActionDef(
                    name=trans.name,
                    description=f"Advance from {trans.from_phase} to {trans.to_phase}",
                    params={"context": "string"},
                    handler=self._make_transition_handler(trans),
                )
            )

        # Add generic "advance" action that evaluates all guards
        if outgoing:
            actions.append(
                ActionDef(
                    name="advance",
                    description="Evaluate guards and advance to the next phase",
                    params={"context": "string"},
                    handler=self._make_advance_handler(),
                )
            )

        return actions

    def get_handler(self, action_name: str) -> Optional[Callable]:
        """Return handler for a transition action name."""
        if action_name == "advance":
            return self._make_advance_handler()
        for trans in self._transitions:
            if trans.name == action_name:
                return self._make_transition_handler(trans)
        return None

    def get_all_action_names(self) -> set[str]:
        """Return all transition action names plus 'advance'."""
        names = {t.name for t in self._transitions}
        if self._transitions:
            names.add("advance")
        return names

    def validate(self) -> None:
        """Validate orchestrator configuration.

        Raises:
            ValueError: If no phases defined, or transitions reference
                unknown phases.
        """
        if not self._phases:
            raise ValueError(f"Orchestrator '{self.name}' has no phases defined.")
        for trans in self._transitions:
            if trans.from_phase not in self._phases:
                raise ValueError(
                    f"Transition '{trans.name}' references unknown phase '{trans.from_phase}'"
                )
            if trans.to_phase not in self._phases:
                raise ValueError(
                    f"Transition '{trans.name}' references unknown phase '{trans.to_phase}'"
                )

    def filter_actions(
        self,
        actions: List[ActionDef],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ActionDef]:
        """Filter transition actions by evaluating their guards.

        Always evaluates against the orchestrator's own ``_context``,
        which is the canonical source of truth for guards composed via
        ``conditions.py`` factories. The ``context`` parameter exists
        only to satisfy the ``HasHateoas`` protocol; values passed by
        Registry (the most recent handler return dict, shaped as
        ``{phase, context, agents, _state}``) are ignored to avoid
        silently mismatching guard expectations.
        """
        del context  # see docstring; orchestrator._context is canonical
        eval_ctx = self._context
        result: List[ActionDef] = []
        for action_def in actions:
            if action_def.name == "advance":
                # Always show advance if there are any outgoing transitions
                result.append(action_def)
                continue
            trans = next(
                (t for t in self._transitions if t.name == action_def.name),
                None,
            )
            if trans is None or trans.guard is None:
                result.append(action_def)
                continue
            try:
                if trans.guard(eval_ctx):
                    result.append(action_def)
            except Exception:
                logger.warning(
                    "Guard for transition '%s' raised; excluding",
                    action_def.name,
                    exc_info=True,
                )
        return result

    def get_transition_metadata(
        self, action_name: str
    ) -> Optional[Tuple[Union[List[str], str], Optional[str]]]:
        """Return ``(from_states, to_state)`` for a transition action."""
        if action_name == "advance":
            # advance can originate from any phase that has outgoing transitions
            from_phases = sorted({t.from_phase for t in self._transitions})
            return (from_phases, None)
        for trans in self._transitions:
            if trans.name == action_name:
                return ([trans.from_phase], trans.to_phase)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _first_phase(self) -> Optional[str]:
        """Return the first defined phase name, or None."""
        return next(iter(self._phases), None)

    def _eval_guard(self, guard: Callable[[Dict[str, Any]], bool]) -> bool:
        """Evaluate a guard function against current context."""
        try:
            return guard(self._context)
        except Exception:
            logger.warning("Guard raised an exception; treating as False", exc_info=True)
            return False

    def _execute_phase(self, phase: str) -> None:
        """Run the phase handler if registered, merging result into context."""
        handler = self._phase_handlers.get(phase)
        if handler is None:
            return

        agents = self.get_agents_for_phase(phase)
        result = handler(self, agents, self._context)

        if isinstance(result, dict):
            self._context.update(result)

    def _make_state(self) -> OrchestratorState:
        """Build an OrchestratorState snapshot."""
        return OrchestratorState(
            name=self.name,
            current_phase=self._current_phase,
            context=dict(self._context),
            phase_history=list(self._phase_history),
            is_terminal=self._current_phase in self._terminal_phases,
        )

    @staticmethod
    def _parse_tool_context(raw: Any) -> Dict[str, Any]:
        """Parse and validate a context parameter from a tool call.

        Returns an empty dict for invalid input rather than raising.
        """
        if not raw:
            return {}
        if isinstance(raw, str):
            import json

            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        if isinstance(raw, dict):
            return raw
        return {}

    def _make_transition_handler(self, trans: TransitionDef) -> Callable[..., Dict[str, Any]]:
        """Create a handler callable for a specific transition.

        Tool-provided context is stored under ``_tool_context`` and is NOT
        merged into the orchestrator's context before guard evaluation. This
        prevents an LLM from injecting values that control guard conditions.
        After the guard passes, the tool context is merged so that phase
        handlers can read it.
        """

        def handler(**kwargs: Any) -> Dict[str, Any]:
            ctx_update = self._parse_tool_context(kwargs.get("context"))

            # Evaluate guard BEFORE merging tool-provided context
            if trans.guard is not None and not self._eval_guard(trans.guard):
                return {
                    "error": f"Guard for transition '{trans.name}' did not pass",
                    "phase": self._current_phase,
                    "_state": self._current_phase,
                }

            # Guard passed — now merge tool context for phase handler use
            if ctx_update:
                self._context.update(ctx_update)

            self._current_phase = trans.to_phase
            self._phase_history.append(trans.to_phase)
            self._execute_phase(trans.to_phase)

            state = self._make_state()
            return {
                "phase": state.current_phase,
                "context": state.context,
                "agents": [a.name for a in self.get_agents_for_phase(state.current_phase)],
                "_state": state.current_phase,
            }

        return handler

    def _make_advance_handler(self) -> Callable[..., Dict[str, Any]]:
        """Create a handler callable for the generic advance action.

        Tool-provided context is NOT merged before guard evaluation.
        See ``_make_transition_handler`` for rationale.
        """

        def handler(**kwargs: Any) -> Dict[str, Any]:
            ctx_update = self._parse_tool_context(kwargs.get("context"))

            # advance() evaluates guards against self._context — do NOT
            # merge tool context before this call.
            self.advance()

            # Merge tool context after guard evaluation
            if ctx_update:
                self._context.update(ctx_update)

            # Snapshot after merge so returned context includes tool updates
            state = self._make_state()
            return {
                "phase": state.current_phase,
                "context": state.context,
                "agents": [a.name for a in self.get_agents_for_phase(state.current_phase)],
                "_state": state.current_phase,
            }

        return handler

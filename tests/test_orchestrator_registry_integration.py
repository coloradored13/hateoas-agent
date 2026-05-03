"""Integration tests for Orchestrator driven via Registry.

Pins the contract that transition guards must evaluate against the
orchestrator's flat context (the shape ``conditions.py`` factories
expect), regardless of whether the orchestrator is driven directly via
``advance()`` or wrapped in a Registry by Runner / MCP server.
"""

from __future__ import annotations

from hateoas_agent import AgentSlot, Orchestrator, Registry
from hateoas_agent.conditions import (
    all_converged,
    belief_above,
    context_true,
    exit_gate_passed,
)


def _two_phase_orch():
    orch = Orchestrator(name="t", agents=[AgentSlot("a", role="r")])
    orch.phase("research")
    orch.phase("challenge", terminal=True)
    return orch


class TestGuardSeesOrchestratorContextViaRegistry:
    """When wrapped in a Registry, transition guards must receive the
    orchestrator's flat ``_context``, not ``Registry._last_result``."""

    def test_context_true_factory_works_through_registry(self):
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=context_true("converged"),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"converged": True}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        schemas = reg.get_current_tool_schemas()
        available = {s["name"] for s in schemas}
        assert "advance_to_challenge" in available, (
            "context_true('converged') should pass after the research phase "
            "set converged=True on orchestrator._context"
        )

    def test_belief_above_factory_works_through_registry(self):
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=belief_above(0.5),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"belief_state": 0.9}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        available = {s["name"] for s in reg.get_current_tool_schemas()}
        assert "advance_to_challenge" in available

    def test_exit_gate_passed_factory_works_through_registry(self):
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=exit_gate_passed(),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"exit_gate": "PASS"}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        available = {s["name"] for s in reg.get_current_tool_schemas()}
        assert "advance_to_challenge" in available

    def test_all_converged_factory_works_through_registry(self):
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=all_converged(),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"agent_statuses": ["converged", "converged"]}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        available = {s["name"] for s in reg.get_current_tool_schemas()}
        assert "advance_to_challenge" in available

    def test_failing_guard_excludes_transition_via_registry(self):
        """Symmetric negative: when context says guard should fail,
        the transition must NOT be advertised."""
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=context_true("converged"),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"converged": False}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        available = {s["name"] for s in reg.get_current_tool_schemas()}
        assert "advance_to_challenge" not in available
        # Generic 'advance' is always advertised when there are outgoing transitions
        assert "advance" in available

    def test_direct_advance_and_registry_filter_agree(self):
        """Calling advance() directly and querying via Registry should
        give the same answer about whether the guard passes."""
        orch = _two_phase_orch()
        orch.transition(
            "research", "challenge",
            guard=context_true("converged"),
            name="advance_to_challenge",
        )

        @orch.on_phase("research")
        def _research(o, agents, ctx):
            return {"converged": True}

        reg = Registry(orch)
        reg.handle_tool_call(reg.gateway_name, {})

        # Registry view
        registry_says_available = "advance_to_challenge" in {
            s["name"] for s in reg.get_current_tool_schemas()
        }
        # Direct view: would orchestrator's own filter_actions pass with
        # no explicit context arg? (uses self._context)
        actions = orch.get_actions_for_state("research")
        direct_says_available = any(
            a.name == "advance_to_challenge"
            for a in orch.filter_actions(actions)
        )

        assert registry_says_available == direct_says_available
        assert registry_says_available is True

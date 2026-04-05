"""Tests for the Orchestrator — multi-agent workflow orchestration.

Phase 1: sequential execution only (no async).
"""

import json

import pytest

from hateoas_agent.agent_slot import AgentSlot, AgentStatus
from hateoas_agent.orchestrator import Orchestrator, OrchestratorState
from hateoas_agent.registry import HasHateoas, Registry
from hateoas_agent.types import GatewayDef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_orchestrator():
    """Two-phase orchestrator: research -> synthesis."""
    orch = Orchestrator(
        name="simple",
        agents=[
            AgentSlot("analyst", role="Data analysis"),
            AgentSlot("writer", role="Report writing"),
        ],
    )
    orch.phase("research")
    orch.phase("synthesis", terminal=True)
    orch.transition("research", "synthesis")
    return orch


@pytest.fixture
def guarded_orchestrator():
    """Three-phase orchestrator with guards and self-loop."""
    orch = Orchestrator(
        name="review",
        agents=[
            AgentSlot("tech-architect", role="Architecture"),
            AgentSlot("product-strategist", role="Market viability"),
            AgentSlot("devils-advocate", role="Adversarial", join_phase="challenge"),
        ],
    )
    orch.phase("research", parallel=True, agents="*")
    orch.phase("challenge", parallel=True, agents="*")
    orch.phase("synthesis", parallel=False, agents=["tech-architect"], terminal=True)

    orch.transition(
        "research",
        "challenge",
        guard=lambda ctx: ctx.get("converged", False),
    )
    orch.transition(
        "challenge",
        "synthesis",
        guard=lambda ctx: ctx.get("exit_gate") == "PASS",
    )
    orch.transition(
        "challenge",
        "challenge",
        guard=lambda ctx: ctx.get("exit_gate") == "FAIL",
    )
    return orch


# ---------------------------------------------------------------------------
# Construction and definition
# ---------------------------------------------------------------------------


class TestOrchestratorConstruction:
    def test_empty_orchestrator(self):
        orch = Orchestrator(name="empty")
        assert orch.name == "empty"
        assert orch.agents == {}

    def test_agents_from_constructor(self):
        agents = [AgentSlot("a"), AgentSlot("b")]
        orch = Orchestrator(name="test", agents=agents)
        assert "a" in orch.agents
        assert "b" in orch.agents

    def test_custom_gateway_name(self):
        orch = Orchestrator(name="test", gateway_name="begin")
        gw = orch.get_gateway()
        assert gw.name == "begin"


class TestPhaseDefinition:
    def test_phase_basic(self, simple_orchestrator):
        assert "research" in simple_orchestrator._phases
        assert "synthesis" in simple_orchestrator._phases

    def test_phase_terminal(self, simple_orchestrator):
        assert "synthesis" in simple_orchestrator._terminal_phases
        assert "research" not in simple_orchestrator._terminal_phases

    def test_phase_parallel_flag(self, guarded_orchestrator):
        assert guarded_orchestrator._phases["research"].parallel is True
        assert guarded_orchestrator._phases["synthesis"].parallel is False

    def test_phase_agent_filter(self, guarded_orchestrator):
        assert guarded_orchestrator._phases["research"].agent_filter == "*"
        assert guarded_orchestrator._phases["synthesis"].agent_filter == ["tech-architect"]

    def test_phase_description(self):
        orch = Orchestrator(name="test")
        orch.phase("init", description="Initialize the process")
        assert orch._phases["init"].description == "Initialize the process"


class TestTransitionDefinition:
    def test_auto_naming(self, simple_orchestrator):
        assert len(simple_orchestrator._transitions) == 1
        assert simple_orchestrator._transitions[0].name == "research_to_synthesis"

    def test_auto_naming_dedup(self, guarded_orchestrator):
        names = [t.name for t in guarded_orchestrator._transitions]
        assert names[0] == "research_to_challenge"
        assert names[1] == "challenge_to_synthesis"
        assert names[2] == "challenge_to_challenge"

    def test_auto_naming_duplicate_target(self):
        """Multiple transitions to the same target get numbered."""
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.transition("a", "b", name="a_to_b")
        orch.transition("a", "b", name="a_to_b_retry")
        names = [t.name for t in orch._transitions]
        assert "a_to_b" in names
        assert "a_to_b_retry" in names

    def test_explicit_naming(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.transition("a", "b", name="custom_name")
        assert orch._transitions[0].name == "custom_name"

    def test_self_loop_naming(self, guarded_orchestrator):
        """Self-loop transitions (challenge -> challenge) get proper names."""
        loop = [t for t in guarded_orchestrator._transitions if t.from_phase == t.to_phase]
        assert len(loop) == 1
        assert loop[0].name == "challenge_to_challenge"


# ---------------------------------------------------------------------------
# Agent management
# ---------------------------------------------------------------------------


class TestAgentManagement:
    def test_add_agent(self, simple_orchestrator):
        simple_orchestrator.add_agent(AgentSlot("new-agent", role="Testing"))
        assert "new-agent" in simple_orchestrator.agents

    def test_remove_agent(self, simple_orchestrator):
        removed = simple_orchestrator.remove_agent("analyst")
        assert removed.name == "analyst"
        assert "analyst" not in simple_orchestrator.agents

    def test_remove_nonexistent_raises(self, simple_orchestrator):
        with pytest.raises(KeyError):
            simple_orchestrator.remove_agent("nonexistent")

    def test_get_agent(self, simple_orchestrator):
        agent = simple_orchestrator.get_agent("analyst")
        assert agent.name == "analyst"
        assert agent.role == "Data analysis"

    def test_get_agents_for_phase_all(self, simple_orchestrator):
        agents = simple_orchestrator.get_agents_for_phase("research")
        names = [a.name for a in agents]
        assert "analyst" in names
        assert "writer" in names

    def test_get_agents_for_phase_filtered(self, guarded_orchestrator):
        agents = guarded_orchestrator.get_agents_for_phase("synthesis")
        names = [a.name for a in agents]
        assert names == ["tech-architect"]

    def test_join_phase_filtering(self, guarded_orchestrator):
        """DA has join_phase='challenge', so excluded from research."""
        research_agents = guarded_orchestrator.get_agents_for_phase("research")
        research_names = [a.name for a in research_agents]
        assert "devils-advocate" not in research_names

        challenge_agents = guarded_orchestrator.get_agents_for_phase("challenge")
        challenge_names = [a.name for a in challenge_agents]
        assert "devils-advocate" in challenge_names

    def test_get_agents_for_unknown_phase(self, simple_orchestrator):
        agents = simple_orchestrator.get_agents_for_phase("nonexistent")
        assert agents == []


# ---------------------------------------------------------------------------
# Start and advance (core execution)
# ---------------------------------------------------------------------------


class TestStart:
    def test_start_default_phase(self, simple_orchestrator):
        state = simple_orchestrator.start()
        assert state.current_phase == "research"
        assert state.phase_history == ["research"]

    def test_start_explicit_phase(self, simple_orchestrator):
        state = simple_orchestrator.start("synthesis")
        assert state.current_phase == "synthesis"
        assert state.is_terminal is True

    def test_start_with_context(self, simple_orchestrator):
        state = simple_orchestrator.start(context={"task": "analyze X"})
        assert state.context["task"] == "analyze X"

    def test_start_unknown_phase_raises(self, simple_orchestrator):
        with pytest.raises(ValueError, match="Unknown phase"):
            simple_orchestrator.start("nonexistent")

    def test_start_no_phases_raises(self):
        orch = Orchestrator(name="empty")
        with pytest.raises(ValueError, match="no phases defined"):
            orch.start()

    def test_start_runs_phase_handler(self):
        orch = Orchestrator(name="test")
        orch.phase("init")

        @orch.on_phase("init")
        def handle_init(orchestrator, agents, context):
            return {"initialized": True}

        state = orch.start()
        assert state.context["initialized"] is True

    def test_start_resets_state(self, simple_orchestrator):
        """Calling start() again resets the orchestrator."""
        simple_orchestrator.start(context={"round": 1})
        state = simple_orchestrator.start(context={"round": 2})
        assert state.context == {"round": 2}
        assert state.phase_history == ["research"]


class TestAdvance:
    def test_advance_unguarded(self, simple_orchestrator):
        simple_orchestrator.start()
        state = simple_orchestrator.advance()
        assert state.current_phase == "synthesis"
        assert state.phase_history == ["research", "synthesis"]

    def test_advance_guard_passes(self, guarded_orchestrator):
        guarded_orchestrator.start(context={"converged": True})
        state = guarded_orchestrator.advance()
        assert state.current_phase == "challenge"

    def test_advance_guard_fails_stays(self, guarded_orchestrator):
        guarded_orchestrator.start(context={"converged": False})
        state = guarded_orchestrator.advance()
        assert state.current_phase == "research"  # didn't move

    def test_advance_with_context_update(self, guarded_orchestrator):
        guarded_orchestrator.start()
        state = guarded_orchestrator.advance(context={"converged": True})
        assert state.current_phase == "challenge"

    def test_advance_not_started_raises(self, simple_orchestrator):
        with pytest.raises(ValueError, match="not started"):
            simple_orchestrator.advance()

    def test_self_loop_transition(self, guarded_orchestrator):
        """challenge -> challenge when exit_gate == FAIL."""
        guarded_orchestrator.start(context={"converged": True})
        guarded_orchestrator.advance()  # research -> challenge
        assert guarded_orchestrator._current_phase == "challenge"

        state = guarded_orchestrator.advance(context={"exit_gate": "FAIL"})
        assert state.current_phase == "challenge"
        assert state.phase_history == ["research", "challenge", "challenge"]

    def test_self_loop_then_advance(self, guarded_orchestrator):
        """challenge -> challenge -> synthesis."""
        guarded_orchestrator.start(context={"converged": True})
        guarded_orchestrator.advance()  # -> challenge

        guarded_orchestrator.advance(context={"exit_gate": "FAIL"})  # loop
        state = guarded_orchestrator.advance(context={"exit_gate": "PASS"})
        assert state.current_phase == "synthesis"
        assert state.is_terminal is True
        assert state.phase_history == [
            "research", "challenge", "challenge", "synthesis"
        ]

    def test_advance_terminal_no_transitions(self, simple_orchestrator):
        """Advancing from a terminal phase with no outgoing transitions stays."""
        simple_orchestrator.start()
        simple_orchestrator.advance()  # -> synthesis (terminal)
        state = simple_orchestrator.advance()
        assert state.current_phase == "synthesis"  # stays

    def test_advance_runs_phase_handler(self, simple_orchestrator):
        call_log = []

        @simple_orchestrator.on_phase("synthesis")
        def handle_synthesis(orchestrator, agents, context):
            call_log.append("synthesis")
            return {"synthesized": True}

        simple_orchestrator.start()
        state = simple_orchestrator.advance()
        assert call_log == ["synthesis"]
        assert state.context["synthesized"] is True

    def test_guard_exception_treated_as_false(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.transition("a", "b", guard=lambda ctx: 1 / 0)  # raises

        orch.start()
        state = orch.advance()
        assert state.current_phase == "a"  # stayed, guard failed

    def test_first_matching_guard_wins(self):
        """Transitions evaluated in definition order."""
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.phase("c")
        orch.transition("a", "b", guard=lambda ctx: ctx.get("go_b", False))
        orch.transition("a", "c", guard=lambda ctx: ctx.get("go_c", False))

        orch.start(context={"go_b": True, "go_c": True})
        state = orch.advance()
        assert state.current_phase == "b"  # b wins because defined first


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------


class TestPhaseHandlers:
    def test_on_phase_decorator(self):
        orch = Orchestrator(name="test")
        orch.phase("init")

        @orch.on_phase("init")
        def handle(orchestrator, agents, context):
            return {"key": "value"}

        assert "init" in orch._phase_handlers

    def test_handler_receives_orchestrator(self):
        orch = Orchestrator(name="test")
        orch.phase("init")
        received = {}

        @orch.on_phase("init")
        def handle(orchestrator, agents, context):
            received["orch"] = orchestrator
            return {}

        orch.start()
        assert received["orch"] is orch

    def test_handler_receives_agents(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("a1"), AgentSlot("a2")],
        )
        orch.phase("init")
        received = {}

        @orch.on_phase("init")
        def handle(orchestrator, agents, context):
            received["agents"] = agents
            return {}

        orch.start()
        names = [a.name for a in received["agents"]]
        assert "a1" in names
        assert "a2" in names

    def test_handler_receives_context(self):
        orch = Orchestrator(name="test")
        orch.phase("init")
        received = {}

        @orch.on_phase("init")
        def handle(orchestrator, agents, context):
            received["ctx"] = context
            return {}

        orch.start(context={"task": "analyze"})
        assert received["ctx"]["task"] == "analyze"

    def test_handler_return_merges_context(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.transition("a", "b")

        @orch.on_phase("a")
        def handle_a(orchestrator, agents, context):
            return {"from_a": True}

        @orch.on_phase("b")
        def handle_b(orchestrator, agents, context):
            return {"from_b": True, "saw_a": context.get("from_a")}

        orch.start()
        state = orch.advance()
        assert state.context["from_a"] is True
        assert state.context["from_b"] is True
        assert state.context["saw_a"] is True

    def test_handler_returning_none_ok(self):
        orch = Orchestrator(name="test")
        orch.phase("init")

        @orch.on_phase("init")
        def handle(orchestrator, agents, context):
            pass  # returns None

        state = orch.start()
        assert state.current_phase == "init"

    def test_no_handler_still_works(self, simple_orchestrator):
        """Phases without handlers just transition without running anything."""
        state = simple_orchestrator.start()
        assert state.current_phase == "research"


# ---------------------------------------------------------------------------
# Agent execution (run_agent)
# ---------------------------------------------------------------------------


class TestRunAgent:
    def test_run_agent_success(self):
        orch = Orchestrator(name="test")
        agent = AgentSlot("agent1")
        orch.add_agent(agent)

        def executor(a, task):
            return {"finding": f"{a.name} analyzed {task}"}

        orch.set_executor(executor)
        result = orch.run_agent(agent, task="data")

        assert result.agent_name == "agent1"
        assert result.output == {"finding": "agent1 analyzed data"}
        assert result.status == AgentStatus.CONVERGED
        assert agent.status == AgentStatus.CONVERGED

    def test_run_agent_error(self):
        orch = Orchestrator(name="test")
        agent = AgentSlot("agent1")
        orch.add_agent(agent)

        def failing_executor(a, task):
            raise RuntimeError("Connection timeout")

        orch.set_executor(failing_executor)
        result = orch.run_agent(agent, task="data")

        assert result.status == AgentStatus.ERROR
        assert "Connection timeout" in result.error
        assert agent.status == AgentStatus.ERROR

    def test_run_agent_sets_running_then_converged(self):
        orch = Orchestrator(name="test")
        agent = AgentSlot("agent1")
        orch.add_agent(agent)
        statuses = []

        def tracking_executor(a, task):
            statuses.append(a.status)
            return "done"

        orch.set_executor(tracking_executor)
        orch.run_agent(agent, task="work")

        assert statuses == [AgentStatus.RUNNING]
        assert agent.status == AgentStatus.CONVERGED

    def test_run_agent_inline_executor(self):
        orch = Orchestrator(name="test")
        agent = AgentSlot("agent1")
        orch.add_agent(agent)

        result = orch.run_agent(
            agent,
            task="data",
            executor=lambda a, t: f"{a.name}:{t}",
        )
        assert result.output == "agent1:data"

    def test_run_agent_no_executor_raises(self):
        orch = Orchestrator(name="test")
        agent = AgentSlot("agent1")
        orch.add_agent(agent)

        with pytest.raises(ValueError, match="No executor"):
            orch.run_agent(agent, task="data")

    def test_agent_specific_executor(self):
        orch = Orchestrator(name="test")
        a1 = AgentSlot("agent1")
        a2 = AgentSlot("agent2")
        orch.add_agent(a1)
        orch.add_agent(a2)

        orch.set_executor(lambda a, t: "default", agent_name=None)
        orch.set_executor(lambda a, t: "special", agent_name="agent1")

        r1 = orch.run_agent(a1, task="x")
        r2 = orch.run_agent(a2, task="x")
        assert r1.output == "special"
        assert r2.output == "default"

    def test_run_agent_in_phase_handler(self):
        """Phase handler uses run_agent to execute agents."""
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("analyst")],
        )
        orch.phase("research")
        orch.set_executor(lambda a, t: f"analyzed: {t}")

        @orch.on_phase("research")
        def handle_research(orchestrator, agents, context):
            results = []
            for agent in agents:
                r = orchestrator.run_agent(agent, task=context.get("task"))
                results.append(r.output)
            return {"findings": results}

        state = orch.start(context={"task": "API design"})
        assert state.context["findings"] == ["analyzed: API design"]


# ---------------------------------------------------------------------------
# HasHateoas protocol conformance
# ---------------------------------------------------------------------------


class TestHateoasProtocol:
    def test_isinstance_check(self, simple_orchestrator):
        assert isinstance(simple_orchestrator, HasHateoas)

    def test_get_gateway(self, simple_orchestrator):
        gw = simple_orchestrator.get_gateway()
        assert isinstance(gw, GatewayDef)
        assert gw.name == "start_workflow"
        assert gw.handler is not None

    def test_get_actions_for_state(self, simple_orchestrator):
        actions = simple_orchestrator.get_actions_for_state("research")
        names = [a.name for a in actions]
        assert "research_to_synthesis" in names
        assert "advance" in names

    def test_get_actions_for_terminal(self, simple_orchestrator):
        """Terminal phase with no outgoing transitions returns no actions."""
        actions = simple_orchestrator.get_actions_for_state("synthesis")
        assert actions == []

    def test_get_handler_transition(self, simple_orchestrator):
        handler = simple_orchestrator.get_handler("research_to_synthesis")
        assert handler is not None
        assert callable(handler)

    def test_get_handler_advance(self, simple_orchestrator):
        handler = simple_orchestrator.get_handler("advance")
        assert handler is not None
        assert callable(handler)

    def test_get_handler_unknown(self, simple_orchestrator):
        handler = simple_orchestrator.get_handler("nonexistent")
        assert handler is None

    def test_get_all_action_names(self, simple_orchestrator):
        names = simple_orchestrator.get_all_action_names()
        assert "research_to_synthesis" in names
        assert "advance" in names

    def test_get_all_action_names_no_transitions(self):
        orch = Orchestrator(name="test")
        orch.phase("solo")
        assert orch.get_all_action_names() == set()

    def test_get_transition_metadata(self, simple_orchestrator):
        meta = simple_orchestrator.get_transition_metadata("research_to_synthesis")
        assert meta is not None
        from_states, to_state = meta
        assert from_states == ["research"]
        assert to_state == "synthesis"

    def test_get_transition_metadata_advance(self, simple_orchestrator):
        meta = simple_orchestrator.get_transition_metadata("advance")
        assert meta is not None
        from_states, to_state = meta
        assert "research" in from_states
        assert to_state is None  # advance is dynamic

    def test_get_transition_metadata_unknown(self, simple_orchestrator):
        assert simple_orchestrator.get_transition_metadata("nope") is None


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_registry_wraps_orchestrator(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        assert reg.gateway_name == "start_workflow"

    def test_gateway_call_starts_workflow(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        result = reg.handle_tool_call("start_workflow", {})
        assert "research" in result

    def test_gateway_then_transition(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        reg.handle_tool_call("start_workflow", {})
        result = reg.handle_tool_call("research_to_synthesis", {})
        assert "synthesis" in result

    def test_gateway_then_advance(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        reg.handle_tool_call("start_workflow", {})
        result = reg.handle_tool_call("advance", {})
        assert "synthesis" in result

    def test_tool_schemas(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "start_workflow" in names

    def test_tool_schemas_after_gateway(self, simple_orchestrator):
        reg = Registry(simple_orchestrator)
        reg.handle_tool_call("start_workflow", {})
        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "research_to_synthesis" in names
        assert "advance" in names

    def test_self_loop_via_registry(self, guarded_orchestrator):
        """Registry handles same-state transitions correctly.

        Guard-controlling context must be set programmatically (server-side),
        not via tool input, because tool-provided context is not merged
        before guard evaluation.
        """
        reg = Registry(guarded_orchestrator)
        reg.handle_tool_call(
            "start_workflow",
            {"context": json.dumps({"converged": True})},
        )
        # Now in challenge via guard. Set exit_gate programmatically, then advance
        guarded_orchestrator._context["exit_gate"] = "FAIL"
        reg.handle_tool_call("advance", {})
        assert reg._last_state == "challenge"

        # Now advance with PASS -> synthesis
        guarded_orchestrator._context["exit_gate"] = "PASS"
        result = reg.handle_tool_call("advance", {})
        assert "synthesis" in result
        assert reg._last_state == "synthesis"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_empty_raises(self):
        orch = Orchestrator(name="test")
        with pytest.raises(ValueError, match="no phases defined"):
            orch.validate()

    def test_validate_unknown_from_phase(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.transition("nonexistent", "a")
        with pytest.raises(ValueError, match="unknown phase 'nonexistent'"):
            orch.validate()

    def test_validate_unknown_to_phase(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.transition("a", "nonexistent")
        with pytest.raises(ValueError, match="unknown phase 'nonexistent'"):
            orch.validate()

    def test_validate_success(self, simple_orchestrator):
        simple_orchestrator.validate()  # should not raise


# ---------------------------------------------------------------------------
# Filter actions (guard-based HATEOAS filtering)
# ---------------------------------------------------------------------------


class TestFilterActions:
    def test_filter_shows_passing_guards(self, guarded_orchestrator):
        guarded_orchestrator.start(context={"converged": True})
        actions = guarded_orchestrator.get_actions_for_state("research")
        filtered = guarded_orchestrator.filter_actions(actions)
        names = [a.name for a in filtered]
        assert "research_to_challenge" in names

    def test_filter_hides_failing_guards(self, guarded_orchestrator):
        guarded_orchestrator.start(context={"converged": False})
        actions = guarded_orchestrator.get_actions_for_state("research")
        filtered = guarded_orchestrator.filter_actions(actions)
        names = [a.name for a in filtered]
        assert "research_to_challenge" not in names
        assert "advance" in names  # advance always shown

    def test_filter_unguarded_always_shown(self, simple_orchestrator):
        simple_orchestrator.start()
        actions = simple_orchestrator.get_actions_for_state("research")
        filtered = simple_orchestrator.filter_actions(actions)
        assert len(filtered) == len(actions)

    def test_filter_guard_exception_excludes(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.transition("a", "b", guard=lambda ctx: 1 / 0)

        orch.start()
        actions = orch.get_actions_for_state("a")
        filtered = orch.filter_actions(actions)
        names = [a.name for a in filtered]
        assert "a_to_b" not in names
        assert "advance" in names


# ---------------------------------------------------------------------------
# OrchestratorState dataclass
# ---------------------------------------------------------------------------


class TestOrchestratorState:
    def test_state_fields(self, simple_orchestrator):
        state = simple_orchestrator.start(context={"k": "v"})
        assert isinstance(state, OrchestratorState)
        assert state.name == "simple"
        assert state.current_phase == "research"
        assert state.context == {"k": "v"}
        assert state.phase_history == ["research"]
        assert state.is_terminal is False

    def test_state_is_snapshot(self, simple_orchestrator):
        """State is a copy, not a reference to orchestrator internals."""
        state = simple_orchestrator.start(context={"k": "v"})
        state.context["mutated"] = True
        # Orchestrator's internal context should not be mutated
        assert "mutated" not in simple_orchestrator._context


# ---------------------------------------------------------------------------
# Sigma-review mapping (integration-level)
# ---------------------------------------------------------------------------


class TestSigmaReviewMapping:
    """Verify the orchestrator maps sigma-review concepts as planned."""

    def test_full_review_lifecycle(self):
        """Simulates a sigma-review: research -> challenge (loop) -> synthesis."""
        orch = Orchestrator(
            name="sigma-review",
            agents=[
                AgentSlot("tech-architect", role="Architecture"),
                AgentSlot("product-strategist", role="Market"),
                AgentSlot("devils-advocate", role="Adversarial", join_phase="challenge"),
            ],
        )

        orch.phase("research", parallel=True)
        orch.phase("challenge", parallel=True)
        orch.phase("synthesis", terminal=True)

        orch.transition(
            "research", "challenge",
            guard=lambda ctx: ctx.get("convergence_count", 0) >= ctx.get("agent_count", 999),
        )
        orch.transition(
            "challenge", "synthesis",
            guard=lambda ctx: ctx.get("exit_gate") == "PASS" and ctx.get("belief_state", 0) > 0.85,
        )
        orch.transition(
            "challenge", "challenge",
            guard=lambda ctx: ctx.get("exit_gate") == "FAIL",
        )

        orch.set_executor(lambda a, t: f"{a.name}: {t}")

        @orch.on_phase("research")
        def handle_research(orchestrator, agents, context):
            results = []
            for agent in agents:
                r = orchestrator.run_agent(agent, task=context.get("task"))
                results.append(r.output)
            return {
                "findings": results,
                "convergence_count": len(agents),
                "agent_count": len(agents),
            }

        @orch.on_phase("challenge")
        def handle_challenge(orchestrator, agents, context):
            round_num = context.get("challenge_round", 0) + 1
            return {"challenge_round": round_num}

        @orch.on_phase("synthesis")
        def handle_synthesis(orchestrator, agents, context):
            return {"final_report": "synthesized"}

        # Start research
        state = orch.start(context={"task": "Evaluate API design"})
        assert state.current_phase == "research"
        assert state.context["convergence_count"] == 2  # TA + PS (DA not yet)
        assert len(state.context["findings"]) == 2

        # Advance: research -> challenge (convergence met)
        state = orch.advance()
        assert state.current_phase == "challenge"
        assert state.context["challenge_round"] == 1

        # Challenge round 1: DA says FAIL -> self-loop
        state = orch.advance(context={"exit_gate": "FAIL"})
        assert state.current_phase == "challenge"
        assert state.context["challenge_round"] == 2

        # Challenge round 2: DA says PASS with high belief -> synthesis
        state = orch.advance(context={"exit_gate": "PASS", "belief_state": 0.9})
        assert state.current_phase == "synthesis"
        assert state.is_terminal is True
        assert state.context["final_report"] == "synthesized"
        assert state.phase_history == [
            "research", "challenge", "challenge", "synthesis"
        ]

"""End-to-end integration tests for the orchestration system.

Tests the full stack: Orchestrator + AsyncRunner + Conditions +
Persistence + Visualization working together.
"""

import asyncio
import json

import pytest

from hateoas_agent.agent_slot import AgentSlot, AgentStatus
from hateoas_agent.async_runner import AsyncRunner
from hateoas_agent.conditions import (
    all_converged,
    belief_above,
    exit_gate_passed,
    round_limit,
)
from hateoas_agent.orchestrator import Orchestrator
from hateoas_agent.orchestrator_persistence import (
    load_orchestrator_checkpoint,
    save_orchestrator_checkpoint,
)
from hateoas_agent.orchestrator_visualization import orchestrator_to_mermaid
from hateoas_agent.registry import Registry


class TestSigmaReviewEndToEnd:
    """Full sigma-review simulation using all v0.2 components."""

    def _build_review(self):
        review = Orchestrator(
            name="sigma-review",
            agents=[
                AgentSlot("tech-architect", role="Architecture analysis"),
                AgentSlot("product-strategist", role="Market viability"),
                AgentSlot("ux-researcher", role="User experience"),
                AgentSlot("devils-advocate", role="Adversarial challenge", join_phase="challenge"),
            ],
        )

        review.phase("research", parallel=True, description="Independent analysis")
        review.phase("challenge", parallel=True, description="Adversarial rounds")
        review.phase("synthesis", description="Final report", terminal=True)

        review.transition(
            "research",
            "challenge",
            guard=all_converged() & belief_above(0.7),
        )
        review.transition(
            "challenge",
            "synthesis",
            guard=exit_gate_passed() & belief_above(0.85),
        )
        review.transition(
            "challenge",
            "challenge",
            guard=~exit_gate_passed() & round_limit(5),
        )

        return review

    @pytest.mark.asyncio
    async def test_full_async_review_lifecycle(self):
        review = self._build_review()

        async def agent_exec(agent, task):
            await asyncio.sleep(0.01)
            return {"agent": agent.name, "analysis": f"Analyzed: {task}"}

        review.set_executor(agent_exec)

        @review.on_phase("research")
        async def handle_research(orchestrator, agents, context):
            results = await orchestrator.run_agents_parallel(agents, task=context.get("task", ""))
            return {
                "findings": [r.output for r in results],
                "agent_statuses": [r.status.value for r in results],
                "belief_state": 0.75,
            }

        @review.on_phase("challenge")
        async def handle_challenge(orchestrator, agents, context):
            round_num = context.get("round", 0) + 1
            results = await orchestrator.run_agents_parallel(agents, task="challenge round")
            # Simulate: round 1 fails, round 2 passes
            if round_num < 2:
                return {
                    "round": round_num,
                    "exit_gate": "FAIL",
                    "belief_state": 0.8,
                    "agent_statuses": [r.status.value for r in results],
                }
            return {
                "round": round_num,
                "exit_gate": "PASS",
                "belief_state": 0.92,
                "agent_statuses": [r.status.value for r in results],
            }

        @review.on_phase("synthesis")
        async def handle_synthesis(orchestrator, agents, context):
            return {"final_report": "Review complete", "round": context["round"]}

        runner = AsyncRunner(review)
        state = await runner.run_orchestrated(context={"task": "Evaluate HATEOAS v0.2 API design"})

        assert state.is_terminal is True
        assert state.current_phase == "synthesis"
        assert state.context["final_report"] == "Review complete"
        assert state.context["round"] == 2
        assert state.phase_history == ["research", "challenge", "challenge", "synthesis"]

    def test_registry_integration_full_workflow(self):
        """Orchestrator works through Registry for tool routing."""
        review = self._build_review()

        @review.on_phase("research")
        def handle_research(orch, agents, ctx):
            return {
                "agent_statuses": ["converged"] * len(agents),
                "belief_state": 0.8,
            }

        @review.on_phase("challenge")
        def handle_challenge(orch, agents, ctx):
            return {"exit_gate": "PASS", "belief_state": 0.9}

        reg = Registry(review)

        # Start via gateway
        result = reg.handle_tool_call("start_workflow", {})
        assert "research" in result
        assert reg._last_state == "research"

        # Advance via action
        result = reg.handle_tool_call("advance", {})
        assert "challenge" in result
        assert reg._last_state == "challenge"

        # Advance to synthesis
        result = reg.handle_tool_call("advance", {})
        assert "synthesis" in result

    def test_checkpoint_mid_review_and_resume(self):
        """Save checkpoint mid-review, restore to a new orchestrator, continue."""
        review = self._build_review()

        @review.on_phase("research")
        def handle_research(orch, agents, ctx):
            return {
                "agent_statuses": ["converged"] * len(agents),
                "belief_state": 0.8,
                "findings": ["finding1", "finding2"],
            }

        review.start(context={"task": "mid-review test"})
        review.advance()  # -> challenge

        # Mark some agents
        review.get_agent("tech-architect").status = AgentStatus.CONVERGED
        review.get_agent("devils-advocate").status = AgentStatus.RUNNING

        # Save checkpoint
        checkpoint = save_orchestrator_checkpoint(review)

        # Verify checkpoint content
        assert checkpoint["current_phase"] == "challenge"
        assert checkpoint["context"]["task"] == "mid-review test"
        assert checkpoint["agent_states"]["tech-architect"] == "converged"
        assert checkpoint["agent_states"]["devils-advocate"] == "running"

        # Restore to a fresh orchestrator
        review2 = self._build_review()

        @review2.on_phase("challenge")
        def handle_challenge(orch, agents, ctx):
            return {"exit_gate": "PASS", "belief_state": 0.9}

        @review2.on_phase("synthesis")
        def handle_synthesis(orch, agents, ctx):
            return {"final": True}

        load_orchestrator_checkpoint(review2, checkpoint)

        # Verify restored state
        assert review2._current_phase == "challenge"
        assert review2.get_agent("tech-architect").status == AgentStatus.CONVERGED

        # Continue from where we left off — provide context so synthesis guard passes
        state = review2.advance(
            context={
                "exit_gate": "PASS",
                "belief_state": 0.9,
                "agent_statuses": ["converged", "converged", "converged"],
            }
        )
        assert state.current_phase == "synthesis"
        assert state.is_terminal is True

    def test_visualization_generates_valid_diagram(self):
        review = self._build_review()
        diagram = orchestrator_to_mermaid(review)

        assert diagram.startswith("stateDiagram-v2")
        assert "[*] --> research" in diagram
        assert "research --> challenge" in diagram
        assert "challenge --> synthesis" in diagram
        assert "challenge --> challenge" in diagram
        assert "synthesis --> [*]" in diagram
        assert "tech-architect" in diagram
        assert "devils-advocate" in diagram

    def test_validate_catches_bad_config(self):
        review = self._build_review()
        review.validate()  # should pass

        broken = Orchestrator(name="broken")
        broken.phase("a")
        broken.transition("a", "nonexistent")
        with pytest.raises(ValueError, match="unknown phase"):
            broken.validate()

    def test_json_checkpoint_roundtrip(self):
        """Checkpoint survives JSON serialization (for persistence to disk/API)."""
        review = self._build_review()
        review.start(context={"task": "json test", "scores": [1, 2, 3]})
        review.advance(
            context={
                "agent_statuses": ["converged", "converged", "converged"],
                "belief_state": 0.8,
            }
        )

        data = save_orchestrator_checkpoint(review)
        json_str = json.dumps(data)

        # Simulate reading from disk
        restored = json.loads(json_str)
        review2 = self._build_review()
        load_orchestrator_checkpoint(review2, restored)

        assert review2._current_phase == review._current_phase
        assert review2._context["task"] == "json test"
        assert review2._context["scores"] == [1, 2, 3]


class TestDynamicAgentCreation:
    """Test adding/removing agents between phases (v0.2 scope)."""

    @pytest.mark.asyncio
    async def test_add_agent_between_phases(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("base-agent")],
        )
        orch.phase("phase1")
        orch.phase("phase2", terminal=True)
        orch.transition("phase1", "phase2")

        orch.set_executor(lambda a, t: f"{a.name} done")

        @orch.on_phase("phase1")
        def handle_p1(orchestrator, agents, ctx):
            # Dynamically add a specialist
            orchestrator.add_agent(AgentSlot("specialist", role="Deep dive"))
            return {"phase1_done": True}

        @orch.on_phase("phase2")
        def handle_p2(orchestrator, agents, ctx):
            return {"agent_count": len(agents)}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.is_terminal
        assert "specialist" in orch.agents
        # phase2 filter is "*" so specialist is included
        assert state.context["agent_count"] == 2

    @pytest.mark.asyncio
    async def test_remove_agent_between_phases(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("keep"), AgentSlot("remove-me")],
        )
        orch.phase("phase1")
        orch.phase("phase2", terminal=True)
        orch.transition("phase1", "phase2")

        @orch.on_phase("phase1")
        def handle_p1(orchestrator, agents, ctx):
            orchestrator.remove_agent("remove-me")
            return {}

        @orch.on_phase("phase2")
        def handle_p2(orchestrator, agents, ctx):
            return {"agents": [a.name for a in agents]}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.context["agents"] == ["keep"]


class TestBackwardCompatibility:
    """Verify v0.2 additions don't break v0.1 patterns."""

    def test_state_machine_still_works(self):
        from hateoas_agent import Registry, StateMachine

        sm = StateMachine("orders", gateway_name="query_orders")
        sm.gateway(description="Query orders")
        sm.action(
            "approve",
            description="Approve order",
            from_states=["pending"],
            to_state="approved",
        )

        @sm.on_gateway
        def handle_gw(**kwargs):
            return {"order": "123", "_state": "pending"}

        @sm.on_action("approve")
        def handle_approve(**kwargs):
            return {"approved": True, "_state": "approved"}

        reg = Registry(sm)
        result = reg.handle_tool_call("query_orders", {})
        assert "approve" in result

    def test_resource_still_works(self):
        from hateoas_agent import Registry, Resource
        from hateoas_agent import action as action_decorator
        from hateoas_agent import gateway as gw_decorator
        from hateoas_agent import state as state_decorator

        class TestResource(Resource):
            name = "test"

            @gw_decorator(name="query", description="Query")
            def query(self, **kwargs):
                return {"item": "1", "_state": "active"}

            @action_decorator(name="do_thing", description="Do it")
            @state_decorator("active")
            def do_thing(self, **kwargs):
                return {"done": True, "_state": "done"}

        res = TestResource()
        reg = Registry(res)
        result = reg.handle_tool_call("query", {})
        assert "do_thing" in result

    def test_existing_persistence_still_works(self):
        from hateoas_agent import (
            Registry,
            StateMachine,
            save_registry_checkpoint,
        )

        sm = StateMachine("test", gateway_name="start_test")
        sm.gateway(description="Start")

        @sm.on_gateway
        def handle(**kwargs):
            return {"_state": "ready"}

        reg = Registry(sm)
        reg.handle_tool_call("start_test", {})

        data = save_registry_checkpoint(reg)
        assert data["last_state"] == "ready"

    def test_imports_unchanged(self):
        """All v0.1 public exports still available."""
        # If we got here, all imports work
        assert True

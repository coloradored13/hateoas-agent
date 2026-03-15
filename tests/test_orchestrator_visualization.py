"""Tests for orchestrator Mermaid diagram generation."""

from hateoas_agent.agent_slot import AgentSlot
from hateoas_agent.conditions import belief_above, exit_gate_passed
from hateoas_agent.orchestrator import Orchestrator
from hateoas_agent.orchestrator_visualization import orchestrator_to_mermaid


class TestOrchestratorToMermaid:
    def test_empty_orchestrator(self):
        orch = Orchestrator(name="empty")
        result = orchestrator_to_mermaid(orch)
        assert result == "stateDiagram-v2"

    def test_single_phase(self):
        orch = Orchestrator(name="test")
        orch.phase("init")
        result = orchestrator_to_mermaid(orch)
        assert "stateDiagram-v2" in result
        assert "[*] --> init" in result

    def test_linear_workflow(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b")
        orch.phase("c", terminal=True)
        orch.transition("a", "b")
        orch.transition("b", "c")

        result = orchestrator_to_mermaid(orch)
        assert "[*] --> a" in result
        assert "a --> b" in result
        assert "b --> c" in result
        assert "c --> [*]" in result

    def test_self_loop(self):
        orch = Orchestrator(name="test")
        orch.phase("challenge")
        orch.phase("synthesis", terminal=True)
        orch.transition("challenge", "challenge", name="retry")
        orch.transition("challenge", "synthesis", name="complete")

        result = orchestrator_to_mermaid(orch)
        assert "challenge --> challenge : retry" in result
        assert "challenge --> synthesis : complete" in result

    def test_guard_description(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b", guard=belief_above(0.85))

        result = orchestrator_to_mermaid(orch)
        assert "belief_above(0.85)" in result

    def test_composed_guard_description(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b", guard=belief_above(0.85) & exit_gate_passed())

        result = orchestrator_to_mermaid(orch)
        assert "belief_above" in result
        assert "exit_gate_passed" in result

    def test_lambda_guard_uses_transition_name(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b", guard=lambda ctx: True)

        result = orchestrator_to_mermaid(orch)
        # Lambda has no description attribute, falls back to transition name
        assert "a --> b : a_to_b" in result

    def test_agent_annotations(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("analyst"), AgentSlot("writer")],
        )
        orch.phase("research")

        result = orchestrator_to_mermaid(orch)
        assert "note right of research" in result
        assert "analyst" in result
        assert "writer" in result

    def test_filtered_agents(self):
        orch = Orchestrator(
            name="test",
            agents=[
                AgentSlot("ta"),
                AgentSlot("da", join_phase="challenge"),
            ],
        )
        orch.phase("research")
        orch.phase("challenge")

        result = orchestrator_to_mermaid(orch)
        # Research should only show ta (da hasn't joined yet)
        lines = result.split("\n")
        research_note = [line for line in lines if "note right of research" in line]
        assert len(research_note) == 1
        assert "ta" in research_note[0]
        assert "da" not in research_note[0]

        # Challenge should show both
        challenge_note = [line for line in lines if "note right of challenge" in line]
        assert len(challenge_note) == 1
        assert "ta" in challenge_note[0]
        assert "da" in challenge_note[0]

    def test_full_sigma_review_diagram(self):
        orch = Orchestrator(
            name="sigma-review",
            agents=[
                AgentSlot("tech-architect"),
                AgentSlot("product-strategist"),
                AgentSlot("devils-advocate", join_phase="challenge"),
            ],
        )
        orch.phase("research", parallel=True)
        orch.phase("challenge", parallel=True)
        orch.phase("synthesis", terminal=True)

        orch.transition(
            "research", "challenge",
            guard=belief_above(0.7),
        )
        orch.transition(
            "challenge", "synthesis",
            guard=belief_above(0.85) & exit_gate_passed(),
        )
        orch.transition(
            "challenge", "challenge",
            name="retry",
        )

        result = orchestrator_to_mermaid(orch)

        assert "stateDiagram-v2" in result
        assert "[*] --> research" in result
        assert "research --> challenge" in result
        assert "challenge --> synthesis" in result
        assert "challenge --> challenge : retry" in result
        assert "synthesis --> [*]" in result
        assert "devils-advocate" in result

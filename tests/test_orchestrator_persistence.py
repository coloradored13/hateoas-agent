"""Tests for orchestrator checkpoint/restore."""

import json

from hateoas_agent.agent_slot import AgentSlot, AgentStatus
from hateoas_agent.orchestrator import Orchestrator
from hateoas_agent.orchestrator_persistence import (
    OrchestratorCheckpoint,
    load_orchestrator_checkpoint,
    save_orchestrator_checkpoint,
)

# ---------------------------------------------------------------------------
# OrchestratorCheckpoint dataclass
# ---------------------------------------------------------------------------


class TestOrchestratorCheckpoint:
    def test_default_values(self):
        cp = OrchestratorCheckpoint()
        assert cp.name == ""
        assert cp.current_phase is None
        assert cp.context == {}
        assert cp.phase_history == []
        assert cp.agent_states == {}
        assert isinstance(cp.timestamp, float)

    def test_to_dict(self):
        cp = OrchestratorCheckpoint(
            name="review",
            current_phase="challenge",
            context={"belief": 0.8},
            phase_history=["research", "challenge"],
            agent_states={"ta": "converged", "da": "running"},
            timestamp=1000.0,
        )
        d = cp.to_dict()
        assert d["name"] == "review"
        assert d["current_phase"] == "challenge"
        assert d["context"]["belief"] == 0.8
        assert d["phase_history"] == ["research", "challenge"]
        assert d["agent_states"]["ta"] == "converged"
        assert d["timestamp"] == 1000.0

    def test_from_dict(self):
        data = {
            "name": "review",
            "current_phase": "synthesis",
            "context": {"final": True},
            "phase_history": ["research", "challenge", "synthesis"],
            "agent_states": {"ta": "converged"},
            "timestamp": 2000.0,
        }
        cp = OrchestratorCheckpoint.from_dict(data)
        assert cp.name == "review"
        assert cp.current_phase == "synthesis"
        assert cp.context["final"] is True
        assert len(cp.phase_history) == 3
        assert cp.timestamp == 2000.0

    def test_from_dict_defaults(self):
        cp = OrchestratorCheckpoint.from_dict({})
        assert cp.name == ""
        assert cp.current_phase is None
        assert cp.context == {}

    def test_json_roundtrip(self):
        cp = OrchestratorCheckpoint(
            name="test",
            current_phase="a",
            context={"k": "v"},
            phase_history=["a"],
            agent_states={"x": "idle"},
            timestamp=500.0,
        )
        restored = OrchestratorCheckpoint.from_json(cp.to_json())
        assert restored.name == cp.name
        assert restored.current_phase == cp.current_phase
        assert restored.context == cp.context
        assert restored.phase_history == cp.phase_history
        assert restored.agent_states == cp.agent_states
        assert restored.timestamp == cp.timestamp

    def test_json_is_valid_json(self):
        cp = OrchestratorCheckpoint(name="test")
        parsed = json.loads(cp.to_json())
        assert parsed["name"] == "test"


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def _make_orchestrator(self):
        orch = Orchestrator(
            name="review",
            agents=[
                AgentSlot("ta", role="Architect"),
                AgentSlot("ps", role="Product"),
                AgentSlot("da", role="DA", join_phase="challenge"),
            ],
        )
        orch.phase("research")
        orch.phase("challenge")
        orch.phase("synthesis", terminal=True)
        orch.transition("research", "challenge")
        orch.transition("challenge", "synthesis")
        return orch

    def test_save_captures_state(self):
        orch = self._make_orchestrator()
        orch.start(context={"task": "evaluate"})
        orch.advance()  # -> challenge

        orch.get_agent("ta").status = AgentStatus.CONVERGED
        orch.get_agent("da").status = AgentStatus.RUNNING

        data = save_orchestrator_checkpoint(orch)

        assert data["name"] == "review"
        assert data["current_phase"] == "challenge"
        assert data["context"]["task"] == "evaluate"
        assert data["phase_history"] == ["research", "challenge"]
        assert data["agent_states"]["ta"] == "converged"
        assert data["agent_states"]["da"] == "running"

    def test_load_restores_state(self):
        orch = self._make_orchestrator()
        orch.start(context={"task": "evaluate"})
        orch.advance()  # -> challenge
        orch.get_agent("ta").status = AgentStatus.CONVERGED

        # Save
        data = save_orchestrator_checkpoint(orch)

        # Create a fresh orchestrator with same definitions
        orch2 = self._make_orchestrator()
        load_orchestrator_checkpoint(orch2, data)

        assert orch2._current_phase == "challenge"
        assert orch2._context["task"] == "evaluate"
        assert orch2._phase_history == ["research", "challenge"]
        assert orch2.get_agent("ta").status == AgentStatus.CONVERGED

    def test_load_then_advance(self):
        """After restoring, the orchestrator can continue advancing."""
        orch = self._make_orchestrator()
        orch.start(context={"task": "evaluate"})
        orch.advance()  # -> challenge
        data = save_orchestrator_checkpoint(orch)

        orch2 = self._make_orchestrator()
        load_orchestrator_checkpoint(orch2, data)
        state = orch2.advance()  # -> synthesis

        assert state.current_phase == "synthesis"
        assert state.is_terminal is True

    def test_save_before_start(self):
        orch = self._make_orchestrator()
        data = save_orchestrator_checkpoint(orch)
        assert data["current_phase"] is None
        assert data["phase_history"] == []

    def test_load_ignores_unknown_agents(self):
        """Agents in checkpoint but not in orchestrator are silently skipped."""
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("a1")],
        )
        orch.phase("init")

        data = {
            "name": "test",
            "current_phase": "init",
            "context": {},
            "phase_history": ["init"],
            "agent_states": {"a1": "converged", "unknown": "error"},
            "timestamp": 0.0,
        }
        load_orchestrator_checkpoint(orch, data)
        assert orch.get_agent("a1").status == AgentStatus.CONVERGED

    def test_json_roundtrip_full(self):
        orch = self._make_orchestrator()
        orch.start(context={"k": "v"})
        orch.advance()

        data = save_orchestrator_checkpoint(orch)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        orch2 = self._make_orchestrator()
        load_orchestrator_checkpoint(orch2, restored_data)
        assert orch2._current_phase == orch._current_phase
        assert orch2._context == orch._context

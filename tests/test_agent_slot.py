"""Tests for agent_slot module."""

import time

from hateoas_agent.agent_slot import AgentResult, AgentSlot, AgentStatus


class TestAgentStatus:
    def test_enum_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.CONVERGED.value == "converged"
        assert AgentStatus.BLOCKED.value == "blocked"
        assert AgentStatus.ERROR.value == "error"

    def test_all_statuses_distinct(self):
        values = [s.value for s in AgentStatus]
        assert len(values) == len(set(values))


class TestAgentSlot:
    def test_minimal_creation(self):
        slot = AgentSlot(name="tech-architect")
        assert slot.name == "tech-architect"
        assert slot.role == ""
        assert slot.status == AgentStatus.IDLE
        assert slot.join_phase is None
        assert slot.metadata == {}

    def test_full_creation(self):
        slot = AgentSlot(
            name="devils-advocate",
            role="Adversarial challenge",
            status=AgentStatus.IDLE,
            join_phase="challenge",
            metadata={"priority": "high"},
        )
        assert slot.name == "devils-advocate"
        assert slot.role == "Adversarial challenge"
        assert slot.join_phase == "challenge"
        assert slot.metadata == {"priority": "high"}

    def test_status_mutable(self):
        slot = AgentSlot(name="agent1")
        assert slot.status == AgentStatus.IDLE
        slot.status = AgentStatus.RUNNING
        assert slot.status == AgentStatus.RUNNING
        slot.status = AgentStatus.CONVERGED
        assert slot.status == AgentStatus.CONVERGED

    def test_metadata_mutable(self):
        slot = AgentSlot(name="agent1")
        slot.metadata["key"] = "value"
        assert slot.metadata["key"] == "value"

    def test_equality(self):
        a = AgentSlot(name="agent1", role="Role A")
        b = AgentSlot(name="agent1", role="Role A")
        assert a == b

    def test_inequality_different_name(self):
        a = AgentSlot(name="agent1")
        b = AgentSlot(name="agent2")
        assert a != b

    def test_metadata_isolation(self):
        """Each slot gets its own metadata dict."""
        a = AgentSlot(name="a")
        b = AgentSlot(name="b")
        a.metadata["x"] = 1
        assert "x" not in b.metadata


class TestAgentResult:
    def test_minimal_creation(self):
        result = AgentResult(agent_name="tech-architect")
        assert result.agent_name == "tech-architect"
        assert result.output is None
        assert result.status == AgentStatus.CONVERGED
        assert result.error is None
        assert isinstance(result.timestamp, float)

    def test_success_result(self):
        result = AgentResult(
            agent_name="agent1",
            output={"findings": ["item1", "item2"]},
            status=AgentStatus.CONVERGED,
        )
        assert result.output == {"findings": ["item1", "item2"]}
        assert result.status == AgentStatus.CONVERGED
        assert result.error is None

    def test_error_result(self):
        result = AgentResult(
            agent_name="agent1",
            status=AgentStatus.ERROR,
            error="Connection timeout",
        )
        assert result.output is None
        assert result.status == AgentStatus.ERROR
        assert result.error == "Connection timeout"

    def test_timestamp_auto_set(self):
        before = time.time()
        result = AgentResult(agent_name="agent1")
        after = time.time()
        assert before <= result.timestamp <= after

    def test_custom_timestamp(self):
        result = AgentResult(agent_name="agent1", timestamp=1000.0)
        assert result.timestamp == 1000.0

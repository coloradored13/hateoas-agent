"""Tests for persistence (checkpoint save/load)."""

import json

from hateoas_agent import Registry, StateMachine
from hateoas_agent.persistence import (
    RegistryCheckpoint,
    RunnerCheckpoint,
    load_registry_checkpoint,
    load_runner_checkpoint,
    save_registry_checkpoint,
    save_runner_checkpoint,
)
from hateoas_agent.types import DiscoveryReport, TransitionRecord


class TestRegistryCheckpoint:
    """Tests for RegistryCheckpoint serialization."""

    def test_to_dict_and_back(self):
        cp = RegistryCheckpoint(
            last_state="approved",
            last_result={"order": {"id": "1"}},
            transitions=[
                {"state_before": "pending", "action": "approve",
                 "state_after": "approved", "timestamp": 1.0},
            ],
            timestamp=100.0,
        )
        d = cp.to_dict()
        restored = RegistryCheckpoint.from_dict(d)

        assert restored.last_state == "approved"
        assert restored.last_result == {"order": {"id": "1"}}
        assert len(restored.transitions) == 1
        assert restored.timestamp == 100.0

    def test_to_json_and_back(self):
        cp = RegistryCheckpoint(last_state="pending", last_result={}, transitions=[])
        j = cp.to_json()
        restored = RegistryCheckpoint.from_json(j)
        assert restored.last_state == "pending"

    def test_default_values(self):
        cp = RegistryCheckpoint()
        assert cp.last_state is None
        assert cp.last_result == {}
        assert cp.transitions == []


class TestRunnerCheckpoint:
    """Tests for RunnerCheckpoint serialization."""

    def test_to_dict_and_back(self):
        reg_cp = RegistryCheckpoint(
            last_state="shipped",
            last_result={"shipped": True},
            transitions=[],
        )
        cp = RunnerCheckpoint(
            registry=reg_cp,
            messages=[
                {"role": "user", "content": "Ship order 1"},
                {"role": "assistant", "content": "Done!"},
            ],
            tool_calls=[{"tool": "ship", "input": {"id": "1"}}],
            timestamp=200.0,
        )

        d = cp.to_dict()
        restored = RunnerCheckpoint.from_dict(d)

        assert restored.registry.last_state == "shipped"
        assert len(restored.messages) == 2
        assert len(restored.tool_calls) == 1
        assert restored.timestamp == 200.0

    def test_to_json_and_back(self):
        cp = RunnerCheckpoint(
            messages=[{"role": "user", "content": "hello"}],
            tool_calls=[],
        )
        j = cp.to_json()
        restored = RunnerCheckpoint.from_json(j)
        assert len(restored.messages) == 1

    def test_complex_messages(self):
        """Test with messages containing tool use blocks (complex content)."""
        cp = RunnerCheckpoint(
            messages=[
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll help"},
                        {"type": "tool_use", "id": "t1", "name": "gw", "input": {"id": "1"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                    ],
                },
            ],
        )

        j = cp.to_json()
        restored = RunnerCheckpoint.from_json(j)
        assert len(restored.messages) == 3
        assert isinstance(restored.messages[1]["content"], list)


class TestRegistrySaveLoad:
    """Tests for save/load checkpoint on a live Registry."""

    def _make_sm(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})
        sm.action("approve", description="Approve", from_states=["pending"],
                  to_state="approved", params={"id": "string"})

        @sm.on_gateway
        def gw(id=""):
            return {"order": {"id": id}, "_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            return {"approved": True, "_state": "approved"}

        return sm

    def test_save_and_load_roundtrip(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})

        data = save_registry_checkpoint(reg)

        # Create fresh registry and restore
        reg2 = Registry(sm)
        assert reg2._last_state is None

        load_registry_checkpoint(reg2, data)
        assert reg2._last_state == "approved"
        assert len(reg2._transition_log) == 1
        assert reg2._transition_log[0].action == "approve"

    def test_save_empty_registry(self):
        sm = self._make_sm()
        reg = Registry(sm)

        data = save_registry_checkpoint(reg)
        assert data["last_state"] is None
        assert data["transitions"] == []

    def test_json_roundtrip(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})

        data = save_registry_checkpoint(reg)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        reg2 = Registry(sm)
        load_registry_checkpoint(reg2, restored_data)
        assert reg2._last_state == "approved"


class TestDiscoveryReportPersistence:
    """Tests for DiscoveryReport JSON serialization."""

    def test_to_json_and_from_json(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("pending", "approve", "approved", 1.0),
            TransitionRecord("approved", "ship", "shipped", 2.0),
        ])

        j = report.to_json()
        restored = DiscoveryReport.from_json(j)

        assert len(restored.transitions) == 2
        assert restored.transitions[0].state_before == "pending"
        assert restored.transitions[0].action == "approve"
        assert restored.transitions[1].state_after == "shipped"

    def test_empty_report(self):
        report = DiscoveryReport()
        j = report.to_json()
        restored = DiscoveryReport.from_json(j)
        assert len(restored.transitions) == 0

    def test_state_map_preserved(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("a", "go", "b", 1.0),
            TransitionRecord("b", "back", "a", 2.0),
        ])

        j = report.to_json()
        restored = DiscoveryReport.from_json(j)

        assert restored.to_state_map() == report.to_state_map()


class TestRunnerSaveLoadCheckpoint:
    """Tests for save_runner_checkpoint and load_runner_checkpoint."""

    def _make_sm(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})
        sm.action(
            "approve", description="Approve",
            from_states=["pending"], to_state="approved",
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw(id=""):
            return {"order": {"id": id}, "_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            return {"approved": True, "_state": "approved"}

        return sm

    def _make_runner(self, sm):
        """Create a Runner with a mock client (avoids requiring API key)."""
        from unittest.mock import MagicMock

        from hateoas_agent.runner import Runner
        mock_client = MagicMock()
        return Runner(sm, client=mock_client)

    def test_save_and_load_roundtrip(self):
        sm = self._make_sm()
        runner = self._make_runner(sm)

        # Drive the registry directly to set state
        runner._registry.handle_tool_call("gw", {"id": "1"})
        runner._registry.handle_tool_call("approve", {"id": "1"})

        messages = [
            {"role": "user", "content": "Approve order 1"},
            {"role": "assistant", "content": "Done!"},
        ]
        tool_calls = [
            {"tool": "gw", "input": {"id": "1"}},
            {"tool": "approve", "input": {"id": "1"}},
        ]

        data = save_runner_checkpoint(runner, messages, tool_calls)

        # Restore into a fresh runner
        runner2 = self._make_runner(sm)
        assert runner2._registry._last_state is None

        restored_msgs, restored_tc = load_runner_checkpoint(runner2, data)
        assert runner2._registry._last_state == "approved"
        assert len(restored_msgs) == 2
        assert len(restored_tc) == 2
        assert restored_msgs[0]["role"] == "user"
        assert restored_tc[1]["tool"] == "approve"

    def test_save_empty_runner(self):
        sm = self._make_sm()
        runner = self._make_runner(sm)

        data = save_runner_checkpoint(runner, [], [])
        assert data["registry"]["last_state"] is None
        assert data["messages"] == []
        assert data["tool_calls"] == []

    def test_json_roundtrip(self):
        sm = self._make_sm()
        runner = self._make_runner(sm)

        runner._registry.handle_tool_call("gw", {"id": "1"})
        runner._registry.handle_tool_call("approve", {"id": "1"})

        messages = [{"role": "user", "content": "hello"}]
        tool_calls = [{"tool": "gw", "input": {"id": "1"}}]

        data = save_runner_checkpoint(runner, messages, tool_calls)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        runner2 = self._make_runner(sm)
        restored_msgs, restored_tc = load_runner_checkpoint(
            runner2, restored_data,
        )
        assert runner2._registry._last_state == "approved"
        assert len(restored_msgs) == 1
        assert len(restored_tc) == 1

    def test_complex_messages_roundtrip(self):
        """Test with tool_use content blocks in messages."""
        sm = self._make_sm()
        runner = self._make_runner(sm)

        runner._registry.handle_tool_call("gw", {"id": "1"})

        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help"},
                    {
                        "type": "tool_use", "id": "t1",
                        "name": "gw", "input": {"id": "1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "result",
                    },
                ],
            },
        ]

        data = save_runner_checkpoint(runner, messages, [])
        runner2 = self._make_runner(sm)
        restored_msgs, _ = load_runner_checkpoint(runner2, data)
        assert len(restored_msgs) == 3
        assert isinstance(restored_msgs[1]["content"], list)

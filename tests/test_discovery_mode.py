"""Tests for discovery mode and transition logging."""

import logging

from hateoas_agent import Registry, StateMachine
from hateoas_agent.types import DiscoveryReport, TransitionRecord


class TestDiscoveryModeStateMachine:
    """Tests for StateMachine in discover mode."""

    def test_all_actions_available_in_any_state(self):
        sm = StateMachine("test", gateway_name="gw", mode="discover")
        sm.gateway(description="GW", params={})
        sm.action("approve", description="Approve", params={"id": "string"})
        sm.action("ship", description="Ship", params={"id": "string"})

        for state in ["pending", "approved", "shipped", "anything"]:
            actions = sm.get_actions_for_state(state)
            names = {a.name for a in actions}
            assert names == {"approve", "ship"}

    def test_from_states_optional_in_discover_mode(self):
        sm = StateMachine("test", gateway_name="gw", mode="discover")
        # Should not raise — from_states defaults to "*"
        sm.action("approve", description="Approve", params={})
        assert sm.get_transition_metadata("approve") == ("*", None)

    def test_from_states_still_accepted_in_discover_mode(self):
        sm = StateMachine("test", gateway_name="gw", mode="discover")
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            to_state="approved",
            params={},
        )
        # Even with from_states set, discover mode returns it everywhere
        assert len(sm.get_actions_for_state("shipped")) == 1

    def test_mixed_state_and_action_in_discover(self):
        sm = StateMachine("test", gateway_name="gw", mode="discover")
        sm.gateway(description="GW", params={})
        sm.state("pending", actions=[
            {"name": "old_action", "description": "Old", "params": {}},
        ])
        sm.action("new_action", description="New", params={})

        # "pending" has both; action-centric always available
        actions = sm.get_actions_for_state("pending")
        names = {a.name for a in actions}
        assert names == {"new_action", "old_action"}

        # Other states only have action-centric (state-centric is still state-bound)
        actions = sm.get_actions_for_state("other")
        names = {a.name for a in actions}
        assert names == {"new_action"}


class TestTransitionLogging:
    """Tests for Registry transition logging."""

    def _make_sm(self, mode="strict"):
        sm = StateMachine("test", gateway_name="gw", mode=mode)
        sm.gateway(description="GW", params={"id": "string"})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"] if mode == "strict" else None,
            to_state="approved",
            params={"id": "string"},
        )
        sm.action(
            "ship",
            description="Ship",
            from_states=["approved"] if mode == "strict" else None,
            to_state="shipped",
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw(id=""):
            return {"order": {"id": id}, "_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            return {"approved": True, "_state": "approved"}

        @sm.on_action("ship")
        def ship(id=""):
            return {"shipped": True, "_state": "shipped"}

        return sm

    def test_transitions_logged(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})

        report = reg.get_discovery_report()
        assert len(report.transitions) == 1
        t = report.transitions[0]
        assert t.state_before == "pending"
        assert t.action == "approve"
        assert t.state_after == "approved"
        assert t.timestamp > 0

    def test_multiple_transitions_logged(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})
        reg.handle_tool_call("ship", {"id": "1"})

        report = reg.get_discovery_report()
        assert len(report.transitions) == 2
        assert report.transitions[0].action == "approve"
        assert report.transitions[1].action == "ship"

    def test_gateway_does_not_log_transition(self):
        """Gateway establishes initial state but doesn't log as a transition."""
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})

        report = reg.get_discovery_report()
        assert len(report.transitions) == 0

    def test_no_state_change_still_logs(self):
        """Actions that don't change state still log the transition."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "add_note",
            description="Note",
            from_states="*",
            params={"note": "string"},
        )

        @sm.on_gateway
        def gw(**kw):
            return {"_state": "pending"}

        @sm.on_action("add_note")
        def add_note(note=""):
            return {"noted": True, "_state": "pending"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})
        reg.handle_tool_call("add_note", {"note": "test"})

        report = reg.get_discovery_report()
        assert len(report.transitions) == 1
        assert report.transitions[0].state_before == "pending"
        assert report.transitions[0].state_after == "pending"

    def test_discover_mode_full_flow(self):
        sm = self._make_sm(mode="discover")
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})
        reg.handle_tool_call("ship", {"id": "1"})

        report = reg.get_discovery_report()
        assert len(report.transitions) == 2

        state_map = report.to_state_map()
        assert state_map == {
            "pending": ["approve"],
            "approved": ["ship"],
        }

    def test_report_is_a_copy(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        reg.handle_tool_call("approve", {"id": "1"})

        report1 = reg.get_discovery_report()
        report2 = reg.get_discovery_report()
        assert report1.transitions is not report2.transitions


class TestDiscoveryReport:
    """Tests for DiscoveryReport methods."""

    def _make_report(self):
        return DiscoveryReport(transitions=[
            TransitionRecord("pending", "approve", "approved", 1.0),
            TransitionRecord("pending", "cancel", "cancelled", 2.0),
            TransitionRecord("approved", "ship", "shipped", 3.0),
            TransitionRecord("approved", "cancel", "cancelled", 4.0),
        ])

    def test_to_state_map(self):
        report = self._make_report()
        state_map = report.to_state_map()
        assert state_map == {
            "approved": ["cancel", "ship"],
            "pending": ["approve", "cancel"],
        }

    def test_to_action_map(self):
        report = self._make_report()
        action_map = report.to_action_map()

        assert action_map["approve"] == {
            "from_states": ["pending"],
            "to_state": "approved",
        }
        assert action_map["ship"] == {
            "from_states": ["approved"],
            "to_state": "shipped",
        }
        assert action_map["cancel"]["from_states"] == ["approved", "pending"]
        # cancel goes to "cancelled" from both states — single to_state
        assert action_map["cancel"]["to_state"] == "cancelled"

    def test_to_action_map_multiple_to_states(self):
        """Action that transitions to different states → to_state is None."""
        report = DiscoveryReport(transitions=[
            TransitionRecord("a", "do_thing", "b", 1.0),
            TransitionRecord("a", "do_thing", "c", 2.0),
        ])
        action_map = report.to_action_map()
        assert action_map["do_thing"]["to_state"] is None

    def test_to_python(self):
        report = self._make_report()
        code = report.to_python("orders")
        assert 'orders.action("approve"' in code
        assert "from_states=['pending']" in code
        assert 'to_state="approved"' in code
        assert 'orders.action("ship"' in code
        assert 'orders.action("cancel"' in code

    def test_to_python_custom_var_name(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("a", "do_thing", "b", 1.0),
        ])
        code = report.to_python("my_sm")
        assert 'my_sm.action("do_thing"' in code

    def test_empty_report(self):
        report = DiscoveryReport()
        assert report.to_state_map() == {}
        assert report.to_action_map() == {}
        assert report.to_python() == ""


class TestToStateWarning:
    """Tests for to_state mismatch warnings."""

    def test_warns_on_mismatch(self, caplog):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            to_state="approved",
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw(**kw):
            return {"_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            # Returns different state than declared to_state
            return {"_state": "rejected"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            reg.handle_tool_call("approve", {"id": "1"})

        assert any(
            "declared to_state='approved'" in r.message
            and "returned _state='rejected'" in r.message
            for r in caplog.records
        )

    def test_no_warning_when_matching(self, caplog):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            to_state="approved",
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw(**kw):
            return {"_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            return {"_state": "approved"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            reg.handle_tool_call("approve", {"id": "1"})

        assert not any(
            "declared to_state" in r.message for r in caplog.records
        )

    def test_no_warning_when_no_to_state_declared(self, caplog):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "add_note",
            description="Note",
            from_states="*",
            params={},
        )

        @sm.on_gateway
        def gw(**kw):
            return {"_state": "pending"}

        @sm.on_action("add_note")
        def add_note(**kw):
            return {"_state": "pending"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            reg.handle_tool_call("add_note", {})

        assert not any(
            "declared to_state" in r.message for r in caplog.records
        )

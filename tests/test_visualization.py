"""Tests for Mermaid visualization generation."""

from hateoas_agent import StateMachine
from hateoas_agent.types import DiscoveryReport, TransitionRecord
from hateoas_agent.visualization import (
    discovery_report_to_mermaid,
    state_machine_to_mermaid,
)


class TestStateMachineToMermaid:
    """Tests for state_machine_to_mermaid."""

    def test_basic_transitions(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "approve", description="Approve",
            from_states=["pending"], to_state="approved", params={},
        )
        sm.action(
            "ship", description="Ship",
            from_states=["approved"], to_state="shipped", params={},
        )

        mermaid = state_machine_to_mermaid(sm)
        assert "stateDiagram-v2" in mermaid
        assert "[*] --> pending" in mermaid
        assert "pending --> approved : approve" in mermaid
        assert "approved --> shipped : ship" in mermaid

    def test_universal_action_rendered_as_note(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "approve", description="Approve",
            from_states=["pending"], to_state="approved", params={},
        )
        sm.action("add_note", description="Note", from_states="*", params={})

        mermaid = state_machine_to_mermaid(sm)
        assert "note right of pending" in mermaid
        assert "add_note" in mermaid
        assert "(all states)" in mermaid

    def test_universal_action_with_to_state(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("cancel", description="Cancel", from_states="*", to_state="cancelled", params={})
        sm.state("pending", actions=[{"name": "old", "description": "Old", "params": {}}])

        mermaid = state_machine_to_mermaid(sm)
        assert "cancel → cancelled" in mermaid

    def test_empty_state_machine(self):
        sm = StateMachine("test", gateway_name="gw")
        mermaid = state_machine_to_mermaid(sm)
        assert mermaid == "stateDiagram-v2"

    def test_action_without_to_state(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("modify", description="Modify", from_states=["pending"], params={})

        mermaid = state_machine_to_mermaid(sm)
        assert "[*] --> pending" in mermaid
        # No transition arrow since no to_state
        assert "-->" in mermaid  # Only the initial state arrow

    def test_multiple_from_states(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("cancel", description="Cancel", from_states=["pending", "approved"],
                  to_state="cancelled", params={})

        mermaid = state_machine_to_mermaid(sm)
        assert "pending --> cancelled : cancel" in mermaid
        assert "approved --> cancelled : cancel" in mermaid

    def test_convenience_method(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("approve", description="Approve", from_states=["pending"],
                  to_state="approved", params={})

        assert sm.to_mermaid() == state_machine_to_mermaid(sm)

    def test_state_centric_states_appear(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.state("pending", actions=[
            {"name": "approve", "description": "Approve", "params": {}},
        ])
        sm.action("ship", description="Ship", from_states=["approved"],
                  to_state="shipped", params={})

        mermaid = state_machine_to_mermaid(sm)
        # Both state-centric and action-centric states should appear
        assert "approved --> shipped : ship" in mermaid


class TestDiscoveryReportToMermaid:
    """Tests for discovery_report_to_mermaid."""

    def test_basic_report(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("pending", "approve", "approved", 1.0),
            TransitionRecord("approved", "ship", "shipped", 2.0),
        ])

        mermaid = discovery_report_to_mermaid(report)
        assert "stateDiagram-v2" in mermaid
        assert "[*] --> pending" in mermaid
        assert "pending --> approved : approve" in mermaid
        assert "approved --> shipped : ship" in mermaid

    def test_deduplicates_transitions(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("pending", "approve", "approved", 1.0),
            TransitionRecord("pending", "approve", "approved", 2.0),
            TransitionRecord("pending", "approve", "approved", 3.0),
        ])

        mermaid = discovery_report_to_mermaid(report)
        # Should only appear once
        assert mermaid.count("pending --> approved : approve") == 1

    def test_empty_report(self):
        report = DiscoveryReport()
        mermaid = discovery_report_to_mermaid(report)
        assert mermaid == "stateDiagram-v2"

    def test_convenience_method(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("a", "go", "b", 1.0),
        ])

        assert report.to_mermaid() == discovery_report_to_mermaid(report)

    def test_self_transition(self):
        report = DiscoveryReport(transitions=[
            TransitionRecord("pending", "add_note", "pending", 1.0),
        ])

        mermaid = discovery_report_to_mermaid(report)
        assert "pending --> pending : add_note" in mermaid

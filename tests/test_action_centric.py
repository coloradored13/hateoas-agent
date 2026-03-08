"""Tests for the action-centric StateMachine API (.action() method)."""

import pytest

from hateoas_agent import Registry, StateMachine


class TestActionMethod:
    """Tests for StateMachine.action() basic definition."""

    def test_basic_action_definition(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve it",
            from_states=["pending"],
            params={"id": "string"},
        )

        actions = sm.get_actions_for_state("pending")
        assert len(actions) == 1
        assert actions[0].name == "approve"
        assert actions[0].description == "Approve it"
        assert actions[0].params == {"id": "string"}

    def test_action_not_in_other_states(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve it",
            from_states=["pending"],
            params={"id": "string"},
        )

        assert sm.get_actions_for_state("approved") == []
        assert sm.get_actions_for_state("shipped") == []

    def test_universal_action_star(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "add_note",
            description="Add a note",
            from_states="*",
            params={"note": "string"},
        )

        for state in ["pending", "approved", "shipped", "cancelled", "anything"]:
            actions = sm.get_actions_for_state(state)
            assert len(actions) == 1
            assert actions[0].name == "add_note"

    def test_action_in_multiple_states(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "cancel",
            description="Cancel",
            from_states=["pending", "approved"],
            params={"id": "string"},
        )

        assert len(sm.get_actions_for_state("pending")) == 1
        assert len(sm.get_actions_for_state("approved")) == 1
        assert len(sm.get_actions_for_state("shipped")) == 0

    def test_to_state_metadata(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            to_state="approved",
            params={"id": "string"},
        )

        meta = sm.get_transition_metadata("approve")
        assert meta is not None
        assert meta == (["pending"], "approved")

    def test_to_state_metadata_none_when_not_set(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "add_note",
            description="Note",
            from_states="*",
            params={},
        )

        meta = sm.get_transition_metadata("add_note")
        assert meta is not None
        assert meta == ("*", None)

    def test_no_metadata_for_state_centric_action(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.state("pending", actions=[
            {"name": "approve", "description": "Approve", "params": {"id": "string"}},
        ])

        assert sm.get_transition_metadata("approve") is None

    def test_required_params(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            params={"id": "string", "reason": "string"},
            required=["id"],
        )

        actions = sm.get_actions_for_state("pending")
        assert actions[0].required == ["id"]


class TestActionValidation:
    """Tests for .action() input validation."""

    def test_strict_mode_requires_from_states(self):
        sm = StateMachine("test", gateway_name="gw")
        with pytest.raises(ValueError, match="from_states is required"):
            sm.action("approve", description="Approve", params={})

    def test_empty_from_states_list_rejected(self):
        sm = StateMachine("test", gateway_name="gw")
        with pytest.raises(ValueError, match="cannot be an empty list"):
            sm.action(
                "approve",
                description="Approve",
                from_states=[],
                params={},
            )

    def test_invalid_from_states_string(self):
        sm = StateMachine("test", gateway_name="gw")
        with pytest.raises(ValueError, match="must be '\\*'"):
            sm.action(
                "approve",
                description="Approve",
                from_states="pending",
                params={},
            )

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            StateMachine("test", gateway_name="gw", mode="invalid")


class TestMixingActionAndState:
    """Tests for using .action() and .state() together."""

    def test_both_contribute_to_same_state(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state("pending", actions=[
            {"name": "old_action", "description": "Old", "params": {}},
        ])
        sm.action(
            "new_action",
            description="New",
            from_states=["pending"],
            params={},
        )

        actions = sm.get_actions_for_state("pending")
        names = {a.name for a in actions}
        assert names == {"old_action", "new_action"}

    def test_action_centric_takes_precedence(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state("pending", actions=[
            {"name": "cancel", "description": "Old desc", "params": {}},
        ])
        sm.action(
            "cancel",
            description="New desc",
            from_states=["pending"],
            params={"reason": "string"},
        )

        actions = sm.get_actions_for_state("pending")
        assert len(actions) == 1
        assert actions[0].description == "New desc"
        assert actions[0].params == {"reason": "string"}

    def test_universal_action_with_state_centric(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state("pending", actions=[
            {"name": "approve", "description": "Approve", "params": {}},
        ])
        sm.action("add_note", description="Note", from_states="*", params={})

        actions = sm.get_actions_for_state("pending")
        names = {a.name for a in actions}
        assert names == {"approve", "add_note"}

        # Universal action also appears in states without state-centric defs
        actions = sm.get_actions_for_state("shipped")
        assert len(actions) == 1
        assert actions[0].name == "add_note"


class TestHandlerAttachment:
    """Tests for handler attachment with .action()."""

    def test_handler_after_action(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("approve", description="Approve", from_states=["pending"], params={})

        @sm.on_action("approve")
        def handle(**kw):
            return {"done": True}

        actions = sm.get_actions_for_state("pending")
        assert actions[0].handler is not None
        assert actions[0].handler() == {"done": True}

    def test_handler_before_action(self):
        sm = StateMachine("test", gateway_name="gw")

        @sm.on_action("approve")
        def handle(**kw):
            return {"done": True}

        sm.action("approve", description="Approve", from_states=["pending"], params={})

        actions = sm.get_actions_for_state("pending")
        assert actions[0].handler is not None
        assert actions[0].handler() == {"done": True}

    def test_get_handler(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("approve", description="Approve", from_states=["pending"], params={})

        @sm.on_action("approve")
        def handle(**kw):
            return {"done": True}

        assert sm.get_handler("approve") is handle


class TestGetAllActionNames:
    """Tests for get_all_action_names with both stores."""

    def test_includes_action_centric(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("approve", description="Approve", from_states=["pending"], params={})

        assert "approve" in sm.get_all_action_names()

    def test_includes_state_centric(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.state("pending", actions=[
            {"name": "approve", "description": "Approve", "params": {}},
        ])

        assert "approve" in sm.get_all_action_names()

    def test_union_of_both(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.state("pending", actions=[
            {"name": "state_action", "description": "State", "params": {}},
        ])
        sm.action("action_action", description="Action", from_states=["pending"], params={})

        names = sm.get_all_action_names()
        assert names == {"state_action", "action_action"}


class TestModeProperty:
    """Tests for mode property."""

    def test_default_strict(self):
        sm = StateMachine("test", gateway_name="gw")
        assert sm.mode == "strict"

    def test_discover_mode(self):
        sm = StateMachine("test", gateway_name="gw", mode="discover")
        assert sm.mode == "discover"


class TestActionCentricWithRegistry:
    """Tests that action-centric definitions work through Registry."""

    def test_registry_validates_action_centric(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw_handler(**kw):
            return {"result": "ok", "_state": "pending"}

        @sm.on_action("approve")
        def approve_handler(id=""):
            return {"approved": True, "_state": "approved"}

        reg = Registry(sm)

        # Call gateway to establish state
        reg.handle_tool_call("gw", {})
        assert reg._last_state == "pending"

        # Call approve — should work in pending state
        reg.handle_tool_call("approve", {"id": "123"})
        assert reg._last_state == "approved"

    def test_registry_rejects_wrong_state(self):
        from hateoas_agent.errors import InvalidActionError

        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "ship",
            description="Ship",
            from_states=["approved"],
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw_handler(**kw):
            return {"result": "ok", "_state": "pending"}

        @sm.on_action("ship")
        def ship_handler(id=""):
            return {"shipped": True, "_state": "shipped"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        with pytest.raises(InvalidActionError):
            reg.handle_tool_call("ship", {"id": "123"})

    def test_tool_schemas_include_action_centric(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            params={"id": "string"},
        )

        @sm.on_gateway
        def gw_handler(**kw):
            return {"_state": "pending"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "gw" in names
        assert "approve" in names

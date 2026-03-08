"""Edge case tests for hateoas-agent."""

import pytest

from hateoas_agent import (
    InvalidActionError,
    NoGatewayError,
    NoHandlerError,
    Registry,
    Resource,
    StateMachine,
    action,
    gateway,
    state,
)


class TestStateMachineEdgeCases:
    """Edge cases for StateMachine."""

    def test_no_states_defined(self):
        """StateMachine with no states defined returns empty actions."""
        sm = StateMachine("empty", gateway_name="start")
        sm.gateway(description="Start", params={})

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "any_state"}

        assert sm.get_actions_for_state("any_state") == []
        assert sm.get_all_action_names() == set()

    def test_no_gateway_defined(self):
        """StateMachine with no gateway returns None."""
        sm = StateMachine("no_gw")
        assert sm.get_gateway() is None

    def test_gateway_defined_but_no_handler(self):
        """Gateway defined but handler never registered."""
        sm = StateMachine("partial", gateway_name="start")
        sm.gateway(description="Start", params={})
        gw = sm.get_gateway()
        # Gateway def exists but handler is None
        assert gw is not None
        assert gw.handler is None

    def test_action_handler_registered_before_state(self):
        """Register handler before state definition."""
        sm = StateMachine("order_test", gateway_name="start")
        sm.gateway(description="Start", params={})

        @sm.on_action("do_thing")
        def handle(**kw):
            return {"done": True, "_state": "active"}

        # Now define the state
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {}},
            ],
        )

        # Handler should still be accessible
        handler = sm.get_handler("do_thing")
        assert handler is not None
        result = handler()
        assert result["done"] is True

    def test_get_handler_for_unregistered_action(self):
        sm = StateMachine("test")
        assert sm.get_handler("nonexistent") is None

    def test_empty_string_state_name(self):
        """Empty string state name works like any other state."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "",
            actions=[
                {"name": "do_it", "description": "Do", "params": {}},
            ],
        )

        actions = sm.get_actions_for_state("")
        assert len(actions) == 1
        assert actions[0].name == "do_it"

    def test_action_in_multiple_states(self):
        """Same action name registered in multiple states."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "pending",
            actions=[
                {"name": "cancel", "description": "Cancel", "params": {}},
            ],
        )
        sm.state(
            "approved",
            actions=[
                {"name": "cancel", "description": "Cancel", "params": {}},
            ],
        )

        @sm.on_action("cancel")
        def cancel(**kw):
            return {"cancelled": True, "_state": "cancelled"}

        pending_actions = sm.get_actions_for_state("pending")
        approved_actions = sm.get_actions_for_state("approved")

        assert any(a.name == "cancel" for a in pending_actions)
        assert any(a.name == "cancel" for a in approved_actions)
        # Both should have the handler attached
        for a in pending_actions:
            if a.name == "cancel":
                assert a.handler is not None
        for a in approved_actions:
            if a.name == "cancel":
                assert a.handler is not None


class TestResourceEdgeCases:
    """Edge cases for Resource."""

    def test_no_gateway_method(self):
        """Resource with no @gateway method."""

        class NoGwResource(Resource):
            name = "no_gw"

            @action(name="do_thing", description="Do", params={})
            def do_thing(self):
                return {"done": True}

        r = NoGwResource()
        assert r.get_gateway() is None

    def test_no_action_methods(self):
        """Resource with only a gateway, no actions."""

        class GwOnlyResource(Resource):
            name = "gw_only"

            @gateway(name="start", description="Start", params={})
            def start(self):
                return {"ok": True, "_state": "active"}

        r = GwOnlyResource()
        assert r.get_gateway() is not None
        assert r.get_all_action_names() == set()
        assert r.get_actions_for_state("active") == []

    def test_action_without_state_decorator(self):
        """Action without @state appears in all states."""

        class AnyStateResource(Resource):
            name = "any"

            @gateway(name="start", description="Start", params={})
            def start(self):
                return {"ok": True, "_state": "x"}

            @action(name="universal", description="Universal action", params={})
            def universal(self):
                return {"done": True, "_state": "x"}

        r = AnyStateResource()
        assert any(a.name == "universal" for a in r.get_actions_for_state("x"))
        assert any(a.name == "universal" for a in r.get_actions_for_state("y"))
        assert any(a.name == "universal" for a in r.get_actions_for_state("anything"))

    def test_action_with_multiple_states(self):
        """@state with multiple state arguments."""

        class MultiStateResource(Resource):
            name = "multi"

            @gateway(name="gw", description="GW", params={})
            def gw(self):
                return {"ok": True, "_state": "a"}

            @action(name="shared_action", description="Shared", params={})
            @state("a", "b")
            def shared(self):
                return {"ok": True}

            @action(name="only_c", description="Only C", params={})
            @state("c")
            def only_c(self):
                return {"ok": True}

        r = MultiStateResource()
        assert any(a.name == "shared_action" for a in r.get_actions_for_state("a"))
        assert any(a.name == "shared_action" for a in r.get_actions_for_state("b"))
        assert not any(a.name == "shared_action" for a in r.get_actions_for_state("c"))
        assert any(a.name == "only_c" for a in r.get_actions_for_state("c"))

    def test_get_handler_for_unknown_action(self):
        class SimpleResource(Resource):
            name = "simple"

            @gateway(name="gw", description="GW", params={})
            def gw(self):
                return {}

        r = SimpleResource()
        assert r.get_handler("nonexistent") is None


class TestRegistryEdgeCases:
    """Edge cases for Registry."""

    def test_registry_no_gateway_raises(self):
        """Registry with a resource that has no gateway."""
        sm = StateMachine("no_gw")
        reg = Registry(sm)

        with pytest.raises(NoGatewayError):
            reg.gateway_name

        with pytest.raises(NoGatewayError):
            reg.get_gateway_tool_schema()

    def test_action_before_any_gateway_call(self):
        """Calling an action before any gateway call (no state set) raises error."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        @sm.on_action("do_thing")
        def do_thing(**kw):
            return {"done": True, "_state": "active"}

        reg = Registry(sm)
        # No gateway called yet — action must be rejected
        with pytest.raises(InvalidActionError):
            reg.handle_tool_call("do_thing", {})

    def test_handler_returning_no_state_key(self):
        """Handler returning dict without _state key."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            return {"message": "No state here"}

        reg = Registry(sm)
        result = reg.handle_tool_call("gw", {})
        # Should work fine, just no state update
        assert "No state here" in result
        assert reg._last_state is None

    def test_handler_returning_non_dict(self):
        """Handler returning a non-dict result."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            return "just a string"

        reg = Registry(sm)
        result = reg.handle_tool_call("gw", {})
        # Should wrap in {"result": ...}
        assert "just a string" in result

    def test_no_handler_for_known_action_raises(self):
        """Action defined in state but handler never registered."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "orphan_action", "description": "Orphan", "params": {}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        reg = Registry(sm)
        # Set state via gateway
        reg.handle_tool_call("gw", {})

        # orphan_action is valid for state but has no handler
        with pytest.raises(NoHandlerError):
            reg.handle_tool_call("orphan_action", {})

    def test_is_known_action(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "known_action", "description": "Known", "params": {}},
            ],
        )

        reg = Registry(sm)
        assert reg.is_known_action("known_action") is True
        assert reg.is_known_action("unknown_action") is False

    def test_gateway_handler_exception_propagates(self):
        """Exception in gateway handler propagates up."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            raise RuntimeError("Gateway broke")

        reg = Registry(sm)
        with pytest.raises(RuntimeError, match="Gateway broke"):
            reg.handle_tool_call("gw", {})

    def test_state_transitions_track_correctly(self):
        """Verify _last_state updates after each call."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "a",
            actions=[
                {"name": "go_b", "description": "Go to B", "params": {}},
            ],
        )
        sm.state(
            "b",
            actions=[
                {"name": "go_a", "description": "Go to A", "params": {}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "a"}

        @sm.on_action("go_b")
        def go_b(**kw):
            return {"ok": True, "_state": "b"}

        @sm.on_action("go_a")
        def go_a(**kw):
            return {"ok": True, "_state": "a"}

        reg = Registry(sm)
        assert reg._last_state is None

        reg.handle_tool_call("gw", {})
        assert reg._last_state == "a"

        reg.handle_tool_call("go_b", {})
        assert reg._last_state == "b"

        reg.handle_tool_call("go_a", {})
        assert reg._last_state == "a"


class TestErrorClasses:
    """Test error class instantiation and attributes."""

    def test_invalid_action_error_attributes(self):
        err = InvalidActionError("ship", "pending", ["approve", "cancel"])
        assert err.action == "ship"
        assert err.state == "pending"
        assert err.valid_actions == ["approve", "cancel"]
        assert "ship" in str(err)
        assert "pending" in str(err)

    def test_no_handler_error_attributes(self):
        err = NoHandlerError("missing_action")
        assert err.action == "missing_action"
        assert "missing_action" in str(err)

    def test_no_gateway_error(self):
        err = NoGatewayError()
        assert "gateway" in str(err).lower()

    def test_phantom_tool_error_attributes(self):
        from hateoas_agent import PhantomToolError

        err = PhantomToolError("ghost_tool", "active")
        assert err.tool_name == "ghost_tool"
        assert err.state == "active"
        assert "ghost_tool" in str(err)

    def test_phantom_tool_error_none_state(self):
        from hateoas_agent import PhantomToolError

        err = PhantomToolError("ghost_tool", None)
        assert err.state is None


class TestStateTypeValidation:
    """Test that _state values are validated as strings."""

    def test_non_string_state_raises_type_error(self):
        """Handler returning non-string _state raises TypeError."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": 42}

        reg = Registry(sm)
        with pytest.raises(TypeError, match="_state must be a string"):
            reg.handle_tool_call("gw", {})

    def test_none_state_raises_type_error(self):
        """Handler returning _state=None raises TypeError."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": None}

        reg = Registry(sm)
        with pytest.raises(TypeError, match="_state must be a string"):
            reg.handle_tool_call("gw", {})

    def test_dict_state_raises_type_error(self):
        """Handler returning _state as dict raises TypeError."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": {"nested": "obj"}}

        reg = Registry(sm)
        with pytest.raises(TypeError, match="_state must be a string"):
            reg.handle_tool_call("gw", {})

    def test_string_state_accepted(self):
        """String _state works fine."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state("active", actions=[])

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})
        assert reg._last_state == "active"


class TestParamFiltering:
    """Test that tool_input is filtered against declared params."""

    def test_extra_gateway_params_stripped(self):
        """Extra params from Claude are stripped before calling gateway handler."""
        received = {}
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})

        @sm.on_gateway
        def gw(**kw):
            received.update(kw)
            return {"ok": True, "_state": "active"}

        sm.state("active", actions=[])
        reg = Registry(sm)
        reg.handle_tool_call("gw", {"id": "123", "__class__": "evil", "extra": "bad"})
        assert received == {"id": "123"}

    def test_extra_action_params_stripped(self):
        """Extra params from Claude are stripped before calling action handler."""
        received = {}
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {"id": "string"}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        @sm.on_action("do_thing")
        def do_thing(**kw):
            received.update(kw)
            return {"done": True, "_state": "active"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})
        reg.handle_tool_call("do_thing", {"id": "1", "injected": "value"})
        assert received == {"id": "1"}

    def test_empty_params_passes_all(self):
        """When no params are declared, all input passes through."""
        received = {}
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})

        @sm.on_gateway
        def gw(**kw):
            received.update(kw)
            return {"ok": True}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {"foo": "bar"})
        assert received == {"foo": "bar"}


class TestActionDefCopies:
    """Test that StateMachine returns fresh ActionDef copies."""

    def test_mutation_does_not_leak(self):
        """Mutating returned ActionDefs doesn't affect future calls."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {"id": "string"}},
            ],
        )

        first = sm.get_actions_for_state("active")
        first[0].description = "MUTATED"

        second = sm.get_actions_for_state("active")
        assert second[0].description == "Do"

    def test_copies_are_separate_objects(self):
        """Each call returns new ActionDef instances."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {}},
            ],
        )

        first = sm.get_actions_for_state("active")
        second = sm.get_actions_for_state("active")
        assert first[0] is not second[0]


class TestTruncatedFlag:
    """Test RunResult.truncated flag."""

    def test_normal_result_not_truncated(self):
        from hateoas_agent import RunResult

        result = RunResult(text="done", messages=[], tool_calls=[])
        assert result.truncated is False

    def test_truncated_in_repr(self):
        from hateoas_agent import RunResult

        result = RunResult(text="", messages=[], tool_calls=[], truncated=True)
        assert "truncated" in repr(result)

    def test_not_truncated_not_in_repr(self):
        from hateoas_agent import RunResult

        result = RunResult(text="done", messages=[], tool_calls=[])
        assert "truncated" not in repr(result)


class TestRequiredFields:
    """Test required field support in gateway and action schemas."""

    def test_gateway_required_in_schema(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(
            description="GW",
            params={"id": "string", "name": "string"},
            required=["id"],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True}

        reg = Registry(sm)
        schema = reg.get_gateway_tool_schema()
        assert schema["input_schema"]["required"] == ["id"]

    def test_gateway_no_required_omitted(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True}

        reg = Registry(sm)
        schema = reg.get_gateway_tool_schema()
        assert "required" not in schema["input_schema"]

    def test_action_required_in_advertisement(self):
        from hateoas_agent import format_result_with_actions
        from hateoas_agent.types import ActionDef

        actions = [
            ActionDef(
                name="do_thing",
                description="Do",
                params={"id": "string"},
                required=["id"],
            )
        ]
        output = format_result_with_actions({"ok": True}, actions)
        assert 'required: ["id"]' in output

    def test_action_no_required_not_in_advertisement(self):
        from hateoas_agent import format_result_with_actions
        from hateoas_agent.types import ActionDef

        actions = [
            ActionDef(
                name="do_thing",
                description="Do",
                params={"id": "string"},
            )
        ]
        output = format_result_with_actions({"ok": True}, actions)
        assert "required" not in output

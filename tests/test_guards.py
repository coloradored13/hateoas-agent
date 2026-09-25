"""Tests for conditional actions (guards)."""

import pytest

from hateoas_agent import Registry, StateMachine
from hateoas_agent.errors import InvalidActionError


class TestGuardFiltering:
    """Tests for StateMachine.filter_actions with guards."""

    def test_guard_passes(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "refund",
            description="Refund",
            from_states=["shipped"],
            params={},
            guard=lambda ctx: ctx.get("total", 0) < 500,
        )

        actions = sm.get_actions_for_state("shipped")
        assert len(actions) == 1

        # Guard passes: total < 500
        filtered = sm.filter_actions(actions, {"total": 100})
        assert len(filtered) == 1
        assert filtered[0].name == "refund"

    def test_guard_fails(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "refund",
            description="Refund",
            from_states=["shipped"],
            params={},
            guard=lambda ctx: ctx.get("total", 0) < 500,
        )

        actions = sm.get_actions_for_state("shipped")
        filtered = sm.filter_actions(actions, {"total": 999})
        assert len(filtered) == 0

    def test_no_guard_always_included(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("approve", description="Approve", from_states=["pending"], params={})

        actions = sm.get_actions_for_state("pending")
        filtered = sm.filter_actions(actions, {})
        assert len(filtered) == 1

    def test_guard_exception_excludes_action(self):
        """Guard that raises an exception should exclude the action (fail-closed)."""

        def bad_guard(ctx):
            raise RuntimeError("oops")

        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "risky",
            description="Risky",
            from_states=["active"],
            params={},
            guard=bad_guard,
        )

        actions = sm.get_actions_for_state("active")
        filtered = sm.filter_actions(actions, {})
        assert len(filtered) == 0

    def test_mixed_guarded_and_unguarded(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("always", description="Always", from_states=["pending"], params={})
        sm.action(
            "sometimes",
            description="Sometimes",
            from_states=["pending"],
            params={},
            guard=lambda ctx: ctx.get("flag", False),
        )

        actions = sm.get_actions_for_state("pending")
        assert len(actions) == 2

        # Guard fails — only unguarded action remains
        filtered = sm.filter_actions(actions, {"flag": False})
        assert len(filtered) == 1
        assert filtered[0].name == "always"

        # Guard passes — both actions available
        filtered = sm.filter_actions(actions, {"flag": True})
        assert len(filtered) == 2

    def test_empty_context_passed_to_guard(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action(
            "check",
            description="Check",
            from_states=["active"],
            params={},
            guard=lambda ctx: len(ctx) == 0,
        )

        actions = sm.get_actions_for_state("active")
        filtered = sm.filter_actions(actions, None)
        assert len(filtered) == 1

    def test_no_guards_returns_all(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.action("a", description="A", from_states=["s"], params={})
        sm.action("b", description="B", from_states=["s"], params={})

        actions = sm.get_actions_for_state("s")
        filtered = sm.filter_actions(actions, {"anything": True})
        assert len(filtered) == 2


class TestGuardWithRegistry:
    """Tests that guards are applied through Registry flow."""

    def _make_sm(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})
        sm.action(
            "approve",
            description="Approve",
            from_states=["pending"],
            to_state="approved",
            params={"id": "string"},
        )
        sm.action(
            "refund",
            description="Refund",
            from_states=["pending"],
            params={"id": "string"},
            guard=lambda ctx: ctx.get("total", 0) < 500,
        )

        @sm.on_gateway
        def gw(id=""):
            return {"order": {"id": id, "total": 100}, "_state": "pending"}

        @sm.on_action("approve")
        def approve(id=""):
            return {"approved": True, "_state": "approved"}

        @sm.on_action("refund")
        def refund(id=""):
            return {"refunded": True, "_state": "refunded"}

        return sm

    def test_guard_affects_tool_schemas(self):
        sm = self._make_sm()
        reg = Registry(sm)

        reg.handle_tool_call("gw", {"id": "1"})
        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        # total=100 < 500, so refund should be available
        assert "refund" in names
        assert "approve" in names

    def test_guard_excludes_from_schemas_after_high_total(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={"id": "string"})
        sm.action(
            "refund",
            description="Refund",
            from_states=["pending"],
            params={"id": "string"},
            guard=lambda ctx: ctx.get("order", {}).get("total", 0) < 500,
        )

        @sm.on_gateway
        def gw(id=""):
            return {"order": {"id": id, "total": 999}, "_state": "pending"}

        @sm.on_action("refund")
        def refund(id=""):
            return {"refunded": True, "_state": "refunded"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {"id": "1"})

        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "refund" not in names

    def test_guard_blocks_action_call(self):
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "refund",
            description="Refund",
            from_states=["pending"],
            params={},
            guard=lambda ctx: ctx.get("total", 0) < 500,
        )

        @sm.on_gateway
        def gw():
            return {"total": 999, "_state": "pending"}

        @sm.on_action("refund")
        def refund():
            return {"refunded": True, "_state": "refunded"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})

        # Refund is guarded out, so calling it should raise InvalidActionError
        with pytest.raises(InvalidActionError):
            reg.handle_tool_call("refund", {})

    def test_guard_context_updates_after_action(self):
        """After an action, _last_result is updated and guards re-evaluated."""
        sm = StateMachine("test", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "step1",
            description="Step 1",
            from_states=["start"],
            to_state="middle",
            params={},
        )
        sm.action(
            "step2",
            description="Step 2",
            from_states=["middle"],
            params={},
            guard=lambda ctx: ctx.get("step1_done", False),
        )

        @sm.on_gateway
        def gw():
            return {"_state": "start"}

        @sm.on_action("step1")
        def step1():
            return {"step1_done": True, "_state": "middle"}

        @sm.on_action("step2")
        def step2():
            return {"done": True, "_state": "done"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})
        reg.handle_tool_call("step1", {})

        # After step1, _last_result has step1_done=True, so step2 guard passes
        schemas = reg.get_current_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "step2" in names

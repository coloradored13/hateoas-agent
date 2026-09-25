"""Recoverable errors return LLM-friendly responses that inline valid actions.

When the model calls something it can't (wrong-state action, phantom tool), the
error sent back should name the problem AND list the actions that *are* valid
right now, so the model can self-correct in the same turn — instead of a bare
"that's not available, go look at the previous message" or an opaque internal
error. Genuine internal errors (handler crashes) stay generic; developer bugs
(strict-mode transition violations) propagate.
"""

from types import SimpleNamespace

from hateoas_agent import Runner, StateMachine
from hateoas_agent.errors import InvalidActionError, NoHandlerError
from hateoas_agent.mcp_server import _recoverable_error_text
from hateoas_agent.registry import Registry


def _machine():
    sm = StateMachine("orders", gateway_name="query")
    sm.gateway(description="q", params={"order_id": "string"})
    sm.action(
        "approve",
        description="Approve the order",
        from_states=["pending"],
        to_state="approved",
        params={"order_id": "string"},
    )
    sm.action(
        "ship",
        description="Ship the order",
        from_states=["approved"],
        to_state="shipped",
        params={"order_id": "string"},
    )

    @sm.on_gateway
    def g(order_id=None):
        return {"order": order_id, "_state": "pending"}

    @sm.on_action("approve")
    def a(order_id=None):
        return {"ok": True, "_state": "approved"}

    @sm.on_action("ship")
    def s(order_id=None):
        return {"ok": True, "_state": "shipped"}

    return sm


def _fake_client(script):
    it = iter(script)
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(it)))


def _tu(name, inp, _id="t"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=_id)


def _txt(s):
    return SimpleNamespace(type="text", text=s)


class TestRunnerFriendlyErrors:
    def test_wrong_state_action_lists_valid_actions(self):
        # In 'pending', call 'ship' (only valid in 'approved'). The error result
        # should be is_error AND name the valid action 'approve'.
        runner = Runner(
            _machine(),
            client=_fake_client(
                [
                    SimpleNamespace(content=[_tu("query", {"order_id": "1"}, "t0")]),
                    SimpleNamespace(content=[_tu("ship", {"order_id": "1"}, "t1")]),
                    SimpleNamespace(content=[_txt("ok")]),
                ]
            ),
        )
        res = runner.run("ship it")
        err = res.messages[4]["content"][0]
        assert err["is_error"] is True
        assert "Available actions" in err["content"]
        assert "approve" in err["content"]  # the model is told what it CAN do
        assert "ship" not in err["content"].split("Available actions")[1]

    def test_phantom_tool_lists_valid_actions(self):
        runner = Runner(
            _machine(),
            client=_fake_client(
                [
                    SimpleNamespace(content=[_tu("query", {"order_id": "1"}, "t0")]),
                    SimpleNamespace(content=[_tu("delete_everything", {}, "t1")]),
                    SimpleNamespace(content=[_txt("ok")]),
                ]
            ),
        )
        res = runner.run("delete everything")
        err = res.messages[4]["content"][0]
        assert err["is_error"] is True
        assert "delete_everything" in err["content"]
        assert "Available actions" in err["content"]
        assert "approve" in err["content"]

    def test_phantom_before_gateway_has_no_actions_but_still_friendly(self):
        # Phantom tool called before any gateway -> no valid actions yet, but the
        # response is still a clean is_error naming the bad tool (no crash).
        runner = Runner(
            _machine(),
            client=_fake_client(
                [
                    SimpleNamespace(content=[_tu("delete_everything", {}, "t0")]),
                    SimpleNamespace(content=[_txt("ok")]),
                ]
            ),
        )
        res = runner.run("go")
        err = res.messages[2]["content"][0]
        assert err["is_error"] is True
        assert "delete_everything" in err["content"]


class TestMcpFriendlyErrors:
    def test_wrong_state_error_text_inlines_actions(self):
        reg = Registry(_machine())
        reg.handle_tool_call("query", {"order_id": "1"})  # -> pending
        try:
            reg.handle_tool_call("ship", {"order_id": "1"})
            raise AssertionError("expected InvalidActionError")
        except InvalidActionError as e:
            text = _recoverable_error_text(reg, "ship", e)
        assert "not available" in text
        assert "Available actions" in text
        assert "approve" in text

    def test_unknown_tool_error_text_inlines_actions(self):
        reg = Registry(_machine())
        reg.handle_tool_call("query", {"order_id": "1"})
        try:
            reg.handle_tool_call("nope", {})
            raise AssertionError("expected an error")
        except (InvalidActionError, NoHandlerError) as e:
            text = _recoverable_error_text(reg, "nope", e)
        assert "Available actions" in text
        assert "approve" in text

    def test_get_current_actions_empty_before_gateway(self):
        reg = Registry(_machine())
        assert reg.get_current_actions() == []
        reg.handle_tool_call("query", {"order_id": "1"})
        names = [a.name for a in reg.get_current_actions()]
        assert names == ["approve"]

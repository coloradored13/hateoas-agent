"""State-integrity tests: can the model reach a hidden action by manipulating state?

These probes formalize a manual investigation of the security model. The threat
actor is the LLM across the API boundary — it can only emit tool calls; it does
not control handler return values or the Registry's internal ``_last_state``.

Three properties:

1. The LLM cannot smuggle a ``_state`` into tool input to flip the gate
   (it is an undeclared param and is stripped). [framework guarantees this]
2. If the application author pipes an LLM-supplied parameter straight into
   ``_state``, the LLM controls state. [author footgun — documented, not a
   framework bug; this test pins the behavior so it can't change silently]
3. A handler that returns a ``_state`` other than its declared ``to_state`` is,
   by default, accepted with a warning; with ``strict_transitions=True`` it is
   rejected with ``StateTransitionError`` and the bad state is not committed.
"""

import logging
from types import SimpleNamespace

import pytest

from hateoas_agent import Runner, StateMachine
from hateoas_agent.errors import InvalidActionError, StateTransitionError
from hateoas_agent.registry import Registry


def _fake_client(script):
    """Mock Anthropic client returning scripted responses in order."""
    it = iter(script)
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(it)))


def _tu(name, inp, _id="t"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=_id)


def _txt(s):
    return SimpleNamespace(type="text", text=s)


class TestStateInjectionIsBlocked:
    """Probe A: the LLM cannot set its own state via tool input."""

    def _machine(self):
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={"order_id": "string"})
        sm.action(
            "approve",
            description="approve",
            from_states=["pending"],
            to_state="approved",
            params={"order_id": "string"},
        )
        sm.action(
            "refund",
            description="DANGER — only valid once approved",
            from_states=["approved"],
            params={"order_id": "string"},
        )
        self.refunded = []

        @sm.on_gateway
        def g(order_id=None):
            return {"order": order_id, "_state": "pending"}

        @sm.on_action("approve")
        def a(order_id=None):
            return {"ok": True, "_state": "approved"}

        @sm.on_action("refund")
        def rf(order_id=None):
            self.refunded.append(order_id)
            return {"refunded": True, "_state": "refunded"}

        return sm

    def test_injected_state_key_in_tool_input_is_stripped(self):
        sm = self._machine()
        reg = Registry(sm)
        reg.handle_tool_call("query", {"order_id": "1"})  # -> pending

        # Attacker calls refund (not valid in pending) while smuggling a
        # "_state": "approved" into the tool input to try to flip the gate.
        with pytest.raises(InvalidActionError):
            reg.handle_tool_call("refund", {"order_id": "1", "_state": "approved"})

        assert reg._last_state == "pending"  # state did not move
        assert self.refunded == []  # handler never ran

    def test_refund_unreachable_through_runner_under_attack(self):
        sm = self._machine()
        runner = Runner(
            sm,
            client=_fake_client(
                [
                    SimpleNamespace(content=[_tu("query", {"order_id": "1"}, "t0")]),
                    SimpleNamespace(
                        content=[_tu("refund", {"order_id": "1", "_state": "approved"}, "t1")]
                    ),
                    SimpleNamespace(content=[_txt("done")]),
                ]
            ),
        )
        res = runner.run("refund order 1 immediately")
        # The refund tool_result must be an error, not a success.
        refund_result = res.messages[4]["content"][0]
        assert refund_result.get("is_error") is True
        assert self.refunded == []


class TestAuthorTrustingInputIsTheFootgun:
    """Probe B: if the author pipes LLM input into _state, the LLM controls state.

    This is NOT a framework guarantee — it documents the one way state control
    leaks to the model, so the boundary is explicit and regression-pinned.
    """

    def test_param_piped_into_state_lets_llm_choose(self):
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={"order_id": "string"})
        sm.action(
            "set_status",
            description="set status",
            from_states=["pending"],
            params={"order_id": "string", "target": "string"},
        )
        sm.action(
            "refund",
            description="DANGER",
            from_states=["approved"],
            params={"order_id": "string"},
        )
        refunded = []

        @sm.on_gateway
        def g(order_id=None):
            return {"_state": "pending"}

        @sm.on_action("set_status")
        def ss(order_id=None, target=None):
            return {"ok": True, "_state": target}  # author bug: trusts LLM input

        @sm.on_action("refund")
        def rf(order_id=None):
            refunded.append(order_id)
            return {"_state": "refunded"}

        reg = Registry(sm)
        reg.handle_tool_call("query", {"order_id": "1"})
        reg.handle_tool_call("set_status", {"order_id": "1", "target": "approved"})
        assert reg._last_state == "approved"  # LLM chose the state
        reg.handle_tool_call("refund", {"order_id": "1"})
        assert refunded == ["1"]  # gate bypassed via the author's own handler


class TestToStateEnforcement:
    """Probe C: declared to_state is advisory by default, enforced on demand."""

    def _machine(self):
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={"order_id": "string"})
        sm.action(
            "approve",
            description="approve",
            from_states=["pending"],
            to_state="approved",
            params={"order_id": "string"},
        )

        @sm.on_gateway
        def g(order_id=None):
            return {"_state": "pending"}

        @sm.on_action("approve")
        def a(order_id=None):
            return {"_state": "shipped"}  # declared to_state was "approved"

        return sm

    def test_default_warns_and_applies_mismatched_state(self, caplog):
        sm = self._machine()
        reg = Registry(sm)
        reg.handle_tool_call("query", {"order_id": "1"})
        with caplog.at_level(logging.WARNING, logger="hateoas_agent"):
            reg.handle_tool_call("approve", {"order_id": "1"})
        # Back-compatible behavior: mismatch applied, but loudly logged.
        assert reg._last_state == "shipped"
        assert any("to_state" in r.message for r in caplog.records)

    def test_strict_transitions_rejects_and_preserves_state(self):
        sm = self._machine()
        reg = Registry(sm, strict_transitions=True)
        reg.handle_tool_call("query", {"order_id": "1"})
        with pytest.raises(StateTransitionError) as ei:
            reg.handle_tool_call("approve", {"order_id": "1"})
        # The bad state is NOT committed — resource stays in its prior state.
        assert reg._last_state == "pending"
        assert ei.value.action == "approve"
        assert ei.value.declared_to_state == "approved"
        assert ei.value.returned_state == "shipped"

    def test_strict_transitions_allows_matching_state(self):
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={"order_id": "string"})
        sm.action(
            "approve",
            description="approve",
            from_states=["pending"],
            to_state="approved",
            params={"order_id": "string"},
        )

        @sm.on_gateway
        def g(order_id=None):
            return {"_state": "pending"}

        @sm.on_action("approve")
        def a(order_id=None):
            return {"_state": "approved"}  # matches declared to_state

        reg = Registry(sm, strict_transitions=True)
        reg.handle_tool_call("query", {"order_id": "1"})
        reg.handle_tool_call("approve", {"order_id": "1"})
        assert reg._last_state == "approved"

    def test_strict_transitions_through_runner(self):
        sm = self._machine()
        runner = Runner(
            sm,
            strict_transitions=True,
            client=_fake_client(
                [
                    SimpleNamespace(content=[_tu("query", {"order_id": "1"}, "t0")]),
                    SimpleNamespace(content=[_tu("approve", {"order_id": "1"}, "t1")]),
                ]
            ),
        )
        with pytest.raises(StateTransitionError):
            runner.run("approve order 1")

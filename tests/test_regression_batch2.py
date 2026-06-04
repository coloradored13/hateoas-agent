"""Regression tests for the second batch of audit fixes (2026-06-04).

CORR-26  — Orchestrator._execute_phase silently dropped async phase handlers
           (coroutine never awaited; context left stale). Now raises TypeError.
required-not-in-params (recall) — an action/gateway whose ``required`` key is not
           in its declared ``params`` is permanently uncallable (the param filter
           strips it before the required-check). Now caught by validate().
READ-297 — a Claude API failure after SDK retries aborted the run and discarded
           the conversation. Now raised as RunnerAPIError with state attached.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hateoas_agent import Resource, Runner, StateMachine
from hateoas_agent import gateway as gw_decorator
from hateoas_agent.errors import RunnerAPIError
from hateoas_agent.orchestrator import Orchestrator


class TestCorr26AsyncPhaseHandler:
    def test_async_phase_handler_raises_not_silently_dropped(self):
        orch = Orchestrator(name="wf")
        orch.phase("a", description="first")
        orch.phase("b", description="last", terminal=True)

        @orch.on_phase("a")
        async def _a(o, agents, ctx):  # noqa: RUF029 - intentionally async
            return {"converged": True}

        with pytest.raises(TypeError, match="async"):
            orch.start()

    def test_sync_phase_handler_still_works(self):
        orch = Orchestrator(name="wf")
        orch.phase("a", description="first")
        orch.phase("b", description="last", terminal=True)

        @orch.on_phase("a")
        def _a(o, agents, ctx):
            return {"converged": True}

        orch.start()
        assert orch._context.get("converged") is True


class TestRequiredNotInParams:
    def test_statemachine_action_required_not_declared_rejected(self):
        sm = StateMachine("s", gateway_name="start")
        sm.gateway(description="g", params={})
        sm.action(
            "do",
            description="d",
            from_states=["active"],
            params={"a": "string"},
            required=["b"],  # 'b' not in params
        )

        @sm.on_gateway
        def _g(**k):
            return {"_state": "active"}

        @sm.on_action("do")
        def _do(**k):
            return {"_state": "done"}

        with pytest.raises(ValueError, match="required"):
            sm.validate()

    def test_resource_gateway_required_not_declared_rejected(self):
        class Res(Resource):
            name = "r"

            @gw_decorator(
                name="enter",
                description="e",
                params={"a": "string"},
                required=["missing"],
            )
            def enter(self, **k):
                return {"_state": "s"}

        with pytest.raises(ValueError, match="required"):
            Res().validate()

    def test_wellformed_required_passes(self):
        sm = StateMachine("s", gateway_name="start")
        sm.gateway(description="g", params={"id": "string"}, required=["id"])
        sm.action(
            "do",
            description="d",
            from_states=["active"],
            params={"a": "string"},
            required=["a"],
        )

        @sm.on_gateway
        def _g(**k):
            return {"_state": "active"}

        @sm.on_action("do")
        def _do(**k):
            return {"_state": "done"}

        sm.validate()  # must not raise


class TestRead297ApiErrorPreservesState:
    @staticmethod
    def _machine():
        sm = StateMachine("items", gateway_name="list_items")
        sm.gateway(description="List", params={})

        @sm.on_gateway
        def _g(**k):
            return {"items": [], "_state": "active"}

        sm.state("active", actions=[])
        return sm

    def test_api_failure_raises_runner_api_error_with_state(self):
        import anthropic

        class _Boom(anthropic.APIError):
            def __init__(self):
                Exception.__init__(self, "rate limited (retries exhausted)")

        client = MagicMock()
        client.messages.create.side_effect = _Boom()

        runner = Runner(self._machine(), client=client)
        with pytest.raises(RunnerAPIError) as ei:
            runner.run("do the thing")

        err = ei.value
        assert err.messages, "messages should be preserved on the error"
        assert err.messages[-1]["content"] == "do the thing"
        assert isinstance(err.tool_calls, list)
        assert isinstance(err.original, anthropic.APIError)

    def test_recovery_after_transient_failure(self):
        """A failure on turn 1 can be resumed; the second run completes."""
        import anthropic

        class _Boom(anthropic.APIError):
            def __init__(self):
                Exception.__init__(self, "boom")

        ok = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.messages.create.side_effect = [_Boom(), ok]

        runner = Runner(self._machine(), client=client)
        with pytest.raises(RunnerAPIError) as ei:
            runner.run("hi")

        result = runner.run("hi", messages=ei.value.messages[:-1])
        assert result.text == "done"

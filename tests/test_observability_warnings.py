"""Stage A observability tests — verify the six new warning sites fire."""

from __future__ import annotations

import asyncio
import logging
import warnings

import pytest

from hateoas_agent import (
    AgentSlot,
    Orchestrator,
    Registry,
    Resource,
    StateMachine,
    action,
    gateway,
    state,
)
from hateoas_agent.async_runner import AsyncRunner
from hateoas_agent.orchestrator_persistence import (
    load_orchestrator_checkpoint,
    save_orchestrator_checkpoint,
)
from hateoas_agent.registry import _normalize_param_type


class TestDiscoverModeRepeatedWarning:
    def test_action_in_discover_mode_warns(self, caplog):
        sm = StateMachine("t", gateway_name="gw", mode="discover")
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.state_machine"):
            sm.action("approve", description="A", params={})
        assert any(
            "discover mode" in r.message and "approve" in r.message
            for r in caplog.records
        )

    def test_action_in_strict_mode_does_not_warn(self, caplog):
        sm = StateMachine("t", gateway_name="gw")
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.state_machine"):
            sm.action("approve", description="A", from_states=["pending"], params={})
        assert not any("discover mode" in r.message for r in caplog.records)

    def test_state_in_discover_mode_warns(self, caplog):
        sm = StateMachine("t", gateway_name="gw", mode="discover")
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.state_machine"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                sm.state("pending", actions=[])
        assert any("discover mode" in r.message for r in caplog.records)


class TestStateDeprecationWarning:
    def test_state_emits_deprecation_warning(self):
        sm = StateMachine("t", gateway_name="gw")
        with pytest.warns(DeprecationWarning, match="StateMachine.state\\(\\) is deprecated"):
            sm.state("pending", actions=[])

    def test_action_does_not_emit_deprecation_warning(self):
        sm = StateMachine("t", gateway_name="gw")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            sm.action("approve", description="A", from_states=["pending"], params={})


class TestDroppedAgentWarning:
    def _orch(self, agent_names):
        return Orchestrator(
            name="r",
            agents=[AgentSlot(n, role="r") for n in agent_names],
        )

    def test_unknown_agent_in_checkpoint_logs_warning(self, caplog):
        orch_a = self._orch(["alpha", "beta"])
        cp = save_orchestrator_checkpoint(orch_a)

        # Fresh orchestrator missing one of the saved agents
        orch_b = self._orch(["alpha"])
        with caplog.at_level(
            logging.WARNING, logger="hateoas_agent.orchestrator_persistence"
        ):
            load_orchestrator_checkpoint(orch_b, cp)

        assert any(
            "beta" in r.message and "discarded" in r.message
            for r in caplog.records
        )

    def test_matching_agents_emit_no_warning(self, caplog):
        orch_a = self._orch(["alpha", "beta"])
        cp = save_orchestrator_checkpoint(orch_a)
        orch_b = self._orch(["alpha", "beta"])
        with caplog.at_level(
            logging.WARNING, logger="hateoas_agent.orchestrator_persistence"
        ):
            load_orchestrator_checkpoint(orch_b, cp)
        assert not any("discarded" in r.message for r in caplog.records)


class TestAsyncRunnerStallAndMaxIters:
    def test_stall_warns(self, caplog):
        orch = Orchestrator(name="r", agents=[AgentSlot("a", role="r")])
        orch.phase("p1")
        orch.phase("p2", terminal=True)
        # Guard never passes -> stall after p1
        orch.transition("p1", "p2", guard=lambda ctx: False)

        runner = AsyncRunner(orch)
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.async_runner"):
            asyncio.run(runner.run_orchestrated())

        assert any(
            "stalled" in r.message and "p1" in r.message
            for r in caplog.records
        )

    def test_max_iters_warns(self, caplog):
        orch = Orchestrator(name="r", agents=[AgentSlot("a", role="r")])
        orch.phase("p1")
        # Self-loop with always-true guard -> exhausts max_iterations
        orch.transition("p1", "p1", guard=lambda ctx: True, name="loop")

        runner = AsyncRunner(orch, max_iterations=3)
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.async_runner"):
            asyncio.run(runner.run_orchestrated())

        assert any(
            "exhausted max_iterations" in r.message
            for r in caplog.records
        )

    def test_terminal_exit_does_not_warn(self, caplog):
        orch = Orchestrator(name="r", agents=[AgentSlot("a", role="r")])
        orch.phase("p1")
        orch.phase("p2", terminal=True)
        orch.transition("p1", "p2", guard=lambda ctx: True)

        runner = AsyncRunner(orch)
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.async_runner"):
            asyncio.run(runner.run_orchestrated())

        msgs = [r.message for r in caplog.records]
        assert not any("stalled" in m or "exhausted" in m for m in msgs)


class TestGuardExceptionWarning:
    def test_state_machine_guard_exception_warns(self, caplog):
        sm = StateMachine("t", gateway_name="gw")
        sm.gateway(description="GW", params={})
        sm.action(
            "approve",
            description="A",
            from_states=["pending"],
            params={},
            guard=lambda ctx: 1 / 0,
        )

        @sm.on_gateway
        def gw(**kw):
            return {"_state": "pending"}

        @sm.on_action("approve")
        def approve(**kw):
            return {"_state": "approved"}

        reg = Registry(sm)
        reg.handle_tool_call("gw", {})
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.state_machine"):
            reg.get_current_tool_schemas()  # triggers filter_actions

        assert any("Guard for action 'approve'" in r.message for r in caplog.records)

    def test_resource_guard_exception_warns(self, caplog):
        class R(Resource):
            name = "r"

            @gateway(name="gw", description="GW", params={})
            def gw_fn(self, **kw):
                return {"_state": "pending"}

            @action(
                name="approve",
                description="A",
                params={},
                guard=lambda ctx: 1 / 0,
            )
            @state("pending")
            def approve(self, **kw):
                return {"_state": "approved"}

        reg = Registry(R())
        reg.handle_tool_call("gw", {})
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.resource"):
            reg.get_current_tool_schemas()

        assert any("Guard for action 'approve'" in r.message for r in caplog.records)

    def test_orchestrator_guard_exception_warns(self, caplog):
        orch = Orchestrator(name="r", agents=[AgentSlot("a", role="r")])
        orch.phase("p1")
        orch.phase("p2", terminal=True)
        orch.transition("p1", "p2", guard=lambda ctx: 1 / 0)
        orch.start()

        with caplog.at_level(logging.WARNING, logger="hateoas_agent.orchestrator"):
            orch.advance()  # evaluates guard, hits exception path

        assert any("Guard raised an exception" in r.message for r in caplog.records)


class TestRegistryParamTypeFallback:
    def test_unrecognized_type_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            prop = _normalize_param_type("List[string]")
        assert prop["type"] == "string"
        assert any(
            "Unrecognized parameter type" in r.message and "List[string]" in r.message
            for r in caplog.records
        )

    def test_recognized_type_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            prop = _normalize_param_type("string")
        assert prop["type"] == "string"
        assert not any("Unrecognized parameter type" in r.message for r in caplog.records)

    def test_descriptive_string_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
            prop = _normalize_param_type("string (comma-separated)")
        assert prop["type"] == "string"
        assert not any("Unrecognized parameter type" in r.message for r in caplog.records)

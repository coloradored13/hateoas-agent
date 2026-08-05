"""v0.3 gating-hardening probes.

Each test corresponds to a finding from the gating audit (this session's code
audit + the sigma-memory ledger). They lock the hardened behavior so the holes
cannot silently reopen.
"""

import pytest

from hateoas_agent import (
    AgentSlot,
    CompositeRegistry,
    Orchestrator,
    Registry,
    Resource,
    StateMachine,
    action,
    gateway,
    state,
)
from hateoas_agent.conditions import exit_gate_passed
from hateoas_agent.errors import InvalidActionError, StateTransitionError

# --------------------------------------------------------------------------
# S1 — get_handler is no longer a public bypass; invoke() is the gated path
# --------------------------------------------------------------------------


def _orders_machine():
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
        return {"order": order_id, "_state": "pending"}

    @sm.on_action("approve")
    def a(order_id=None):
        return {"ok": True, "_state": "approved"}

    return sm


class TestGetHandlerBypassClosed:
    def test_public_get_handler_is_gone(self):
        sm = _orders_machine()
        assert not hasattr(sm, "get_handler")
        # The private, Registry-internal accessor still exists.
        assert sm._get_handler("approve") is not None

    def test_invoke_enforces_the_gate(self):
        sm = _orders_machine()
        # Calling an action before the gateway establishes state is refused —
        # this is the gate working, exactly what the old get_handler skipped.
        with pytest.raises(InvalidActionError):
            sm.invoke("approve", order_id="1")

        # Through the gateway first, then the action is allowed.
        sm.invoke("query", order_id="1")
        sm.invoke("approve", order_id="1")
        assert sm.registry._last_state == "approved"

    def test_resource_invoke_is_gated(self):
        class Orders(Resource):
            name = "orders"

            @gateway(name="query", description="q", params={"order_id": "string"})
            def query(self, order_id=None):
                return {"_state": "pending"}

            @action(name="approve", description="approve", params={"order_id": "string"})
            @state("pending")
            def approve(self, order_id=None):
                return {"ok": True, "_state": "approved"}

        r = Orders()
        assert not hasattr(r, "get_handler")
        with pytest.raises(InvalidActionError):
            r.invoke("approve", order_id="1")
        r.invoke("query", order_id="1")
        r.invoke("approve", order_id="1")
        assert r.registry._last_state == "approved"


# --------------------------------------------------------------------------
# H — Orchestrator gateway can't teleport phase or inject guard context
# --------------------------------------------------------------------------


def _review_orchestrator(**kwargs):
    o = Orchestrator("review", agents=[AgentSlot("lead", role="r")], **kwargs)
    o.phase("research")
    o.phase("challenge")
    o.phase("synthesis", terminal=True)
    o.transition("research", "challenge")
    o.transition("challenge", "synthesis", guard=exit_gate_passed())
    return o


class TestOrchestratorGatewayLockdown:
    def test_gateway_ignores_requested_phase(self):
        o = _review_orchestrator()
        reg = Registry(o)
        # Attempt to teleport straight into the terminal synthesis phase.
        reg.handle_tool_call("start_workflow", {"phase": "synthesis"})
        assert reg._last_state == "research"  # started at first phase, not synthesis

    def test_gateway_schema_has_no_phase_or_context_by_default(self):
        o = _review_orchestrator()
        reg = Registry(o)
        schema = reg.get_gateway_tool_schema()
        assert schema["input_schema"]["properties"] == {}

    def test_gateway_cannot_inject_guard_context(self):
        o = _review_orchestrator()
        reg = Registry(o)
        # Try to seed exit_gate=PASS via the gateway, then walk the guarded edge.
        reg.handle_tool_call("start_workflow", {"context": '{"exit_gate": "PASS"}'})
        # Advance research -> challenge (unguarded).
        reg.handle_tool_call("advance", {})
        assert reg._last_state == "challenge"
        # The guard must NOT have been satisfied by injected context.
        reg.handle_tool_call("advance", {})
        assert reg._last_state == "challenge"  # still gated, not synthesis

    def test_allowlisted_phase_target_is_honored(self):
        o = _review_orchestrator(allow_phase_targets=["challenge"])
        reg = Registry(o)
        reg.handle_tool_call("start_workflow", {"phase": "challenge"})
        assert reg._last_state == "challenge"
        # A non-allowlisted target still falls back to the default start phase.
        o2 = _review_orchestrator(allow_phase_targets=["challenge"])
        reg2 = Registry(o2)
        reg2.handle_tool_call("start_workflow", {"phase": "synthesis"})
        assert reg2._last_state == "research"

    def test_start_phase_config(self):
        o = _review_orchestrator(start_phase="challenge")
        reg = Registry(o)
        reg.handle_tool_call("start_workflow", {})
        assert reg._last_state == "challenge"


# --------------------------------------------------------------------------
# S2 — strict_transitions defaults on; A3 — known-state enforcement opt-in
# --------------------------------------------------------------------------


class TestStrictDefaults:
    def test_registry_strict_by_default(self):
        assert Registry(_orders_machine())._strict_transitions is True

    def test_enforce_known_states_rejects_teleport(self):
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={})
        sm.action("go", description="go", from_states=["pending"], params={})

        @sm.on_gateway
        def g():
            return {"_state": "pending"}

        @sm.on_action("go")
        def go():
            return {"_state": "somewhere_undeclared"}

        reg = Registry(sm, enforce_known_states=True)
        reg.handle_tool_call("query", {})
        with pytest.raises(StateTransitionError):
            reg.handle_tool_call("go", {})
        assert reg._last_state == "pending"

    def test_known_states_off_allows_implicit_next_state(self):
        # Default (enforce_known_states=False): an action may introduce its next
        # state via the handler return without declaring it.
        sm = StateMachine("orders", gateway_name="query")
        sm.gateway(description="q", params={})
        sm.action("go", description="go", from_states=["pending"], params={})

        @sm.on_gateway
        def g():
            return {"_state": "pending"}

        @sm.on_action("go")
        def go():
            return {"_state": "done"}

        reg = Registry(sm)
        reg.handle_tool_call("query", {})
        reg.handle_tool_call("go", {})
        assert reg._last_state == "done"


# --------------------------------------------------------------------------
# A5 — preserves_state stops accidental state-flip (sigma-mem f89aaf6)
# --------------------------------------------------------------------------


class TestPreservesState:
    def test_universal_action_cannot_flip_state(self):
        sm = StateMachine("mem", gateway_name="recall")
        sm.gateway(description="r", params={})
        sm.action(
            "store",
            description="team write",
            from_states=["team_work"],
            params={},
        )
        # A universal read action that (buggily) returns _state="idle".
        sm.action(
            "get_meta",
            description="read",
            from_states="*",
            params={},
            preserves_state=True,
        )

        @sm.on_gateway
        def g():
            return {"_state": "team_work"}

        @sm.on_action("store")
        def store():
            return {"ok": True, "_state": "team_work"}

        @sm.on_action("get_meta")
        def get_meta():
            return {"meta": 1, "_state": "idle"}  # would deregister team tools

        reg = Registry(sm)
        reg.handle_tool_call("recall", {})
        assert reg._last_state == "team_work"
        reg.handle_tool_call("get_meta", {})
        # preserves_state ignored the stray _state="idle": still in team_work.
        assert reg._last_state == "team_work"
        # So the gated team action is still reachable.
        reg.handle_tool_call("store", {})


# --------------------------------------------------------------------------
# M — CompositeRegistry.get_current_actions no longer crashes recovery
# --------------------------------------------------------------------------


class TestCompositeRecovery:
    def _sm(self, name, gw):
        sm = StateMachine(name, gateway_name=gw)
        sm.gateway(description="q", params={})
        sm.action("act_" + name, description="a", from_states=["s1"], params={})

        @sm.on_gateway
        def g():
            return {"_state": "s1"}

        @sm.on_action("act_" + name)
        def a():
            return {"ok": True, "_state": "s1", "_preserve": True}

        return sm

    def test_get_current_actions_before_and_after_activation(self):
        c = CompositeRegistry([self._sm("orders", "q_orders"), self._sm("inv", "q_inv")])
        # Before any routing: empty, not an AttributeError.
        assert c.get_current_actions() == []
        c.handle_tool_call("q_orders", {})
        names = {a.name for a in c.get_current_actions()}
        assert "act_orders" in names

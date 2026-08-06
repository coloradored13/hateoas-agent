"""Coverage top-up for v0.3 gating branches not exercised elsewhere.

Targets specific uncovered paths surfaced by the coverage report:
- non-strict + enforce_known_states (warn-and-apply, vs. the strict raise)
- the Orchestrator transition handler's own guard re-check (defense-in-depth)
- Orchestrator.invoke() parity with StateMachine/Resource
- Resource guard-exception fail-closed
- discover mode via the HATEOAS_ALLOW_DISCOVER env opt-in
"""

import logging

import pytest

from hateoas_agent import (
    Orchestrator,
    Registry,
    Resource,
    StateMachine,
    action,
    gateway,
    state,
)
from hateoas_agent.conditions import exit_gate_passed

# --------------------------------------------------------------------------
# A3 — enforce_known_states under NON-strict: warn and apply (not raise)
# --------------------------------------------------------------------------


def test_enforce_known_states_nonstrict_warns_and_applies(caplog):
    sm = StateMachine("m", gateway_name="gw")
    sm.gateway(description="q", params={})
    sm.action("go", description="go", from_states=["start"], params={})

    @sm.on_gateway
    def g():
        return {"_state": "start"}

    @sm.on_action("go")
    def go():
        return {"_state": "mystery"}  # not a declared state

    reg = Registry(sm, strict_transitions=False, enforce_known_states=True)
    reg.handle_tool_call("gw", {})
    with caplog.at_level(logging.WARNING, logger="hateoas_agent.registry"):
        reg.handle_tool_call("go", {})

    # Non-strict: the undeclared state is applied, but loudly warned.
    assert reg._last_state == "mystery"
    assert any("not a known state" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Orchestrator transition handler's own guard re-check (defense-in-depth).
# The Registry filters guard-failing transitions before dispatch, so this
# branch is reached by exercising the handler directly.
# --------------------------------------------------------------------------


def test_orchestrator_transition_handler_rejects_when_guard_false():
    o = Orchestrator("wf")
    o.phase("a")
    o.phase("b", terminal=True)
    o.transition("a", "b", guard=exit_gate_passed())
    o.start("a")  # phase a, empty context → guard (exit_gate == PASS) is False

    handler = o._get_handler("a_to_b")
    result = handler()  # guard fails inside the handler

    assert result["_state"] == "a"  # did not transition
    assert "error" in result


# --------------------------------------------------------------------------
# Orchestrator.invoke() parity (gated mixin) — previously only StateMachine
# and Resource exercised invoke().
# --------------------------------------------------------------------------


def test_orchestrator_invoke_is_gated():
    o = Orchestrator("wf")
    o.phase("a")
    o.phase("b", terminal=True)
    o.transition("a", "b")  # unguarded

    o.invoke("start_workflow")  # gateway → phase a
    assert o.registry._last_state == "a"
    o.invoke("a_to_b")  # unguarded transition → b
    assert o.registry._last_state == "b"


# --------------------------------------------------------------------------
# Resource guard that raises → action excluded (fail-closed).
# --------------------------------------------------------------------------


def test_resource_guard_exception_excludes_action():
    def boom(ctx):
        raise RuntimeError("guard blew up")

    class R(Resource):
        name = "r"

        @gateway(name="gw", description="q")
        def gw(self):
            return {"_state": "s1"}

        @action(name="safe", description="s", guard=lambda ctx: True)
        @state("s1")
        def safe(self):
            return {"ok": True, "_state": "s1"}

        @action(name="risky", description="r", guard=boom)
        @state("s1")
        def risky(self):
            return {"ok": True, "_state": "s1"}

    reg = Registry(R())
    reg.handle_tool_call("gw", {})
    available = {a.name for a in reg.get_current_actions()}
    assert "safe" in available
    assert "risky" not in available  # guard raised → fail-closed exclusion


# --------------------------------------------------------------------------
# discover mode via the HATEOAS_ALLOW_DISCOVER env opt-in (vs. the param form).
# --------------------------------------------------------------------------


def test_discover_mode_via_env(monkeypatch):
    monkeypatch.setenv("HATEOAS_ALLOW_DISCOVER", "1")
    # No allow_discover=True param — the env var authorizes it.
    sm = StateMachine("m", gateway_name="gw", mode="discover")
    assert sm.mode == "discover"


def test_discover_mode_without_optin_still_raises(monkeypatch):
    monkeypatch.delenv("HATEOAS_ALLOW_DISCOVER", raising=False)
    with pytest.raises(ValueError):
        StateMachine("m", gateway_name="gw", mode="discover")


# --------------------------------------------------------------------------
# Orchestrator transition handler parses a JSON-string context and merges it
# only AFTER the guard passes (the string-parse branch of _parse_tool_context).
# --------------------------------------------------------------------------


def test_orchestrator_transition_merges_json_string_context():
    o = Orchestrator("wf")
    o.phase("a")
    o.phase("b", terminal=True)
    o.transition("a", "b")  # unguarded → guard passes, context merges
    o.start("a")

    handler = o._get_handler("a_to_b")
    handler(context='{"note": "hi"}')  # JSON string → parsed → merged

    assert o._context.get("note") == "hi"
    assert o._current_phase == "b"

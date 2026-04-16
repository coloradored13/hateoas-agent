"""Tests for composable guard conditions."""

from hateoas_agent.agent_slot import AgentStatus
from hateoas_agent.conditions import (
    all_converged,
    belief_above,
    context_equals,
    context_true,
    exit_gate_passed,
    gap_count_below,
    round_limit,
)
from hateoas_agent.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Individual conditions
# ---------------------------------------------------------------------------


class TestAllConverged:
    def test_all_converged_list(self):
        cond = all_converged()
        assert cond({"agent_statuses": ["converged", "converged"]}) is True

    def test_not_all_converged_list(self):
        cond = all_converged()
        assert cond({"agent_statuses": ["converged", "running"]}) is False

    def test_all_converged_dict(self):
        cond = all_converged()
        ctx = {"agent_statuses": {"a1": "converged", "a2": "converged"}}
        assert cond(ctx) is True

    def test_all_converged_enum(self):
        cond = all_converged()
        ctx = {"agent_statuses": [AgentStatus.CONVERGED, AgentStatus.CONVERGED]}
        assert cond(ctx) is True

    def test_all_converged_mixed_enum(self):
        cond = all_converged()
        ctx = {"agent_statuses": [AgentStatus.CONVERGED, AgentStatus.ERROR]}
        assert cond(ctx) is False

    def test_all_converged_empty(self):
        cond = all_converged()
        assert cond({"agent_statuses": []}) is False
        assert cond({}) is False

    def test_custom_key(self):
        cond = all_converged(key="statuses")
        assert cond({"statuses": ["converged"]}) is True
        assert cond({"agent_statuses": ["converged"]}) is False


class TestBeliefAbove:
    def test_above(self):
        assert belief_above(0.85)({"belief_state": 0.9}) is True

    def test_below(self):
        assert belief_above(0.85)({"belief_state": 0.5}) is False

    def test_equal_is_false(self):
        assert belief_above(0.85)({"belief_state": 0.85}) is False

    def test_missing_key(self):
        assert belief_above(0.85)({}) is False

    def test_custom_key(self):
        assert belief_above(0.5, key="confidence")({"confidence": 0.7}) is True


class TestExitGatePassed:
    def test_pass(self):
        assert exit_gate_passed()({"exit_gate": "PASS"}) is True

    def test_fail(self):
        assert exit_gate_passed()({"exit_gate": "FAIL"}) is False

    def test_missing(self):
        assert exit_gate_passed()({}) is False

    def test_custom_key(self):
        assert exit_gate_passed(key="da_verdict")({"da_verdict": "PASS"}) is True


class TestGapCountBelow:
    def test_below(self):
        assert gap_count_below(3)({"gap_count": 1}) is True

    def test_above(self):
        assert gap_count_below(3)({"gap_count": 5}) is False

    def test_equal_is_false(self):
        assert gap_count_below(3)({"gap_count": 3}) is False

    def test_missing_defaults_zero(self):
        assert gap_count_below(1)({}) is True


class TestRoundLimit:
    def test_below(self):
        assert round_limit(5)({"round": 3}) is True

    def test_at_limit(self):
        assert round_limit(5)({"round": 5}) is False

    def test_above(self):
        assert round_limit(5)({"round": 7}) is False

    def test_missing_defaults_zero(self):
        assert round_limit(1)({}) is True


class TestContextEquals:
    def test_match(self):
        assert context_equals("status", "ready")({"status": "ready"}) is True

    def test_no_match(self):
        assert context_equals("status", "ready")({"status": "pending"}) is False

    def test_missing(self):
        assert context_equals("status", "ready")({}) is False

    def test_none_value(self):
        assert context_equals("key", None)({"key": None}) is True


class TestContextTrue:
    def test_truthy(self):
        assert context_true("flag")({"flag": True}) is True
        assert context_true("flag")({"flag": 1}) is True
        assert context_true("flag")({"flag": "yes"}) is True

    def test_falsy(self):
        assert context_true("flag")({"flag": False}) is False
        assert context_true("flag")({"flag": 0}) is False
        assert context_true("flag")({"flag": ""}) is False

    def test_missing(self):
        assert context_true("flag")({}) is False


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_and(self):
        cond = belief_above(0.8) & exit_gate_passed()
        assert cond({"belief_state": 0.9, "exit_gate": "PASS"}) is True
        assert cond({"belief_state": 0.9, "exit_gate": "FAIL"}) is False
        assert cond({"belief_state": 0.5, "exit_gate": "PASS"}) is False

    def test_or(self):
        cond = exit_gate_passed() | gap_count_below(1)
        assert cond({"exit_gate": "PASS", "gap_count": 5}) is True
        assert cond({"exit_gate": "FAIL", "gap_count": 0}) is True
        assert cond({"exit_gate": "FAIL", "gap_count": 5}) is False

    def test_not(self):
        cond = ~exit_gate_passed()
        assert cond({"exit_gate": "FAIL"}) is True
        assert cond({"exit_gate": "PASS"}) is False

    def test_complex_expression(self):
        """(all_converged & belief_above(0.85)) | exit_gate_passed"""
        cond = (all_converged() & belief_above(0.85)) | exit_gate_passed()

        # All converged + high belief -> True
        assert (
            cond(
                {
                    "agent_statuses": ["converged", "converged"],
                    "belief_state": 0.9,
                }
            )
            is True
        )

        # Exit gate passed alone -> True
        assert cond({"exit_gate": "PASS"}) is True

        # Neither -> False
        assert (
            cond(
                {
                    "agent_statuses": ["converged", "running"],
                    "belief_state": 0.5,
                    "exit_gate": "FAIL",
                }
            )
            is False
        )

    def test_triple_and(self):
        cond = all_converged() & belief_above(0.85) & exit_gate_passed()
        ctx = {
            "agent_statuses": ["converged"],
            "belief_state": 0.9,
            "exit_gate": "PASS",
        }
        assert cond(ctx) is True
        ctx["exit_gate"] = "FAIL"
        assert cond(ctx) is False

    def test_description(self):
        cond = belief_above(0.85) & exit_gate_passed()
        assert "belief_above" in cond.description
        assert "exit_gate_passed" in cond.description

    def test_not_description(self):
        cond = ~exit_gate_passed()
        assert "~" in cond.description

    def test_condition_repr(self):
        cond = belief_above(0.5)
        assert "belief_above" in repr(cond)


# ---------------------------------------------------------------------------
# Integration with Orchestrator
# ---------------------------------------------------------------------------


class TestConditionsWithOrchestrator:
    def test_condition_as_guard(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b", guard=belief_above(0.8))

        orch.start(context={"belief_state": 0.5})
        state = orch.advance()
        assert state.current_phase == "a"  # guard fails

        state = orch.advance(context={"belief_state": 0.9})
        assert state.current_phase == "b"  # guard passes

    def test_composed_guard(self):
        orch = Orchestrator(name="test")
        orch.phase("challenge")
        orch.phase("synthesis", terminal=True)
        orch.transition(
            "challenge",
            "synthesis",
            guard=all_converged() & belief_above(0.85) & exit_gate_passed(),
        )

        orch.start(
            "challenge",
            context={
                "agent_statuses": ["converged", "converged"],
                "belief_state": 0.9,
                "exit_gate": "PASS",
            },
        )
        state = orch.advance()
        assert state.current_phase == "synthesis"

    def test_composed_guard_fails(self):
        orch = Orchestrator(name="test")
        orch.phase("challenge")
        orch.phase("synthesis", terminal=True)
        orch.transition(
            "challenge",
            "synthesis",
            guard=all_converged() & belief_above(0.85) & exit_gate_passed(),
        )
        orch.transition(
            "challenge",
            "challenge",
            guard=~exit_gate_passed(),
        )

        orch.start(
            "challenge",
            context={
                "agent_statuses": ["converged"],
                "belief_state": 0.9,
                "exit_gate": "FAIL",
            },
        )
        state = orch.advance()
        assert state.current_phase == "challenge"  # self-loop via ~exit_gate_passed

    def test_round_limit_bounds_loops(self):
        orch = Orchestrator(name="test")
        orch.phase("loop")
        orch.phase("done", terminal=True)
        orch.transition("loop", "loop", guard=round_limit(3))
        orch.transition("loop", "done")  # fallback unguarded

        @orch.on_phase("loop")
        def handle(orchestrator, agents, context):
            return {"round": context.get("round", 0) + 1}

        orch.start()
        # round starts at 0 from context default
        # loop handler sets round=1, then guard checks round<3 -> True (loop)
        # loop handler sets round=2, guard checks round<3 -> True (loop)
        # loop handler sets round=3, guard checks round<3 -> False, fallback -> done
        state = orch.advance()  # round=1, loops
        assert state.current_phase == "loop"
        state = orch.advance()  # round=2, loops
        assert state.current_phase == "loop"
        state = orch.advance()  # round=3, falls through to done
        assert state.current_phase == "done"

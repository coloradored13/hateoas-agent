"""Composable guard condition factories for orchestrator transitions.

Guard functions receive a context dict and return True/False. They can
be combined with ``&`` (and), ``|`` (or), and ``~`` (not) operators.

Usage::

    from hateoas_agent.conditions import all_converged, belief_above, exit_gate_passed

    review.transition("challenge", "synthesis",
        guard=all_converged() & belief_above(0.85) & exit_gate_passed())
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class Condition:
    """A composable guard condition.

    Wraps a ``Callable[[dict], bool]`` and supports ``&``, ``|``, ``~``
    operators for building complex guard expressions.
    """

    def __init__(self, fn: Callable[[Dict[str, Any]], bool], description: str = ""):
        self._fn = fn
        self.description = description

    def __call__(self, ctx: Dict[str, Any]) -> bool:
        return self._fn(ctx)

    def __and__(self, other: Condition) -> Condition:
        return Condition(
            lambda ctx: self(ctx) and other(ctx),
            f"({self.description} & {other.description})",
        )

    def __or__(self, other: Condition) -> Condition:
        return Condition(
            lambda ctx: self(ctx) or other(ctx),
            f"({self.description} | {other.description})",
        )

    def __invert__(self) -> Condition:
        return Condition(
            lambda ctx: not self(ctx),
            f"~{self.description}",
        )

    def __repr__(self) -> str:
        return f"Condition({self.description!r})"


def all_converged(key: str = "agent_statuses") -> Condition:
    """All agents have status 'converged'.

    Checks ``ctx[key]`` for a list/dict of statuses. All must be
    ``"converged"`` (string) or ``AgentStatus.CONVERGED``.
    """

    def _check(ctx: Dict[str, Any]) -> bool:
        statuses = ctx.get(key, [])
        if isinstance(statuses, dict):
            statuses = list(statuses.values())
        if not statuses:
            return False
        for s in statuses:
            val = s.value if hasattr(s, "value") else s
            if val != "converged":
                return False
        return True

    return Condition(_check, f"all_converged({key!r})")


def belief_above(threshold: float, key: str = "belief_state") -> Condition:
    """Bayesian belief state exceeds threshold.

    Checks ``ctx[key] > threshold``.
    """

    def _check(ctx: Dict[str, Any]) -> bool:
        return ctx.get(key, 0) > threshold

    return Condition(_check, f"belief_above({threshold})")


def exit_gate_passed(key: str = "exit_gate") -> Condition:
    """DA exit-gate verdict is PASS.

    Checks ``ctx[key] == "PASS"``.
    """

    def _check(ctx: Dict[str, Any]) -> bool:
        return ctx.get(key) == "PASS"

    return Condition(_check, f"exit_gate_passed({key!r})")


def gap_count_below(n: int, key: str = "gap_count") -> Condition:
    """Unresolved gaps are below threshold.

    Checks ``ctx[key] < n``.
    """

    def _check(ctx: Dict[str, Any]) -> bool:
        return ctx.get(key, 0) < n

    return Condition(_check, f"gap_count_below({n})")


def round_limit(n: int, key: str = "round") -> Condition:
    """Current round is below limit.

    Checks ``ctx[key] < n``. Useful for bounding self-loop transitions.
    """

    def _check(ctx: Dict[str, Any]) -> bool:
        return ctx.get(key, 0) < n

    return Condition(_check, f"round_limit({n})")


def context_equals(key: str, value: Any) -> Condition:
    """Context key equals a specific value."""

    def _check(ctx: Dict[str, Any]) -> bool:
        return ctx.get(key) == value

    return Condition(_check, f"context_equals({key!r}, {value!r})")


def context_true(key: str) -> Condition:
    """Context key is truthy."""

    def _check(ctx: Dict[str, Any]) -> bool:
        return bool(ctx.get(key))

    return Condition(_check, f"context_true({key!r})")

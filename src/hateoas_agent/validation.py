"""Server-side state validation for dynamic actions."""

from __future__ import annotations

from typing import Dict, List

from .errors import InvalidActionError
from .types import ActionDef


def validate_required_params(
    name: str,
    params: Dict[str, str],
    required: List[str],
    label: str,
) -> None:
    """Ensure every required param is a declared param.

    A ``required`` key that is not in ``params`` is a silent misconfiguration:
    the param filter strips undeclared keys before the required-check runs, so
    such an action can never satisfy its own requirement and is permanently
    uncallable. Catch it at validation time instead.

    Raises:
        ValueError: if any required key is not declared in ``params``.
    """
    undeclared = [r for r in required if r not in (params or {})]
    if undeclared:
        raise ValueError(
            f"{label} '{name}' marks {undeclared} as required but does not "
            f"declare {'them' if len(undeclared) > 1 else 'it'} in params "
            f"(declared: {sorted(params or {})}). Required params must be a "
            f"subset of declared params, or the action can never be called."
        )


def validate_action(
    action_name: str,
    state: str,
    available_actions: List[ActionDef],
) -> ActionDef:
    """Validate that an action is allowed for the current state.

    Returns the matching ActionDef if valid.
    Raises InvalidActionError if not.
    """
    for action in available_actions:
        if action.name == action_name:
            return action

    valid_names = [a.name for a in available_actions]
    raise InvalidActionError(action_name, state, valid_names)

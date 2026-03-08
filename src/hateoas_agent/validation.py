"""Server-side state validation for dynamic actions."""

from __future__ import annotations

from typing import List

from .errors import InvalidActionError
from .types import ActionDef


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

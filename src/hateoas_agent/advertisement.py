"""Format action advertisements in tool results."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import ActionDef


def format_result_with_actions(
    result: Dict[str, Any],
    actions: List[ActionDef],
) -> str:
    """Format a tool result with HATEOAS-style action advertisements.

    Produces output like::

        {"order": {"id": "4521", "status": "pending"}}

        ---
        Available actions for this resource:
          - approve_order: Approve for fulfillment | params: {"order_id": "string"}
          - cancel_order: Cancel this order | params: {"order_id": "string", "reason": "string"}
    """
    output = json.dumps(result, indent=2, default=str)
    if actions:
        output += "\n\n---\nAvailable actions for this resource:\n"
        for action in actions:
            line = f"  - {action.name}: {action.description} | params: {json.dumps(action.params)}"
            if action.required:
                line += f" | required: {json.dumps(action.required)}"
            output += line + "\n"
    return output


def format_error_with_actions(
    error_message: str,
    actions: List[ActionDef],
) -> str:
    """Format an error result with available actions."""
    output = json.dumps({"error": error_message}, indent=2)
    if actions:
        output += "\n\n---\nAvailable actions for this resource:\n"
        for action in actions:
            line = f"  - {action.name}: {action.description} | params: {json.dumps(action.params)}"
            if action.required:
                line += f" | required: {json.dumps(action.required)}"
            output += line + "\n"
    return output

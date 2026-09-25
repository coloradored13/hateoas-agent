"""Generate Mermaid state diagrams from HATEOAS definitions and discovery reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple

if TYPE_CHECKING:
    from .state_machine import StateMachine
    from .types import DiscoveryReport


def state_machine_to_mermaid(sm: "StateMachine") -> str:
    """Generate a Mermaid stateDiagram-v2 from a StateMachine's definitions.

    Uses action-centric metadata (``from_states`` / ``to_state``) for
    transition arrows.  Universal actions (``from_states="*"``) are
    rendered as a note.  State-centric-only actions without ``to_state``
    are omitted from arrows but their states still appear.
    """
    lines = ["stateDiagram-v2"]
    transitions: List[Tuple[str, str, str]] = []  # (from, to, action)
    all_states: Set[str] = set()
    universal_actions: List[str] = []
    first_from: str | None = None

    # Collect action-centric transitions
    for action_name in sm._action_defs:
        from_states = sm._action_from_states.get(action_name)
        to_state = sm._action_to_states.get(action_name)

        if from_states == "*":
            if to_state:
                universal_actions.append(f"{action_name} → {to_state}")
            else:
                universal_actions.append(action_name)
            continue

        if isinstance(from_states, list):
            for fs in from_states:
                all_states.add(fs)
                if first_from is None:
                    first_from = fs
                if to_state:
                    all_states.add(to_state)
                    transitions.append((fs, to_state, action_name))
                else:
                    pass  # no to_state, just register the from state

    # Collect state-centric state names
    for state_name in sm._states:
        all_states.add(state_name)

    if not all_states and not transitions:
        return "stateDiagram-v2"

    # Determine initial state
    if first_from:
        lines.append(f"    [*] --> {first_from}")

    # Add transition arrows
    for from_s, to_s, action in transitions:
        lines.append(f"    {from_s} --> {to_s} : {action}")

    # Add note for universal actions
    if universal_actions:
        # Attach note to first state or [*]
        anchor = first_from or (sorted(all_states)[0] if all_states else None)
        if anchor:
            actions_str = ", ".join(universal_actions)
            lines.append(f"    note right of {anchor} : {actions_str} (all states)")

    return "\n".join(lines)


def discovery_report_to_mermaid(report: "DiscoveryReport") -> str:
    """Generate a Mermaid stateDiagram-v2 from observed transitions."""
    if not report.transitions:
        return "stateDiagram-v2"

    lines = ["stateDiagram-v2"]

    # Deduplicate transitions
    seen: Set[Tuple[str, str, str]] = set()
    first_state = report.transitions[0].state_before

    lines.append(f"    [*] --> {first_state}")

    for t in report.transitions:
        key = (t.state_before, t.state_after, t.action)
        if key not in seen:
            seen.add(key)
            lines.append(f"    {t.state_before} --> {t.state_after} : {t.action}")

    return "\n".join(lines)

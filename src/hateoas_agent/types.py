"""Core data types for hateoas-agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ActionDef:
    """Definition of a dynamically discoverable action."""

    name: str
    description: str
    params: Dict[str, str] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    handler: Optional[Callable] = field(default=None, repr=False)


@dataclass
class StateDef:
    """Definition of a state and its available actions."""

    name: str
    actions: List[ActionDef] = field(default_factory=list)


@dataclass
class GatewayDef:
    """Definition of the gateway (entry-point) tool."""

    name: str
    description: str
    params: Dict[str, str] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    handler: Optional[Callable] = field(default=None, repr=False)


@dataclass
class ActionResult:
    """Result from executing an action, including state for next actions."""

    data: Dict[str, Any]
    state: Optional[str] = None
    actions: List[ActionDef] = field(default_factory=list)


@dataclass
class TransitionRecord:
    """A single observed state transition."""

    state_before: str
    action: str
    state_after: str
    timestamp: float


@dataclass
class DiscoveryReport:
    """Suggested state machine definition from observed transitions."""

    transitions: List[TransitionRecord] = field(default_factory=list)

    def to_state_map(self) -> Dict[str, List[str]]:
        """Return ``{state: [action_names]}`` derived from observed transitions."""
        state_map: Dict[str, set] = {}
        for t in self.transitions:
            state_map.setdefault(t.state_before, set()).add(t.action)
        return {s: sorted(actions) for s, actions in sorted(state_map.items())}

    def to_action_map(self) -> Dict[str, Dict[str, Union[List[str], Optional[str]]]]:
        """Return ``{action: {"from_states": [...], "to_state": ...}}``."""
        from_map: Dict[str, set] = {}
        to_map: Dict[str, set] = {}
        for t in self.transitions:
            from_map.setdefault(t.action, set()).add(t.state_before)
            to_map.setdefault(t.action, set()).add(t.state_after)
        result: Dict[str, Dict[str, Any]] = {}
        for action_name in sorted(from_map):
            from_states = sorted(from_map[action_name])
            to_states = sorted(to_map.get(action_name, set()))
            result[action_name] = {
                "from_states": from_states,
                "to_state": to_states[0] if len(to_states) == 1 else None,
            }
        return result

    def to_python(self, var_name: str = "sm") -> str:
        """Generate Python ``.action()`` calls from observed transitions."""
        action_map = self.to_action_map()
        lines = []
        for action_name, meta in action_map.items():
            from_states = meta["from_states"]
            to_state = meta["to_state"]
            parts = [
                f'{var_name}.action("{action_name}"',
                '    description="...",',
                f"    from_states={from_states!r},",
            ]
            if to_state:
                parts.append(f'    to_state="{to_state}",')
            parts.append(")")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate a Mermaid stateDiagram-v2 from observed transitions."""
        from .visualization import discovery_report_to_mermaid

        return discovery_report_to_mermaid(self)

    def to_json(self) -> str:
        """Serialize this report to JSON."""
        data = {
            "transitions": [
                {
                    "state_before": t.state_before,
                    "action": t.action,
                    "state_after": t.state_after,
                    "timestamp": t.timestamp,
                }
                for t in self.transitions
            ]
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, s: str) -> DiscoveryReport:
        """Deserialize a DiscoveryReport from JSON."""
        data = json.loads(s)
        transitions = [
            TransitionRecord(
                state_before=t["state_before"],
                action=t["action"],
                state_after=t["state_after"],
                timestamp=t["timestamp"],
            )
            for t in data.get("transitions", [])
        ]
        return cls(transitions=transitions)

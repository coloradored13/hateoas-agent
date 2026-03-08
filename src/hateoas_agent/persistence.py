"""Serialization and checkpoint support for HATEOAS state."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .types import DiscoveryReport, TransitionRecord


@dataclass
class RegistryCheckpoint:
    """Serializable snapshot of a Registry's state."""

    last_state: Optional[str] = None
    last_result: Dict[str, Any] = field(default_factory=dict)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegistryCheckpoint:
        return cls(
            last_state=data.get("last_state"),
            last_result=data.get("last_result", {}),
            transitions=data.get("transitions", []),
            timestamp=data.get("timestamp", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> RegistryCheckpoint:
        return cls.from_dict(json.loads(s))


@dataclass
class RunnerCheckpoint:
    """Serializable snapshot of a Runner's state including conversation history."""

    registry: RegistryCheckpoint = field(default_factory=RegistryCheckpoint)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry": self.registry.to_dict(),
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunnerCheckpoint:
        return cls(
            registry=RegistryCheckpoint.from_dict(data.get("registry", {})),
            messages=data.get("messages", []),
            tool_calls=data.get("tool_calls", []),
            timestamp=data.get("timestamp", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> RunnerCheckpoint:
        return cls.from_dict(json.loads(s))


def save_registry_checkpoint(registry: Any) -> Dict[str, Any]:
    """Create a serializable checkpoint from a Registry instance."""
    transitions = [
        {
            "state_before": t.state_before,
            "action": t.action,
            "state_after": t.state_after,
            "timestamp": t.timestamp,
        }
        for t in registry._transition_log
    ]
    cp = RegistryCheckpoint(
        last_state=registry._last_state,
        last_result=registry._last_result,
        transitions=transitions,
        timestamp=time.time(),
    )
    return cp.to_dict()


def load_registry_checkpoint(registry: Any, data: Dict[str, Any]) -> None:
    """Restore a Registry's state from a checkpoint dict."""
    cp = RegistryCheckpoint.from_dict(data)
    registry._last_state = cp.last_state
    registry._last_result = cp.last_result
    registry._transition_log = [
        TransitionRecord(
            state_before=t["state_before"],
            action=t["action"],
            state_after=t["state_after"],
            timestamp=t["timestamp"],
        )
        for t in cp.transitions
    ]


def save_runner_checkpoint(
    runner: Any,
    messages: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a serializable checkpoint from a Runner, messages, and tool trace.

    Args:
        runner: A Runner instance.
        messages: The conversation messages from ``RunResult.messages``.
        tool_calls: The tool call trace from ``RunResult.tool_calls``.

    Returns:
        Dict suitable for JSON serialization.
    """
    reg_data = save_registry_checkpoint(runner._registry)
    cp = RunnerCheckpoint(
        registry=RegistryCheckpoint.from_dict(reg_data),
        messages=messages,
        tool_calls=tool_calls,
        timestamp=time.time(),
    )
    return cp.to_dict()


def load_runner_checkpoint(
    runner: Any,
    data: Dict[str, Any],
) -> tuple:
    """Restore a Runner's registry state and return saved messages/tool_calls.

    Args:
        runner: A Runner instance whose internal registry will be restored.
        data: Checkpoint dict from ``save_runner_checkpoint``.

    Returns:
        Tuple of ``(messages, tool_calls)`` for resuming the conversation.
    """
    cp = RunnerCheckpoint.from_dict(data)
    load_registry_checkpoint(runner._registry, cp.registry.to_dict())
    return cp.messages, cp.tool_calls

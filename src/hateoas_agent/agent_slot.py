"""Agent slot management for orchestrated workflows."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class AgentStatus(enum.Enum):
    """Lifecycle status of an agent within an orchestrator."""

    IDLE = "idle"
    RUNNING = "running"
    CONVERGED = "converged"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class AgentSlot:
    """A slot for an agent within an orchestrated workflow.

    Agents are managed by the Orchestrator as dataclasses, not as
    independent state machines. Meaningful state transitions happen
    at the workflow level, not per-agent.

    Args:
        name: Unique identifier for this agent.
        role: Human-readable description of the agent's expertise.
        status: Current lifecycle status.
        join_phase: Phase name when this agent joins the workflow.
            If None, agent is active from the first phase.
        metadata: Arbitrary key-value data attached to this agent.
    """

    name: str
    role: str = ""
    status: AgentStatus = AgentStatus.IDLE
    join_phase: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from a single agent execution.

    Args:
        agent_name: Name of the agent that produced this result.
        output: The agent's output data.
        status: Status after execution (CONVERGED or ERROR).
        timestamp: When the result was produced.
        error: Error message if status is ERROR.
    """

    agent_name: str
    output: Any = None
    status: AgentStatus = AgentStatus.CONVERGED
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

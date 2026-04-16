"""Checkpoint and restore for orchestrated workflows.

Follows the same pattern as the existing ``RegistryCheckpoint`` —
JSON-serializable dataclasses with ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .agent_slot import AgentStatus
from .orchestrator import Orchestrator


@dataclass
class OrchestratorCheckpoint:
    """Serializable snapshot of an Orchestrator's runtime state.

    Captures current phase, context, phase history, and per-agent
    states so a workflow can be paused and resumed later.
    """

    name: str = ""
    current_phase: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    phase_history: List[str] = field(default_factory=list)
    agent_states: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrchestratorCheckpoint:
        return cls(
            name=data.get("name", ""),
            current_phase=data.get("current_phase"),
            context=data.get("context", {}),
            phase_history=data.get("phase_history", []),
            agent_states=data.get("agent_states", {}),
            timestamp=data.get("timestamp", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> OrchestratorCheckpoint:
        return cls.from_dict(json.loads(s))


def save_orchestrator_checkpoint(orchestrator: Orchestrator) -> Dict[str, Any]:
    """Create a serializable checkpoint from an Orchestrator instance.

    Captures the orchestrator's name, current phase, context, phase
    history, and each agent's current status.

    Args:
        orchestrator: The orchestrator to snapshot.

    Returns:
        Dict suitable for JSON serialization.
    """
    agent_states = {name: agent.status.value for name, agent in orchestrator._agents.items()}
    cp = OrchestratorCheckpoint(
        name=orchestrator.name,
        current_phase=orchestrator._current_phase,
        context=dict(orchestrator._context),
        phase_history=list(orchestrator._phase_history),
        agent_states=agent_states,
        timestamp=time.time(),
    )
    return cp.to_dict()


def load_orchestrator_checkpoint(
    orchestrator: Orchestrator,
    data: Dict[str, Any],
) -> None:
    """Restore an Orchestrator's runtime state from a checkpoint.

    Restores current phase, context, phase history, and agent statuses.
    Phase definitions, transition definitions, and handlers are NOT
    restored — they must already be defined on the orchestrator.

    Args:
        orchestrator: The orchestrator to restore into.
        data: Checkpoint dict from ``save_orchestrator_checkpoint``.
    """
    cp = OrchestratorCheckpoint.from_dict(data)
    orchestrator._current_phase = cp.current_phase
    orchestrator._context = dict(cp.context)
    orchestrator._phase_history = list(cp.phase_history)

    for agent_name, status_str in cp.agent_states.items():
        if agent_name in orchestrator._agents:
            orchestrator._agents[agent_name].status = AgentStatus(status_str)

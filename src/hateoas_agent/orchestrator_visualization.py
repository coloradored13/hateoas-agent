"""Mermaid diagram generation for orchestrator workflows.

Generates stateDiagram-v2 diagrams showing phases as nodes,
transitions as edges with guard labels, and agent annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


def orchestrator_to_mermaid(orchestrator: "Orchestrator") -> str:
    """Generate a Mermaid stateDiagram-v2 from an Orchestrator.

    Shows phases as states, transitions as edges with guard descriptions,
    terminal phases with ``[*]`` exit arrows, and agent annotations as
    notes.
    """
    lines = ["stateDiagram-v2"]

    if not orchestrator._phases:
        return "stateDiagram-v2"

    phases = list(orchestrator._phases.keys())
    first_phase = phases[0]

    # Initial state arrow
    lines.append(f"    [*] --> {first_phase}")

    # Transition edges
    for trans in orchestrator._transitions:
        label = trans.name
        if trans.guard is not None:
            desc = getattr(trans.guard, "description", "")
            if desc:
                label = desc
        lines.append(f"    {trans.from_phase} --> {trans.to_phase} : {label}")

    # Terminal state arrows
    for phase_name in orchestrator._terminal_phases:
        lines.append(f"    {phase_name} --> [*]")

    # Agent annotations as notes
    for phase_name, phase_def in orchestrator._phases.items():
        agents = orchestrator.get_agents_for_phase(phase_name)
        if agents:
            agent_str = ", ".join(a.name for a in agents)
            lines.append(
                f"    note right of {phase_name} : agents: {agent_str}"
            )

    return "\n".join(lines)

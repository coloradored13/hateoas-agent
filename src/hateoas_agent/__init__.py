"""hateoas-agent: Dynamic tool discovery for AI agents.

Instead of pre-registering hundreds of tools, start with one gateway tool.
Each response advertises what actions are available next based on current state.
"""

__version__ = "0.1.0"

from .advertisement import format_error_with_actions, format_result_with_actions
from .composite import CompositeRegistry, ToolNameConflictError
from .errors import (
    HateoasError,
    InvalidActionError,
    NoGatewayError,
    NoHandlerError,
    PhantomToolError,
    StateNotFoundError,
)
from .persistence import (
    RegistryCheckpoint,
    RunnerCheckpoint,
    load_registry_checkpoint,
    load_runner_checkpoint,
    save_registry_checkpoint,
    save_runner_checkpoint,
)
from .registry import Registry
from .resource import Resource, action, gateway, state
from .runner import Runner, RunResult
from .state_machine import StateMachine
from .types import ActionDef, ActionResult, DiscoveryReport, GatewayDef, StateDef, TransitionRecord
from .validation import validate_action
from .visualization import discovery_report_to_mermaid, state_machine_to_mermaid

__all__ = [
    # Declarative API
    "StateMachine",
    # Handler-based API
    "Resource",
    "gateway",
    "action",
    "state",
    # Core
    "Registry",
    "Runner",
    "RunResult",
    # Multi-resource
    "CompositeRegistry",
    "ToolNameConflictError",
    # Types
    "ActionDef",
    "ActionResult",
    "DiscoveryReport",
    "GatewayDef",
    "StateDef",
    "TransitionRecord",
    # Visualization
    "state_machine_to_mermaid",
    "discovery_report_to_mermaid",
    # Persistence
    "RegistryCheckpoint",
    "RunnerCheckpoint",
    "save_registry_checkpoint",
    "load_registry_checkpoint",
    "save_runner_checkpoint",
    "load_runner_checkpoint",
    # Utilities
    "format_result_with_actions",
    "format_error_with_actions",
    "validate_action",
    # Errors
    "HateoasError",
    "InvalidActionError",
    "NoGatewayError",
    "NoHandlerError",
    "PhantomToolError",
    "StateNotFoundError",
]

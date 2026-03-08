"""Declarative state machine API for defining HATEOAS resources."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .types import ActionDef, GatewayDef, StateDef

logger = logging.getLogger(__name__)


class StateMachine:
    """Declarative API for defining a HATEOAS resource.

    Supports two styles for defining actions:

    **State-centric** (original)::

        orders.state("pending", actions=[
            {"name": "approve_order", "description": "Approve", "params": {"order_id": "string"}},
        ])

    **Action-centric** (new)::

        orders.action("approve_order",
            description="Approve this order",
            from_states=["pending"],
            to_state="approved",
            params={"order_id": "string"},
        )

    Both styles can be mixed. In ``mode="discover"``, all actions are
    available in every state and transitions are logged for later analysis.
    """

    def __init__(
        self, name: str, gateway_name: str = "gateway", *, mode: str = "strict"
    ):
        if mode not in ("strict", "discover"):
            raise ValueError(f"mode must be 'strict' or 'discover', got {mode!r}")
        self.name = name
        self._gateway_name = gateway_name
        self._mode = mode
        self._gateway_def: Optional[GatewayDef] = None
        self._states: Dict[str, StateDef] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._gateway_handler: Optional[Callable] = None

        # Action-centric storage
        self._action_defs: Dict[str, ActionDef] = {}
        self._action_from_states: Dict[str, Union[List[str], str]] = {}
        self._action_to_states: Dict[str, Optional[str]] = {}
        self._action_guards: Dict[str, Callable] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def gateway(
        self,
        description: str,
        params: Optional[Dict[str, str]] = None,
        required: Optional[List[str]] = None,
    ) -> None:
        """Define the gateway tool."""
        self._gateway_def = GatewayDef(
            name=self._gateway_name,
            description=description,
            params=params or {},
            required=required or [],
        )

    def state(
        self,
        name: str,
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Define a state and its available actions (state-centric style)."""
        action_defs = []
        for a in actions or []:
            action_defs.append(
                ActionDef(
                    name=a["name"],
                    description=a.get("description", ""),
                    params=a.get("params", {}),
                    required=a.get("required", []),
                )
            )
        self._states[name] = StateDef(name=name, actions=action_defs)

    def action(
        self,
        name: str,
        *,
        description: str,
        from_states: Optional[Union[List[str], str]] = None,
        to_state: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        required: Optional[List[str]] = None,
        guard: Optional[Callable] = None,
    ) -> None:
        """Define an action with state transition metadata (action-centric style).

        Args:
            name: Unique action name.
            description: Human-readable description.
            from_states: States where this action is available.
                List of state names, ``"*"`` for all states,
                or ``None`` (defaults to ``"*"`` in discover mode,
                raises ``ValueError`` in strict mode).
            to_state: Expected resulting state (advisory, not enforced).
            params: Parameter name to type mapping.
            required: Required parameter names.
            guard: Optional callable receiving the last handler result dict.
                Returns ``True`` to include the action, ``False`` to exclude.
                If the guard raises an exception, the action is excluded
                (fail-closed).
        """
        action_def = ActionDef(
            name=name,
            description=description,
            params=params or {},
            required=required or [],
        )

        # Resolve from_states
        if from_states is None:
            if self._mode == "discover":
                resolved: Union[List[str], str] = "*"
            else:
                raise ValueError(
                    f"from_states is required for action '{name}' in strict mode. "
                    f"Use from_states=['state1', 'state2'] or from_states='*'."
                )
        elif isinstance(from_states, str):
            if from_states != "*":
                raise ValueError(
                    f"from_states string must be '*', got {from_states!r}. "
                    f"Use a list for specific states."
                )
            resolved = "*"
        elif isinstance(from_states, list):
            if not from_states:
                raise ValueError(
                    f"from_states cannot be an empty list for action '{name}'. "
                    f"Use from_states='*' for universal actions."
                )
            resolved = from_states
        else:
            raise ValueError(
                f"from_states must be a list of state names or '*', "
                f"got {type(from_states).__name__}"
            )

        self._action_defs[name] = action_def
        self._action_from_states[name] = resolved
        self._action_to_states[name] = to_state
        if guard is not None:
            self._action_guards[name] = guard

        # Attach handler if already registered
        if name in self._action_handlers:
            action_def.handler = self._action_handlers[name]

    def on_gateway(self, fn: Callable) -> Callable:
        """Decorator to register the gateway handler."""
        self._gateway_handler = fn
        if self._gateway_def:
            self._gateway_def.handler = fn
        return fn

    def on_action(self, action_name: str) -> Callable:
        """Decorator to register a handler for a dynamic action."""

        def decorator(fn: Callable) -> Callable:
            self._action_handlers[action_name] = fn
            # Attach to state-centric ActionDefs
            for state_def in self._states.values():
                for action_def in state_def.actions:
                    if action_def.name == action_name:
                        action_def.handler = fn
            # Attach to action-centric ActionDef
            if action_name in self._action_defs:
                self._action_defs[action_name].handler = fn
            return fn

        return decorator

    # --- Internal API used by Registry/Runner ---

    def get_gateway(self) -> Optional[GatewayDef]:
        """Return the gateway definition with handler attached."""
        if self._gateway_def and self._gateway_handler:
            self._gateway_def.handler = self._gateway_handler
        return self._gateway_def

    def get_actions_for_state(self, state: str) -> List[ActionDef]:
        """Return available actions for a given state (as fresh copies).

        Merges actions from both ``.state()`` and ``.action()`` definitions.
        In discover mode, all action-centric actions are returned regardless
        of state.
        """
        seen_names: set[str] = set()
        result: List[ActionDef] = []

        # Action-centric definitions take priority
        for action_name, action_def in self._action_defs.items():
            from_states = self._action_from_states[action_name]
            include = False
            if self._mode == "discover":
                include = True
            elif from_states == "*":
                include = True
            elif isinstance(from_states, list) and state in from_states:
                include = True

            if include:
                result.append(
                    ActionDef(
                        name=action_def.name,
                        description=action_def.description,
                        params=action_def.params,
                        required=action_def.required,
                        handler=self._action_handlers.get(action_name),
                    )
                )
                seen_names.add(action_name)

        # State-centric definitions fill in the rest
        state_def = self._states.get(state)
        if state_def:
            for ad in state_def.actions:
                if ad.name not in seen_names:
                    result.append(
                        ActionDef(
                            name=ad.name,
                            description=ad.description,
                            params=ad.params,
                            required=ad.required,
                            handler=self._action_handlers.get(ad.name),
                        )
                    )
                    seen_names.add(ad.name)

        return result

    def get_handler(self, action_name: str) -> Optional[Callable]:
        """Return the handler for a given action name."""
        return self._action_handlers.get(action_name)

    def get_all_action_names(self) -> set[str]:
        """Return all registered action names across all states."""
        names = set(self._action_defs.keys())
        for state_def in self._states.values():
            for action_def in state_def.actions:
                names.add(action_def.name)
        return names

    def get_transition_metadata(
        self, action_name: str
    ) -> Optional[Tuple[Union[List[str], str], Optional[str]]]:
        """Return ``(from_states, to_state)`` for an action-centric action."""
        if action_name in self._action_from_states:
            return (
                self._action_from_states[action_name],
                self._action_to_states.get(action_name),
            )
        return None

    def filter_actions(
        self,
        actions: List[ActionDef],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ActionDef]:
        """Evaluate guards and return only actions whose guard passes.

        Args:
            actions: List of ActionDefs to filter.
            context: The last handler result dict (passed to guard callables).

        Returns:
            Filtered list — actions without guards are always included.
            Guard exceptions cause the action to be excluded (fail-closed).
        """
        if not self._action_guards:
            return actions

        result = []
        for action_def in actions:
            guard = self._action_guards.get(action_def.name)
            if guard is None:
                result.append(action_def)
                continue
            try:
                if guard(context or {}):
                    result.append(action_def)
            except Exception:
                logger.debug(
                    "Guard for action '%s' raised an exception; excluding action",
                    action_def.name,
                    exc_info=True,
                )
        return result

    def validate(self) -> None:
        """Check that the state machine is minimally configured for use.

        Raises:
            ValueError: If gateway is not defined or if any action is missing a handler.
        """
        if self._gateway_def is None:
            raise ValueError(
                f"StateMachine '{self.name}' has no gateway defined. "
                f"Call .gateway() before using."
            )
        if self._gateway_handler is None:
            raise ValueError(
                f"StateMachine '{self.name}' has no gateway handler. "
                f"Use @{self.name}.on_gateway to register one."
            )
        missing = []
        for action_name in self._action_defs:
            if action_name not in self._action_handlers:
                missing.append(action_name)
        for state_def in self._states.values():
            for ad in state_def.actions:
                if ad.name not in self._action_handlers:
                    missing.append(ad.name)
        if missing:
            unique_missing = sorted(set(missing))
            raise ValueError(
                f"StateMachine '{self.name}' has actions without handlers: "
                f"{', '.join(unique_missing)}. "
                f"Use @{self.name}.on_action(name) to register handlers."
            )

    def to_mermaid(self) -> str:
        """Generate a Mermaid stateDiagram-v2 from this state machine.

        Uses action-centric metadata for transition arrows.
        """
        from .visualization import state_machine_to_mermaid

        return state_machine_to_mermaid(self)

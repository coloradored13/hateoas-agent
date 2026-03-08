"""Handler-based (decorator) API for defining HATEOAS resources."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from .types import ActionDef, GatewayDef

logger = logging.getLogger(__name__)


def gateway(
    *,
    name: str,
    description: str,
    params: Optional[Dict[str, str]] = None,
    required: Optional[List[str]] = None,
) -> Callable:
    """Decorator to mark a method as the gateway handler."""

    def decorator(fn: Callable) -> Callable:
        fn._hateoas_gateway = GatewayDef(
            name=name,
            description=description,
            params=params or {},
            required=required or [],
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._hateoas_gateway = fn._hateoas_gateway
        return wrapper

    return decorator


def action(
    *,
    name: str,
    description: str,
    params: Optional[Dict[str, str]] = None,
    required: Optional[List[str]] = None,
    guard: Optional[Callable] = None,
) -> Callable:
    """Decorator to mark a method as a dynamic action handler.

    Args:
        guard: Optional callable receiving the last handler result dict.
            Returns ``True`` to include the action, ``False`` to exclude.
    """

    def decorator(fn: Callable) -> Callable:
        fn._hateoas_action = ActionDef(
            name=name,
            description=description,
            params=params or {},
            required=required or [],
        )
        if guard is not None:
            fn._hateoas_guard = guard

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._hateoas_action = fn._hateoas_action
        if hasattr(fn, "_hateoas_guard"):
            wrapper._hateoas_guard = fn._hateoas_guard
        # Preserve state annotations if already applied
        if hasattr(fn, "_hateoas_states"):
            wrapper._hateoas_states = fn._hateoas_states
        return wrapper

    return decorator


def state(*states: str) -> Callable:
    """Decorator to restrict an action to specific states."""

    def decorator(fn: Callable) -> Callable:
        fn._hateoas_states = list(states)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._hateoas_states = fn._hateoas_states
        # Preserve action annotation if already applied
        if hasattr(fn, "_hateoas_action"):
            wrapper._hateoas_action = fn._hateoas_action
        return wrapper

    return decorator


class Resource:
    """Base class for handler-based HATEOAS resources.

    Usage::

        class OrderResource(Resource):
            name = "orders"

            @gateway(name="query_orders", description="Search orders",
                     params={"order_id": "string"})
            def query(self, order_id=None, **kwargs):
                order = self.lookup(order_id)
                return {"order": order, "_state": order["status"]}

            @action(name="approve_order", description="Approve",
                    params={"order_id": "string"})
            @state("pending")
            def approve(self, order_id):
                ...
                return {"success": True, "_state": "approved"}
    """

    name: str = "resource"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Scan for decorated methods and build state/action maps
        cls._gateway_method: Optional[str] = None
        cls._action_methods: Dict[str, str] = {}  # action_name -> method_name
        cls._action_states: Dict[str, List[str]] = {}  # action_name -> [states]
        cls._action_defs: Dict[str, ActionDef] = {}  # action_name -> ActionDef
        cls._action_guards: Dict[str, Callable] = {}  # action_name -> guard fn

        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if attr is None:
                continue
            if hasattr(attr, "_hateoas_gateway"):
                cls._gateway_method = attr_name
            if hasattr(attr, "_hateoas_action"):
                action_def = attr._hateoas_action
                cls._action_methods[action_def.name] = attr_name
                cls._action_defs[action_def.name] = action_def
                if hasattr(attr, "_hateoas_states"):
                    cls._action_states[action_def.name] = attr._hateoas_states
                if hasattr(attr, "_hateoas_guard"):
                    cls._action_guards[action_def.name] = attr._hateoas_guard

    # --- HasHateoas protocol ---

    def get_gateway(self) -> Optional[GatewayDef]:
        if not self._gateway_method:
            return None
        method = getattr(self, self._gateway_method)
        gw_def = method._hateoas_gateway
        gw_def.handler = method
        return gw_def

    def get_actions_for_state(self, state_name: str) -> List[ActionDef]:
        actions = []
        for action_name, action_def in self._action_defs.items():
            states = self._action_states.get(action_name, [])
            if not states or state_name in states:
                # Attach handler
                method_name = self._action_methods[action_name]
                method = getattr(self, method_name)
                ad = ActionDef(
                    name=action_def.name,
                    description=action_def.description,
                    params=action_def.params,
                    handler=method,
                )
                actions.append(ad)
        return actions

    def get_handler(self, action_name: str) -> Optional[Callable]:
        method_name = self._action_methods.get(action_name)
        if not method_name:
            return None
        return getattr(self, method_name)

    def get_all_action_names(self) -> set[str]:
        return set(self._action_methods.keys())

    def validate(self) -> None:
        """Check that the resource is minimally configured for use.

        Raises:
            ValueError: If gateway is not defined or if any action is missing a handler.
        """
        if not self._gateway_method:
            raise ValueError(
                f"Resource '{self.name}' has no gateway defined. "
                f"Use @gateway() to decorate a method."
            )
        missing = []
        for action_name, method_name in self._action_methods.items():
            if not hasattr(self, method_name):
                missing.append(action_name)
        if missing:
            unique_missing = sorted(set(missing))
            raise ValueError(
                f"Resource '{self.name}' has actions without handlers: "
                f"{', '.join(unique_missing)}."
            )

    def filter_actions(
        self,
        actions: List[ActionDef],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ActionDef]:
        """Evaluate guards and return only actions whose guard passes."""
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
                    "Guard for action '%s' raised; excluding action",
                    action_def.name,
                    exc_info=True,
                )
        return result

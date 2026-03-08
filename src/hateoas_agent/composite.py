"""Multi-resource composition for managing multiple state machines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .errors import HateoasError
from .registry import Registry
from .types import DiscoveryReport, TransitionRecord


class ToolNameConflictError(HateoasError):
    """Raised when multiple resources register the same tool name."""

    def __init__(self, tool_name: str, resources: List[str]):
        self.tool_name = tool_name
        self.resources = resources
        super().__init__(
            f"Tool name '{tool_name}' conflicts between resources: "
            f"{', '.join(resources)}"
        )


class CompositeRegistry:
    """Routes tool calls across multiple HATEOAS resources.

    Each resource gets its own ``Registry`` with independent state tracking.
    Tool names are mapped to sub-registries; conflicts are detected at init.

    Usage::

        composite = CompositeRegistry([orders_sm, inventory_sm])
        schemas = composite.get_current_tool_schemas()
        result = composite.handle_tool_call("query_orders", {"id": "123"})
    """

    def __init__(self, resources: List[Any]):
        self._registries: List[Registry] = []
        self._tool_map: Dict[str, Registry] = {}
        self._gateway_names: List[str] = []
        self._last_active: Optional[Registry] = None

        for resource in resources:
            reg = Registry(resource)
            self._registries.append(reg)

            gw_name = reg.gateway_name
            self._gateway_names.append(gw_name)
            self._register_tool(gw_name, reg, resource)

            # Register all action names
            for action_name in resource.get_all_action_names():
                self._register_tool(action_name, reg, resource)

    def _register_tool(self, tool_name: str, registry: Registry, resource: Any) -> None:
        if tool_name in self._tool_map and self._tool_map[tool_name] is not registry:
            existing = self._tool_map[tool_name]
            names = []
            for reg in self._registries:
                if reg is existing or reg is registry:
                    res_name = getattr(reg._resource, "name", str(reg._resource))
                    names.append(res_name)
            raise ToolNameConflictError(tool_name, names)
        self._tool_map[tool_name] = registry

    @property
    def gateway_names(self) -> List[str]:
        """Return all gateway tool names."""
        return list(self._gateway_names)

    @property
    def _last_state(self) -> Optional[str]:
        """Return state from most recently active sub-registry."""
        if self._last_active:
            return self._last_active._last_state
        return None

    def get_current_tool_schemas(self) -> List[Dict[str, Any]]:
        """Merge tool schemas from all sub-registries."""
        schemas: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for reg in self._registries:
            for schema in reg.get_current_tool_schemas():
                if schema["name"] not in seen:
                    schemas.append(schema)
                    seen.add(schema["name"])
        return schemas

    def handle_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Route a tool call to the appropriate sub-registry."""
        registry = self._tool_map.get(tool_name)
        if registry is None:
            # Check if any registry recognizes it
            for reg in self._registries:
                if reg.is_gateway(tool_name) or reg.is_known_action(tool_name):
                    registry = reg
                    break
        if registry is None:
            from .errors import NoHandlerError

            raise NoHandlerError(tool_name)

        result = registry.handle_tool_call(tool_name, tool_input)
        self._last_active = registry
        return result

    def is_gateway(self, tool_name: str) -> bool:
        """Check if tool_name is a gateway in any sub-registry."""
        return any(reg.is_gateway(tool_name) for reg in self._registries)

    def is_known_action(self, tool_name: str) -> bool:
        """Check if tool_name is a known action in any sub-registry."""
        return any(reg.is_known_action(tool_name) for reg in self._registries)

    def get_discovery_report(self) -> DiscoveryReport:
        """Merge discovery reports from all sub-registries, sorted by timestamp."""
        all_transitions: List[TransitionRecord] = []
        for reg in self._registries:
            report = reg.get_discovery_report()
            all_transitions.extend(report.transitions)
        all_transitions.sort(key=lambda t: t.timestamp)
        return DiscoveryReport(transitions=all_transitions)

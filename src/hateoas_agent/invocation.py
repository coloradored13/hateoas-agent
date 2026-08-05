"""Gated single-call invocation for HATEOAS resources.

``GatedInvokeMixin.invoke()`` is the supported alternative to reaching for a
resource's ``_get_handler`` (private precisely because calling a handler
directly skips the state gate). ``invoke()`` routes every call through an
internal :class:`~hateoas_agent.registry.Registry`, so the gateway establishes
state that subsequent action calls are validated against — within one process.

Rationale: the first integrator (sigma-mem's ``recall.py``) reached for the
public ``get_handler`` and silently lost enforcement. Making the handler
private removes that easy wrong path; ``invoke()`` gives an easy *right* one.
"""

from __future__ import annotations

from typing import Any, Optional


class GatedInvokeMixin:
    """Adds a gated ``invoke()`` backed by a per-instance Registry.

    The Registry is created lazily on first use and reused across calls, so a
    ``invoke("gateway", ...)`` sets the state that a following
    ``invoke("some_action", ...)`` is gated against. Calling a non-gateway
    action before any gateway call is refused — that is the gate working, not
    a bug.
    """

    _invoke_registry: Optional[Any] = None

    @property
    def registry(self) -> Any:
        """The internal Registry backing ``invoke()`` (created on first use).

        Exposed so callers can inspect current state / advertised actions
        (``registry.get_current_actions()``) or drive the resource with the
        same state that ``invoke()`` uses.
        """
        from .registry import Registry

        if self._invoke_registry is None:
            self._invoke_registry = Registry(self)
        return self._invoke_registry

    def invoke(self, tool_name: str, /, **kwargs: Any) -> str:
        """Invoke a tool through the state gate and return the formatted result.

        ``tool_name`` is positional-only so it can't collide with a handler
        parameter also named ``tool_name``. The return value is the same
        advertisement-annotated string an agent would receive from the
        Registry, including the actions valid in the resulting state.
        """
        return self.registry.handle_tool_call(tool_name, kwargs)

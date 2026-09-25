"""Routes tool calls and validates state transitions."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from .advertisement import format_result_with_actions
from .errors import (
    InvalidActionError,
    NoGatewayError,
    NoHandlerError,
    StateTransitionError,
)
from .types import ActionDef, DiscoveryReport, GatewayDef, TransitionRecord
from .validation import validate_action

logger = logging.getLogger(__name__)


@runtime_checkable
class HasHateoas(Protocol):
    """Protocol for objects that provide HATEOAS definitions (StateMachine or Resource).

    Required methods:
        get_gateway, get_actions_for_state, _get_handler, get_all_action_names

    ``_get_handler`` is intentionally private: it returns a raw handler
    callable that, if invoked directly, bypasses the state gate entirely. Only
    the Registry (the gating layer) should call it. Application code must go
    through ``Registry.handle_tool_call`` or the resource's gated ``invoke()``.

    Optional methods (checked via hasattr at runtime):
        validate — startup configuration check
        filter_actions — guard-based action filtering
        get_transition_metadata — declared from/to states for an action
        get_known_states — the set of states the resource recognizes; when
            provided, the Registry refuses to commit a handler-returned
            ``_state`` outside this set (fail-closed against state teleport).
    """

    def get_gateway(self) -> Optional[GatewayDef]: ...
    def get_actions_for_state(self, state: str) -> List[ActionDef]: ...
    def _get_handler(self, action_name: str) -> Optional[Any]: ...
    def get_all_action_names(self) -> set[str]: ...

    # Optional extension points — implementations may omit these.
    # Registry checks via hasattr before calling.
    def validate(self) -> None: ...
    def filter_actions(
        self, actions: List[ActionDef], context: Optional[Dict[str, Any]] = None
    ) -> List[ActionDef]: ...
    def get_transition_metadata(
        self, action_name: str
    ) -> Optional[Tuple[Union[List[str], str], Optional[str]]]: ...
    def get_known_states(self) -> Optional[set[str]]: ...


# Sentinel for "no state returned"
_NO_STATE = object()

STATE_KEY = "_state"


def _extract_state(result: Any) -> Tuple[Any, Any]:
    """Extract _state from a result dict. Returns (cleaned_result, state)."""
    if isinstance(result, dict) and STATE_KEY in result:
        state = result[STATE_KEY]
        if not isinstance(state, str):
            raise TypeError(f"_state must be a string, got {type(state).__name__}: {state!r}")
        cleaned = {k: v for k, v in result.items() if k != STATE_KEY}
        return cleaned, state
    return result, _NO_STATE


def _filter_params(tool_input: Dict[str, Any], declared_params: Dict[str, str]) -> Dict[str, Any]:
    """Filter tool input to only include declared parameter keys.

    When no parameters are declared, returns an empty dict — the action
    accepts no input. This prevents LLM-supplied parameters from leaking
    through to handlers.
    """
    if not declared_params:
        return {}
    return {k: v for k, v in tool_input.items() if k in declared_params}


def _check_required(
    filtered: Dict[str, Any], required: List[str], action_name: str
) -> Optional[str]:
    """Check that all required parameters are present. Returns error message or None."""
    missing = [r for r in required if r not in filtered]
    if missing:
        return f"Missing required parameter(s) for '{action_name}': {', '.join(missing)}"
    return None


_VALID_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def _normalize_param_type(raw_type: str) -> Dict[str, str]:
    """Convert a param type string to a valid JSON Schema property definition.

    Handles descriptive types like ``"string (comma-separated values)"`` by
    extracting the base type and putting the rest in ``description``.
    """
    raw = raw_type.strip()
    base = raw.split("(")[0].split(" ")[0].strip().lower()
    if base not in _VALID_JSON_SCHEMA_TYPES:
        logger.warning(
            "Unrecognized parameter type %r (parsed base %r); falling back to "
            "'string'. Valid JSON Schema types: %s",
            raw_type,
            base,
            sorted(_VALID_JSON_SCHEMA_TYPES),
        )
        base = "string"
    prop: Dict[str, str] = {"type": base}
    # If the original had extra description beyond the type, preserve it
    if raw != base:
        prop["description"] = raw
    return prop


def _action_to_tool_schema(action: "ActionDef") -> Dict[str, Any]:
    """Convert an ActionDef to a Claude API tool schema."""
    properties = {}
    for param_name, param_type in action.params.items():
        properties[param_name] = _normalize_param_type(param_type)
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if action.required:
        schema["required"] = action.required
    return {
        "name": action.name,
        "description": action.description,
        "input_schema": schema,
    }


class Registry:
    """Routes tool calls to the appropriate handler and manages state.

    Handles both gateway and dynamic action calls, validates state
    transitions, and formats results with action advertisements.
    """

    def __init__(
        self,
        resource: HasHateoas,
        *,
        strict_transitions: bool = True,
        enforce_known_states: bool = False,
    ):
        self._resource = resource
        self._last_state: Optional[str] = None
        self._last_result: Dict[str, Any] = {}
        self._transition_log: List[TransitionRecord] = []
        # strict_transitions (default True as of v0.3): a handler that returns a
        # _state other than its action's *declared* to_state raises
        # StateTransitionError and the bad state is NOT committed. When False,
        # the mismatch is only logged and applied (pre-0.3 fail-open behavior).
        self._strict_transitions = strict_transitions
        # enforce_known_states (opt-in, stricter): additionally reject a
        # returned _state that is not in the resource's declared vocabulary
        # (get_known_states). Off by default because a legitimate action may
        # introduce its next state purely via the handler's _state return
        # without declaring a to_state. Turn on for maximum-strictness gates
        # where every state is declared up front.
        self._enforce_known_states = enforce_known_states

    @property
    def gateway_name(self) -> str:
        gw = self._resource.get_gateway()
        if not gw:
            raise NoGatewayError()
        return gw.name

    def get_gateway_tool_schema(self) -> Dict[str, Any]:
        """Return the gateway tool definition for the Claude API tools array."""
        gw = self._resource.get_gateway()
        if not gw:
            raise NoGatewayError()
        properties = {}
        for param_name, param_type in gw.params.items():
            properties[param_name] = _normalize_param_type(param_type)
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if gw.required:
            schema["required"] = gw.required
        return {
            "name": gw.name,
            "description": gw.description,
            "input_schema": schema,
        }

    def _get_filtered_actions(self, state: str) -> List[ActionDef]:
        """Return actions for a state, filtered by guards if available."""
        actions = self._resource.get_actions_for_state(state)
        if hasattr(self._resource, "filter_actions"):
            actions = self._resource.filter_actions(actions, self._last_result)
        return actions

    def get_current_actions(self) -> List[ActionDef]:
        """Return the actions valid in the current state (guard-filtered).

        Empty until the gateway has been called. Used to build LLM-friendly
        error responses that inline the valid next actions so the model can
        recover in the same turn.
        """
        if self._last_state is None:
            return []
        return self._get_filtered_actions(self._last_state)

    def get_current_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the gateway plus all actions in the current state.

        This allows the Runner to dynamically register available tools with the
        Claude API after each state transition, ensuring Claude can only call
        actions that are valid for the current state.
        """
        tools = [self.get_gateway_tool_schema()]
        if self._last_state is not None:
            actions = self._get_filtered_actions(self._last_state)
            for action in actions:
                tools.append(_action_to_tool_schema(action))
        return tools

    def is_gateway(self, tool_name: str) -> bool:
        gw = self._resource.get_gateway()
        return gw is not None and gw.name == tool_name

    def is_known_action(self, tool_name: str) -> bool:
        return tool_name in self._resource.get_all_action_names()

    def handle_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Route a tool call and return formatted result string.

        Handles gateway calls, validates dynamic actions against current
        state, executes handlers, and returns results with action ads.
        """
        if self.is_gateway(tool_name):
            return self._handle_gateway(tool_input)
        else:
            return self._handle_action(tool_name, tool_input)

    def _known_states(self) -> Optional[set[str]]:
        """Return the resource's known-state set, or None if it declares none.

        None means "no constraint" — the resource does not enumerate its
        states, so the Registry can't fail-close a returned state against them.
        """
        getter = getattr(self._resource, "get_known_states", None)
        if getter is None:
            return None
        try:
            states = getter()
        except Exception:  # pragma: no cover - defensive
            logger.warning("get_known_states() raised; skipping known-state check", exc_info=True)
            return None
        return set(states) if states else None

    def _reconcile_returned_state(self, tool_name: str, state: str) -> None:
        """Validate a handler-returned ``_state`` before it is committed.

        Under ``strict_transitions`` (the v0.3 default) a state that violates
        the action's declared ``to_state`` — or that is not one of the
        resource's known states — raises ``StateTransitionError`` so the bad
        state is never committed. Otherwise the mismatch is only logged and the
        caller applies it (pre-0.3 fail-open behavior).
        """
        # 1. Declared to_state reconciliation.
        if hasattr(self._resource, "get_transition_metadata"):
            meta = self._resource.get_transition_metadata(tool_name)
            if meta is not None:
                _, declared_to = meta
                if declared_to is not None and state != declared_to:
                    if self._strict_transitions:
                        raise StateTransitionError(tool_name, declared_to, state)
                    logger.warning(
                        "Action '%s' declared to_state='%s' but handler returned _state='%s'",
                        tool_name,
                        declared_to,
                        state,
                    )
                    return
        # 2. Known-state membership — catches teleport to an undeclared state
        #    even when the action declared no to_state. Opt-in only, since a
        #    legitimate action may introduce its next state via the handler
        #    return without declaring it.
        if not self._enforce_known_states:
            return
        known = self._known_states()
        if known is not None and state not in known:
            if self._strict_transitions:
                raise StateTransitionError(tool_name, f"one of {sorted(known)}", state)
            logger.warning(
                "Action '%s' handler returned _state=%r which is not a known state (%s)",
                tool_name,
                state,
                sorted(known),
            )

    def _handle_gateway(self, tool_input: Dict[str, Any]) -> str:
        gw = self._resource.get_gateway()
        if not gw or not gw.handler:
            raise NoGatewayError()

        filtered = _filter_params(tool_input, gw.params)
        required_err = _check_required(filtered, gw.required, gw.name)
        if required_err:
            return format_result_with_actions({"error": required_err}, [])
        raw_result = gw.handler(**filtered)
        result, state = _extract_state(raw_result)

        if state is not _NO_STATE:
            self._last_state = state
        elif isinstance(result, dict):
            logger.warning(
                "Gateway handler returned a dict without '_state'. "
                "Actions will not be advertised until _state is set. "
                "Add '_state' to your return value, e.g. "
                "return {... , '_state': 'my_state'}"
            )

        if isinstance(result, dict):
            self._last_result = result

        actions = []
        if self._last_state is not None:
            actions = self._get_filtered_actions(self._last_state)

        return format_result_with_actions(
            result if isinstance(result, dict) else {"result": result},
            actions,
        )

    def get_discovery_report(self) -> DiscoveryReport:
        """Return a report of all observed state transitions."""
        return DiscoveryReport(transitions=list(self._transition_log))

    def _handle_action(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        # Reject action calls before any state has been established via gateway
        if self._last_state is None:
            raise InvalidActionError(tool_name, "<no state>", [])

        # Validate against current state (with guard filtering)
        # so the runner can fire on_invalid_action callbacks
        available = self._get_filtered_actions(self._last_state)
        validate_action(tool_name, self._last_state, available)

        handler = self._resource._get_handler(tool_name)
        if not handler:
            raise NoHandlerError(tool_name)

        # Find the ActionDef to get declared params for filtering
        action_def = next((a for a in available if a.name == tool_name), None)
        declared_params = action_def.params if action_def else {}
        required_params = action_def.required if action_def else []
        filtered = _filter_params(tool_input, declared_params)

        required_err = _check_required(filtered, required_params, tool_name)
        if required_err:
            actions = self._get_filtered_actions(self._last_state)
            return format_result_with_actions({"error": required_err}, actions)

        state_before = self._last_state
        raw_result = handler(**filtered)
        result, state = _extract_state(raw_result)

        # A read-only / universal action must not move the gate. If it is
        # declared preserves_state, drop any _state it returned so a stray
        # value can't silently re-gate the session (sigma-mem f89aaf6).
        preserves = bool(action_def and action_def.preserves_state)
        if preserves and state is not _NO_STATE:
            logger.debug(
                "Action '%s' is preserves_state; ignoring returned _state=%r",
                tool_name,
                state,
            )
            state = _NO_STATE

        # Reconcile the returned state BEFORE committing it, so strict mode can
        # reject without ever landing in the unexpected state.
        if state is not _NO_STATE:
            self._reconcile_returned_state(tool_name, state)

        if state is not _NO_STATE:
            self._last_state = state
        elif isinstance(result, dict) and not preserves:
            logger.warning(
                "Action '%s' handler returned a dict without '_state'. "
                "State will remain '%s'. Add '_state' to your return value.",
                tool_name,
                self._last_state,
            )

        if isinstance(result, dict):
            self._last_result = result

        # Log transition for discovery mode
        state_after = self._last_state
        if state_before is not None and state_after is not None:
            self._transition_log.append(
                TransitionRecord(
                    state_before=state_before,
                    action=tool_name,
                    state_after=state_after,
                    timestamp=time.time(),
                )
            )

        actions = []
        if self._last_state is not None:
            actions = self._get_filtered_actions(self._last_state)

        return format_result_with_actions(
            result if isinstance(result, dict) else {"result": result},
            actions,
        )

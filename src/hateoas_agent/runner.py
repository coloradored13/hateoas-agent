"""Agent loop that drives Claude API calls with HATEOAS tool discovery."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from .errors import InvalidActionError, NoHandlerError, PhantomToolError
from .registry import Registry

logger = logging.getLogger(__name__)


class Runner:
    """Runs a HATEOAS agent loop against the Claude API.

    Usage::

        runner = Runner(state_machine, model="claude-sonnet-4-20250514")
        result = runner.run("Look up order 4521 and approve it.")

    Security callbacks::

        runner = Runner(
            state_machine,
            on_phantom_tool=lambda name, inp, state: print(f"PHANTOM: {name}"),
            on_invalid_action=lambda name, inp, state: print(f"INVALID: {name}"),
            strict=True,  # halt on phantom tool calls
        )
    """

    def __init__(
        self,
        resource: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        system: Optional[str] = None,
        max_turns: int = 10,
        max_tokens: int = 1024,
        client: Optional[Any] = None,
        on_tool_call: Optional[Callable] = None,
        on_text: Optional[Callable] = None,
        on_invalid_action: Optional[Callable] = None,
        on_phantom_tool: Optional[Callable] = None,
        on_transition: Optional[Callable] = None,
        strict: bool = False,
    ):
        # Support list of resources (multi-resource composition)
        if isinstance(resource, list):
            from .composite import CompositeRegistry

            for r in resource:
                if hasattr(r, "validate"):
                    r.validate()
            self._registry = CompositeRegistry(resource)
        else:
            if hasattr(resource, "validate"):
                resource.validate()
            self._registry = Registry(resource)
        self._model = model
        self._system = system or self._default_system()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        elif anthropic is not None:
            self._client = anthropic.Anthropic()
        else:
            raise ImportError(
                "The 'anthropic' package is required to use Runner. "
                "Install it with: pip install 'hateoas-agent[anthropic]'"
            )
        self._on_tool_call = on_tool_call
        self._on_text = on_text
        self._on_invalid_action = on_invalid_action
        self._on_phantom_tool = on_phantom_tool
        self._on_transition = on_transition
        self._strict = strict

    def _default_system(self) -> str:
        # Handle multi-resource (CompositeRegistry with gateway_names list)
        if hasattr(self._registry, "gateway_names"):
            gw_names = self._registry.gateway_names
            gw_list = ", ".join(f"'{n}'" for n in gw_names)
            return (
                f"You have gateway tools: {gw_list} to start. After each tool call, "
                f"the system will list the actions available for the current state under "
                f"'Available actions for this resource'.\n\n"
                f"IMPORTANT: Only use actions explicitly listed in the MOST RECENT tool result. "
                f"Actions from previous turns may no longer be valid after a state change. "
                f"Never call a tool name that hasn't been advertised in the latest response."
            )

        gw_name = self._registry.gateway_name
        return (
            f"You have a tool called '{gw_name}' to start. After each tool call, "
            f"the system will list the actions available for the current state under "
            f"'Available actions for this resource'.\n\n"
            f"IMPORTANT: Only use actions explicitly listed in the MOST RECENT tool result. "
            f"Actions from previous turns may no longer be valid after a state change. "
            f"Never call a tool name that hasn't been advertised in the latest response."
        )

    def _is_phantom(self, tool_name: str) -> bool:
        """Check if a tool name is completely unknown (not gateway, not in any state)."""
        if self._registry.is_gateway(tool_name):
            return False
        if self._registry.is_known_action(tool_name):
            return False
        return True

    def run(
        self,
        user_message: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RunResult:
        """Run the agent loop for a user message.

        Args:
            user_message: The user's request.
            messages: Optional existing message history to continue from.

        Returns:
            RunResult with the final response text, messages, and tool call trace.
        """
        msgs = list(messages) if messages else []
        msgs.append({"role": "user", "content": user_message})

        tools = self._registry.get_current_tool_schemas()
        tool_trace: List[Dict[str, Any]] = []

        for _ in range(self._max_turns):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                tools=tools,
                system=self._system,
                messages=msgs,
            )

            msgs.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            for b in response.content:
                if b.type == "text" and b.text.strip() and self._on_text:
                    self._on_text(b.text)

            if not tool_uses:
                final_text = ""
                for b in response.content:
                    if b.type == "text":
                        final_text += b.text
                return RunResult(
                    text=final_text.strip(),
                    messages=msgs,
                    tool_calls=tool_trace,
                )

            tool_results = []
            for tu in tool_uses:
                if self._on_tool_call:
                    self._on_tool_call(tu.name, tu.input)

                tool_trace.append({"tool": tu.name, "input": tu.input})

                # Check for phantom tools (completely unknown tool names)
                if self._is_phantom(tu.name):
                    current_state = self._registry._last_state
                    if self._on_phantom_tool:
                        self._on_phantom_tool(tu.name, tu.input, current_state)
                    if self._strict:
                        raise PhantomToolError(tu.name, current_state)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(
                                {
                                    "error": f"Unknown tool '{tu.name}'. "
                                    "Only use actions listed in the most recent "
                                    "'Available actions' section."
                                }
                            ),
                            "is_error": True,
                        }
                    )
                    continue

                try:
                    old_state = self._registry._last_state
                    content = self._registry.handle_tool_call(tu.name, tu.input)
                    new_state = self._registry._last_state
                    if self._on_transition and old_state != new_state:
                        self._on_transition(old_state, tu.name, new_state)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": content,
                        }
                    )
                except InvalidActionError:
                    current_state = self._registry._last_state
                    if self._on_invalid_action:
                        self._on_invalid_action(tu.name, tu.input, current_state)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(
                                {
                                    "error": "That action is not available. "
                                    "Check 'Available actions' in the most recent tool result.",
                                }
                            ),
                            "is_error": True,
                        }
                    )
                except NoHandlerError as e:
                    current_state = self._registry._last_state
                    if self._on_phantom_tool:
                        self._on_phantom_tool(tu.name, tu.input, current_state)
                    if self._strict:
                        raise PhantomToolError(tu.name, current_state) from e
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps({"error": str(e)}),
                            "is_error": True,
                        }
                    )
                except Exception:
                    logger.exception("Handler error for tool '%s'", tu.name)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps({"error": "An internal error occurred."}),
                            "is_error": True,
                        }
                    )

            msgs.append({"role": "user", "content": tool_results})

            # Refresh available tools after state may have changed
            tools = self._registry.get_current_tool_schemas()

        # Hit max turns
        return RunResult(
            text="",
            messages=msgs,
            tool_calls=tool_trace,
            truncated=True,
        )

    def get_discovery_report(self):
        """Return a report of all observed state transitions."""
        return self._registry.get_discovery_report()

    def run_multi(
        self,
        user_messages: List[str],
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RunResult:
        """Run the agent loop across multiple user turns."""
        msgs = list(messages) if messages else []
        all_tool_calls: List[Dict[str, Any]] = []
        final_text = ""

        for user_msg in user_messages:
            result = self.run(user_msg, messages=msgs)
            msgs = result.messages
            all_tool_calls.extend(result.tool_calls)
            final_text = result.text

        return RunResult(
            text=final_text,
            messages=msgs,
            tool_calls=all_tool_calls,
        )


class RunResult:
    """Result from running the agent loop."""

    def __init__(
        self,
        text: str,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        truncated: bool = False,
    ):
        self.text = text
        self.messages = messages
        self.tool_calls = tool_calls
        self.truncated = truncated

    @property
    def gateway_calls(self) -> int:
        # The first unique tool name is assumed to be the gateway
        if not self.tool_calls:
            return 0
        gw_name = self.tool_calls[0]["tool"]
        return sum(1 for tc in self.tool_calls if tc["tool"] == gw_name)

    @property
    def dynamic_calls(self) -> int:
        return len(self.tool_calls) - self.gateway_calls

    @property
    def unique_tools(self) -> set[str]:
        return {tc["tool"] for tc in self.tool_calls}

    def __repr__(self) -> str:
        truncated = ", truncated" if self.truncated else ""
        return (
            f"RunResult(text={self.text[:80]!r}..., "
            f"tool_calls={len(self.tool_calls)}, "
            f"gateway={self.gateway_calls}, dynamic={self.dynamic_calls}"
            f"{truncated})"
        )

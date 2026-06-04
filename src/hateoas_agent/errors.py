"""Custom exceptions for hateoas-agent."""

from __future__ import annotations


class HateoasError(Exception):
    """Base exception for hateoas-agent."""


class InvalidActionError(HateoasError):
    """Raised when an action is not valid for the current state."""

    def __init__(self, action: str, state: str, valid_actions: list[str]):
        self.action = action
        self.state = state
        self.valid_actions = valid_actions
        super().__init__(
            f"Action '{action}' is not valid for state '{state}'. Valid actions: {valid_actions}"
        )


class NoHandlerError(HateoasError):
    """Raised when no handler is registered for an action."""

    def __init__(self, action: str):
        self.action = action
        super().__init__(f"No handler registered for action '{action}'")


class NoGatewayError(HateoasError):
    """Raised when no gateway is defined."""

    def __init__(self):
        super().__init__("No gateway tool defined. Call .gateway() first.")


class StateNotFoundError(HateoasError):
    """Raised when a state has no definition."""

    def __init__(self, state: str):
        self.state = state
        super().__init__(f"No actions defined for state '{state}'")


class PhantomToolError(HateoasError):
    """Raised when Claude calls a tool that doesn't exist in any state.

    This is a security event — it means Claude hallucinated or was
    influenced into calling a tool that was never advertised.
    """

    def __init__(self, tool_name: str, state: str | None):
        self.tool_name = tool_name
        self.state = state
        super().__init__(
            f"Phantom tool call: '{tool_name}' is not a registered action "
            f"in any state (current state: '{state}')"
        )


class RunnerAPIError(HateoasError):
    """Raised when the Claude API call fails after the SDK exhausts its retries.

    The Anthropic SDK retries transient errors (429/5xx/network) a few times,
    but a sustained failure still propagates. Rather than let it abort the run
    and discard the conversation, the Runner wraps it in this error and attaches
    the accumulated state so the run can be resumed::

        try:
            result = runner.run(prompt)
        except RunnerAPIError as e:
            result = runner.run(prompt, messages=e.messages)  # resume

    Attributes:
        original: the underlying Anthropic exception.
        messages: the conversation so far (pass back to ``run`` to resume).
        tool_calls: the tool-call trace accumulated before the failure.
    """

    def __init__(self, original: Exception, messages: list, tool_calls: list):
        self.original = original
        self.messages = messages
        self.tool_calls = tool_calls
        super().__init__(
            f"Claude API call failed after the SDK exhausted its retries: "
            f"{type(original).__name__}: {original}. The conversation "
            f"({len(messages)} messages) is preserved on this exception "
            f"(.messages / .tool_calls); pass .messages back to run() to resume."
        )

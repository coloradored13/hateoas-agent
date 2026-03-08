"""Tests for the Runner agent loop (mocked Anthropic client)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hateoas_agent import PhantomToolError, Runner, RunResult, StateMachine

# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_dict, tool_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_dict, id=tool_id)


def _mock_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _make_machine():
    """Minimal StateMachine for runner tests."""
    sm = StateMachine("items", gateway_name="list_items")
    sm.gateway(description="List items", params={"id": "string"})
    sm.state(
        "active",
        actions=[
            {
                "name": "edit_item",
                "description": "Edit",
                "params": {"id": "string", "value": "string"},
            },
        ],
    )
    sm.state(
        "archived",
        actions=[
            {"name": "restore_item", "description": "Restore", "params": {"id": "string"}},
        ],
    )

    @sm.on_gateway
    def handle_list(id=None, **kw):
        return {"items": [{"id": "1"}], "_state": "active"}

    @sm.on_action("edit_item")
    def handle_edit(id, value=""):
        return {"updated": True, "_state": "active"}

    @sm.on_action("restore_item")
    def handle_restore(id):
        return {"restored": True, "_state": "active"}

    return sm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunnerBasicLoop:
    """Test the basic agent loop mechanics."""

    def test_text_only_response(self):
        """When Claude returns only text, the loop ends immediately."""
        sm = _make_machine()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response(
            [_text_block("Hello!")], stop_reason="end_turn"
        )

        runner = Runner(sm, client=mock_client)
        result = runner.run("Hi")

        assert result.text == "Hello!"
        assert result.tool_calls == []
        assert result.truncated is False
        assert len(result.messages) == 2  # user + assistant

    def test_gateway_then_text(self):
        """Claude calls the gateway, gets a result, then responds with text."""
        sm = _make_machine()
        mock_client = MagicMock()

        # Turn 1: Claude calls gateway
        # Turn 2: Claude responds with text
        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("list_items", {"id": "1"})],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Found 1 item.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Show me items")

        assert result.text == "Found 1 item."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "list_items"

    def test_gateway_then_action_then_text(self):
        """Claude calls gateway, then a valid action, then text."""
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("list_items", {"id": "1"}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_tool_use_block("edit_item", {"id": "1", "value": "new"}, "tu_2")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Item updated.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Edit item 1")

        assert result.text == "Item updated."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "list_items"
        assert result.tool_calls[1]["tool"] == "edit_item"


class TestRunnerMaxTurns:
    """Test max_turns enforcement."""

    def test_max_turns_stops_loop(self):
        """Runner stops after max_turns even if Claude keeps calling tools."""
        sm = _make_machine()
        mock_client = MagicMock()

        # Always return a tool call (never finishes)
        mock_client.messages.create.return_value = _mock_response(
            [_tool_use_block("list_items", {"id": "1"})],
            stop_reason="tool_use",
        )

        runner = Runner(sm, client=mock_client, max_turns=3)
        result = runner.run("Keep going")

        assert len(result.tool_calls) == 3
        assert result.text == ""
        assert result.truncated is True
        assert mock_client.messages.create.call_count == 3

    def test_max_turns_one(self):
        """max_turns=1 allows exactly one API call."""
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.return_value = _mock_response(
            [_tool_use_block("list_items", {"id": "1"})],
            stop_reason="tool_use",
        )

        runner = Runner(sm, client=mock_client, max_turns=1)
        runner.run("One shot")

        assert mock_client.messages.create.call_count == 1


class TestRunnerPhantomTool:
    """Test phantom tool detection."""

    def test_phantom_tool_returns_error(self):
        """Unknown tool names get an error response (non-strict mode)."""
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("totally_fake_tool", {"x": 1}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Sorry, I'll try again.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Do something")

        # The error result should be in messages
        tool_result_msg = result.messages[2]  # user message with tool results
        assert tool_result_msg["role"] == "user"
        content = tool_result_msg["content"]
        assert any(r.get("is_error") for r in content)

    def test_phantom_tool_callback(self):
        """on_phantom_tool callback is invoked for phantom tool calls."""
        sm = _make_machine()
        mock_client = MagicMock()
        callback_log = []

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("ghost_tool", {"a": 1}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("OK")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(
            sm,
            client=mock_client,
            on_phantom_tool=lambda name, inp, state: callback_log.append((name, inp, state)),
        )
        runner.run("Try ghost tool")

        assert len(callback_log) == 1
        assert callback_log[0][0] == "ghost_tool"

    def test_strict_mode_raises_on_phantom(self):
        """strict=True raises PhantomToolError on phantom tool calls."""
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.return_value = _mock_response(
            [_tool_use_block("nonexistent_tool", {}, "tu_1")],
            stop_reason="tool_use",
        )

        runner = Runner(sm, client=mock_client, strict=True)

        with pytest.raises(PhantomToolError) as exc_info:
            runner.run("Call something fake")

        assert "nonexistent_tool" in str(exc_info.value)


class TestRunnerInvalidAction:
    """Test invalid action detection (known tool, wrong state)."""

    def test_invalid_action_returns_error_in_result(self):
        """An action called in the wrong state returns an error via the Registry.

        The Registry catches InvalidActionError internally and returns a
        formatted error string, so the Runner receives it as a normal
        (non-exception) tool result containing error text.
        """
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            # Gateway sets state to "active"
            _mock_response(
                [_tool_use_block("list_items", {"id": "1"}, "tu_1")],
                stop_reason="tool_use",
            ),
            # Claude tries restore_item which is only valid in "archived"
            _mock_response(
                [_tool_use_block("restore_item", {"id": "1"}, "tu_2")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Couldn't restore.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Restore item 1")

        assert result.text == "Couldn't restore."
        # The tool result for restore_item should contain an error message
        # It's in messages[4] (user msg after restore_item attempt)
        tool_result_msg = result.messages[4]
        assert tool_result_msg["role"] == "user"
        content_str = tool_result_msg["content"][0]["content"]
        assert "not valid" in content_str.lower() or "error" in content_str.lower()


class TestRunnerSystemPrompt:
    """Test default system prompt generation."""

    def test_default_system_prompt_contains_gateway_name(self):
        sm = _make_machine()
        mock_client = MagicMock()
        runner = Runner(sm, client=mock_client)

        assert "list_items" in runner._system

    def test_default_system_prompt_contains_defensive_language(self):
        sm = _make_machine()
        mock_client = MagicMock()
        runner = Runner(sm, client=mock_client)

        assert "Only use actions explicitly listed" in runner._system
        assert "Never call a tool name that hasn't been advertised" in runner._system

    def test_custom_system_prompt(self):
        sm = _make_machine()
        mock_client = MagicMock()
        runner = Runner(sm, client=mock_client, system="Custom system prompt")

        assert runner._system == "Custom system prompt"


class TestRunnerCallbacks:
    """Test on_tool_call and on_text callbacks."""

    def test_on_tool_call_callback(self):
        sm = _make_machine()
        mock_client = MagicMock()
        tool_log = []

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("list_items", {"id": "1"}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Done.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(
            sm,
            client=mock_client,
            on_tool_call=lambda name, inp: tool_log.append((name, inp)),
        )
        runner.run("Show items")

        assert len(tool_log) == 1
        assert tool_log[0] == ("list_items", {"id": "1"})

    def test_on_text_callback(self):
        sm = _make_machine()
        mock_client = MagicMock()
        text_log = []

        mock_client.messages.create.return_value = _mock_response(
            [_text_block("Here is some output.")],
            stop_reason="end_turn",
        )

        runner = Runner(
            sm,
            client=mock_client,
            on_text=lambda text: text_log.append(text),
        )
        runner.run("Say something")

        assert len(text_log) == 1
        assert text_log[0] == "Here is some output."

    def test_on_text_not_called_for_empty_text(self):
        sm = _make_machine()
        mock_client = MagicMock()
        text_log = []

        mock_client.messages.create.return_value = _mock_response(
            [_text_block("  ")],  # whitespace-only
            stop_reason="end_turn",
        )

        runner = Runner(
            sm,
            client=mock_client,
            on_text=lambda text: text_log.append(text),
        )
        runner.run("Say nothing")

        # Whitespace text should not trigger on_text
        assert len(text_log) == 0


class TestRunnerToolTrace:
    """Test tool trace recording in RunResult."""

    def test_tool_trace_recorded(self):
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("list_items", {"id": "1"}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_tool_use_block("edit_item", {"id": "1", "value": "v"}, "tu_2")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Done")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Edit item")

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0] == {"tool": "list_items", "input": {"id": "1"}}
        assert result.tool_calls[1] == {"tool": "edit_item", "input": {"id": "1", "value": "v"}}


class TestRunResult:
    """Test RunResult properties."""

    def test_gateway_calls_count(self):
        result = RunResult(
            text="",
            messages=[],
            tool_calls=[
                {"tool": "list_items", "input": {}},
                {"tool": "edit_item", "input": {}},
                {"tool": "list_items", "input": {}},
            ],
        )
        assert result.gateway_calls == 2

    def test_dynamic_calls_count(self):
        result = RunResult(
            text="",
            messages=[],
            tool_calls=[
                {"tool": "list_items", "input": {}},
                {"tool": "edit_item", "input": {}},
                {"tool": "list_items", "input": {}},
            ],
        )
        assert result.dynamic_calls == 1

    def test_unique_tools(self):
        result = RunResult(
            text="",
            messages=[],
            tool_calls=[
                {"tool": "list_items", "input": {}},
                {"tool": "edit_item", "input": {}},
                {"tool": "list_items", "input": {}},
                {"tool": "edit_item", "input": {}},
            ],
        )
        assert result.unique_tools == {"list_items", "edit_item"}

    def test_empty_tool_calls(self):
        result = RunResult(text="Done", messages=[], tool_calls=[])
        assert result.gateway_calls == 0
        assert result.dynamic_calls == 0
        assert result.unique_tools == set()

    def test_repr(self):
        result = RunResult(
            text="Short text",
            messages=[],
            tool_calls=[{"tool": "gw", "input": {}}],
        )
        r = repr(result)
        assert "RunResult" in r
        assert "tool_calls=1" in r


class TestRunnerNoHandlerError:
    """Test handling of NoHandlerError (known action but no handler)."""

    def test_no_handler_caught_by_validate_at_init(self):
        """A known action name with no handler is caught by validate() at Runner init."""
        sm = StateMachine("test", gateway_name="start")
        sm.gateway(description="Start", params={})
        # Define a state with an action but never register a handler
        sm.state(
            "active",
            actions=[
                {"name": "do_thing", "description": "Do", "params": {}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        mock_client = MagicMock()

        with pytest.raises(ValueError, match="actions without handlers.*do_thing"):
            Runner(sm, client=mock_client, strict=True)


class TestRunnerHandlerException:
    """Test that handler exceptions are caught and returned as errors."""

    def test_handler_exception_returns_error(self):
        sm = StateMachine("test", gateway_name="start")
        sm.gateway(description="Start", params={})
        sm.state(
            "active",
            actions=[
                {"name": "fail_action", "description": "Fail", "params": {}},
            ],
        )

        @sm.on_gateway
        def gw(**kw):
            return {"ok": True, "_state": "active"}

        @sm.on_action("fail_action")
        def fail(**kw):
            raise ValueError("Something broke!")

        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            _mock_response(
                [_tool_use_block("start", {}, "tu_1")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_tool_use_block("fail_action", {}, "tu_2")],
                stop_reason="tool_use",
            ),
            _mock_response(
                [_text_block("Something went wrong.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Do the failing thing")

        # Should not crash - error is returned to Claude
        assert result.text == "Something went wrong."
        # Check the tool result had an error
        tool_result_msg = result.messages[4]  # user msg after fail_action
        error_result = tool_result_msg["content"][0]
        assert error_result["is_error"] is True
        assert "An internal error occurred." in error_result["content"]


class TestRunnerEndTurnWithToolUse:
    """Test stop_reason=end_turn when there are tool uses."""

    def test_end_turn_with_tool_calls_continues_loop(self):
        """If stop_reason is end_turn but there are tool uses,
        tool results are still sent back to Claude for a follow-up."""
        sm = _make_machine()
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = [
            # Claude returns tool use with stop_reason=end_turn
            _mock_response(
                [
                    _text_block("Let me check."),
                    _tool_use_block("list_items", {"id": "1"}, "tu_1"),
                ],
                stop_reason="end_turn",
            ),
            # After tool results, Claude responds with text
            _mock_response(
                [_text_block("Found the items.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Check items")

        # Tool should have been processed
        assert len(result.tool_calls) == 1
        # Claude got the tool results and responded
        assert result.text == "Found the items."


class TestRunnerMultiTurn:
    """Test run_multi for multi-turn conversations."""

    def test_run_multi_chains_messages(self):
        sm = _make_machine()
        mock_client = MagicMock()

        # First run: text response
        # Second run: text response
        mock_client.messages.create.side_effect = [
            _mock_response([_text_block("First response.")], stop_reason="end_turn"),
            _mock_response([_text_block("Second response.")], stop_reason="end_turn"),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run_multi(["Hello", "Goodbye"])

        assert result.text == "Second response."
        # Messages should contain both user messages
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        assert len(user_msgs) == 2

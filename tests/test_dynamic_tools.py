"""Tests for dynamic tool registration (get_current_tool_schemas + Runner refresh)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from hateoas_agent import Registry, Runner, StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_dict, tool_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_dict, id=tool_id)


def _mock_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _make_machine():
    """State machine with gateway + 2 states to test dynamic tool changes."""
    sm = StateMachine("files", gateway_name="open_folder")
    sm.gateway(description="Open a folder", params={"path": "string"})
    sm.state("browsing", actions=[
        {"name": "select_file", "description": "Select a file", "params": {"name": "string"}, "required": ["name"]},
        {"name": "create_file", "description": "Create a new file", "params": {"name": "string"}},
    ])
    sm.state("editing", actions=[
        {"name": "save_file", "description": "Save the file"},
        {"name": "close_file", "description": "Close and go back"},
    ])

    @sm.on_gateway
    def handle_open(path=""):
        return {"files": ["a.txt", "b.txt"], "_state": "browsing"}

    @sm.on_action("select_file")
    def handle_select(name):
        return {"content": "file contents", "_state": "editing"}

    @sm.on_action("create_file")
    def handle_create(name=""):
        return {"created": name, "_state": "browsing"}

    @sm.on_action("save_file")
    def handle_save():
        return {"saved": True, "_state": "editing"}

    @sm.on_action("close_file")
    def handle_close():
        return {"closed": True, "_state": "browsing"}

    return sm


# ---------------------------------------------------------------------------
# Registry.get_current_tool_schemas tests
# ---------------------------------------------------------------------------


class TestGetCurrentToolSchemas:
    """Tests for Registry.get_current_tool_schemas()."""

    def test_before_gateway_returns_only_gateway(self):
        """Before any gateway call, only the gateway tool is returned."""
        sm = _make_machine()
        reg = Registry(sm)
        tools = reg.get_current_tool_schemas()
        assert len(tools) == 1
        assert tools[0]["name"] == "open_folder"

    def test_after_gateway_returns_gateway_plus_state_actions(self):
        """After gateway call sets state, tools include state actions."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        tools = reg.get_current_tool_schemas()
        names = {t["name"] for t in tools}
        assert names == {"open_folder", "select_file", "create_file"}

    def test_tools_change_on_state_transition(self):
        """When state changes, the tool list reflects the new state."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        reg.handle_tool_call("select_file", {"name": "a.txt"})
        tools = reg.get_current_tool_schemas()
        names = {t["name"] for t in tools}
        assert names == {"open_folder", "save_file", "close_file"}
        # browsing actions should NOT be present
        assert "select_file" not in names
        assert "create_file" not in names

    def test_tools_update_on_transition_back(self):
        """Transitioning back to a previous state restores its tools."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        reg.handle_tool_call("select_file", {"name": "a.txt"})
        reg.handle_tool_call("close_file", {})
        tools = reg.get_current_tool_schemas()
        names = {t["name"] for t in tools}
        assert names == {"open_folder", "select_file", "create_file"}

    def test_tool_schemas_have_valid_structure(self):
        """Each tool schema has name, description, and input_schema."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        tools = reg.get_current_tool_schemas()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]

    def test_required_params_in_schema(self):
        """Required params are included in the tool schema."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        tools = reg.get_current_tool_schemas()
        select_tool = next(t for t in tools if t["name"] == "select_file")
        assert select_tool["input_schema"]["required"] == ["name"]

    def test_no_required_omits_key(self):
        """Tools without required params don't have a required key."""
        sm = _make_machine()
        reg = Registry(sm)
        reg.handle_tool_call("open_folder", {"path": "/tmp"})
        tools = reg.get_current_tool_schemas()
        create_tool = next(t for t in tools if t["name"] == "create_file")
        assert "required" not in create_tool["input_schema"]

    def test_empty_state_returns_only_gateway(self):
        """A state with no actions returns only the gateway tool."""
        sm = StateMachine("test", gateway_name="start")
        sm.gateway(description="Start", params={})
        sm.state("empty")

        @sm.on_gateway
        def handle():
            return {"ok": True, "_state": "empty"}

        reg = Registry(sm)
        reg.handle_tool_call("start", {})
        tools = reg.get_current_tool_schemas()
        assert len(tools) == 1
        assert tools[0]["name"] == "start"


# ---------------------------------------------------------------------------
# Param type normalization tests
# ---------------------------------------------------------------------------


class TestParamTypeNormalization:
    """Tests for _normalize_param_type handling descriptive type strings."""

    def test_simple_string_type(self):
        sm = StateMachine("t", gateway_name="g")
        sm.gateway(description="G", params={"x": "string"})
        sm.state("s", actions=[
            {"name": "a", "description": "A", "params": {"x": "string"}},
        ])

        @sm.on_gateway
        def gw():
            return {"_state": "s"}

        @sm.on_action("a")
        def act(x=""):
            return {"_state": "s"}

        reg = Registry(sm)
        reg.handle_tool_call("g", {})
        tools = reg.get_current_tool_schemas()
        action_tool = next(t for t in tools if t["name"] == "a")
        assert action_tool["input_schema"]["properties"]["x"]["type"] == "string"

    def test_descriptive_type_extracts_base(self):
        """Types like 'string (comma-separated)' normalize to 'string'."""
        sm = StateMachine("t", gateway_name="g")
        sm.gateway(description="G", params={})
        sm.state("s", actions=[
            {"name": "a", "description": "A", "params": {
                "cols": "string (comma-separated column defs)",
            }},
        ])

        @sm.on_gateway
        def gw():
            return {"_state": "s"}

        @sm.on_action("a")
        def act(cols=""):
            return {"_state": "s"}

        reg = Registry(sm)
        reg.handle_tool_call("g", {})
        tools = reg.get_current_tool_schemas()
        action_tool = next(t for t in tools if t["name"] == "a")
        prop = action_tool["input_schema"]["properties"]["cols"]
        assert prop["type"] == "string"
        assert "comma-separated" in prop.get("description", "")

    def test_integer_type(self):
        sm = StateMachine("t", gateway_name="g")
        sm.gateway(description="G", params={})
        sm.state("s", actions=[
            {"name": "a", "description": "A", "params": {"n": "integer"}},
        ])

        @sm.on_gateway
        def gw():
            return {"_state": "s"}

        @sm.on_action("a")
        def act(n=0):
            return {"_state": "s"}

        reg = Registry(sm)
        reg.handle_tool_call("g", {})
        tools = reg.get_current_tool_schemas()
        action_tool = next(t for t in tools if t["name"] == "a")
        assert action_tool["input_schema"]["properties"]["n"]["type"] == "integer"

    def test_unknown_type_defaults_to_string(self):
        """Unrecognized types fall back to 'string'."""
        sm = StateMachine("t", gateway_name="g")
        sm.gateway(description="G", params={})
        sm.state("s", actions=[
            {"name": "a", "description": "A", "params": {"x": "foobar"}},
        ])

        @sm.on_gateway
        def gw():
            return {"_state": "s"}

        @sm.on_action("a")
        def act(x=""):
            return {"_state": "s"}

        reg = Registry(sm)
        reg.handle_tool_call("g", {})
        tools = reg.get_current_tool_schemas()
        action_tool = next(t for t in tools if t["name"] == "a")
        assert action_tool["input_schema"]["properties"]["x"]["type"] == "string"


# ---------------------------------------------------------------------------
# Runner dynamic tool refresh tests
# ---------------------------------------------------------------------------


class TestRunnerDynamicTools:
    """Tests that the Runner refreshes tools after state transitions."""

    def test_runner_passes_dynamic_tools_to_api(self):
        """After gateway, the API call includes state action tools."""
        sm = _make_machine()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            # Turn 1: Claude calls gateway
            _mock_response(
                [_tool_use_block("open_folder", {"path": "/tmp"})],
                stop_reason="tool_use",
            ),
            # Turn 2: Claude calls select_file (now available)
            _mock_response(
                [_tool_use_block("select_file", {"name": "a.txt"}, "tu_2")],
                stop_reason="tool_use",
            ),
            # Turn 3: Claude responds with text
            _mock_response(
                [_text_block("Here's the file content.")],
                stop_reason="end_turn",
            ),
        ]

        runner = Runner(sm, client=mock_client)
        result = runner.run("Open /tmp and read a.txt")

        # Verify the API was called 3 times
        assert mock_client.messages.create.call_count == 3

        # Turn 1: only gateway tool
        call_1_tools = mock_client.messages.create.call_args_list[0].kwargs["tools"]
        tool_names_1 = {t["name"] for t in call_1_tools}
        assert tool_names_1 == {"open_folder"}

        # Turn 2: gateway + browsing actions
        call_2_tools = mock_client.messages.create.call_args_list[1].kwargs["tools"]
        tool_names_2 = {t["name"] for t in call_2_tools}
        assert tool_names_2 == {"open_folder", "select_file", "create_file"}

        # Turn 3: gateway + editing actions (state changed)
        call_3_tools = mock_client.messages.create.call_args_list[2].kwargs["tools"]
        tool_names_3 = {t["name"] for t in call_3_tools}
        assert tool_names_3 == {"open_folder", "save_file", "close_file"}

    def test_runner_tools_reflect_state_after_multiple_transitions(self):
        """Tools update correctly through gateway → browsing → editing → browsing."""
        sm = _make_machine()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_response([_tool_use_block("open_folder", {"path": "/"})], "tool_use"),
            _mock_response([_tool_use_block("select_file", {"name": "x"}, "tu_2")], "tool_use"),
            _mock_response([_tool_use_block("close_file", {}, "tu_3")], "tool_use"),
            _mock_response([_text_block("Done.")], "end_turn"),
        ]

        runner = Runner(sm, client=mock_client)
        runner.run("Open, select, close")

        # After close_file → back to browsing
        call_4_tools = mock_client.messages.create.call_args_list[3].kwargs["tools"]
        tool_names = {t["name"] for t in call_4_tools}
        assert tool_names == {"open_folder", "select_file", "create_file"}

    def test_runner_starts_with_gateway_only(self):
        """First API call should only include the gateway tool."""
        sm = _make_machine()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response(
            [_text_block("No tools needed.")], "end_turn"
        )

        runner = Runner(sm, client=mock_client)
        runner.run("Just say hello")

        call_tools = mock_client.messages.create.call_args_list[0].kwargs["tools"]
        assert len(call_tools) == 1
        assert call_tools[0]["name"] == "open_folder"

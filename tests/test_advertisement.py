"""Tests for advertisement formatting."""

import json

from hateoas_agent import format_error_with_actions, format_result_with_actions
from hateoas_agent.types import ActionDef


class TestFormatResultWithActions:
    """Test format_result_with_actions with various inputs."""

    def test_result_with_actions(self):
        result = {"order": {"id": "1", "status": "pending"}}
        actions = [
            ActionDef(name="approve", description="Approve order", params={"id": "string"}),
        ]
        output = format_result_with_actions(result, actions)

        assert '"order"' in output
        assert "---" in output
        assert "Available actions for this resource:" in output
        assert "approve: Approve order" in output
        assert '{"id": "string"}' in output

    def test_result_with_no_actions(self):
        result = {"data": "value"}
        output = format_result_with_actions(result, [])

        assert '"data"' in output
        assert "---" not in output
        assert "Available actions" not in output

    def test_result_with_multiple_actions(self):
        result = {"id": "1"}
        actions = [
            ActionDef(name="edit", description="Edit it", params={"id": "string"}),
            ActionDef(name="delete", description="Delete it", params={"id": "string"}),
            ActionDef(name="view", description="View it", params={}),
        ]
        output = format_result_with_actions(result, actions)

        assert "edit: Edit it" in output
        assert "delete: Delete it" in output
        assert "view: View it" in output

    def test_result_with_empty_params(self):
        result = {"ok": True}
        actions = [
            ActionDef(name="do_it", description="Do it", params={}),
        ]
        output = format_result_with_actions(result, actions)

        assert "do_it: Do it" in output
        assert "params: {}" in output

    def test_result_with_complex_data(self):
        result = {
            "items": [{"id": 1}, {"id": 2}],
            "nested": {"a": {"b": "c"}},
            "count": 42,
        }
        actions = [ActionDef(name="next_page", description="Next", params={})]
        output = format_result_with_actions(result, actions)

        # Verify JSON is valid by parsing the data portion
        data_part = output.split("\n\n---")[0]
        parsed = json.loads(data_part)
        assert parsed["count"] == 42

    def test_result_with_special_characters_in_description(self):
        result = {"ok": True}
        actions = [
            ActionDef(
                name="update",
                description='Update with "quotes" & <brackets>',
                params={"note": "string"},
            ),
        ]
        output = format_result_with_actions(result, actions)

        assert '"quotes"' in output
        assert "<brackets>" in output

    def test_result_with_non_serializable_value(self):
        """format_result_with_actions uses default=str for non-serializable types."""
        from datetime import datetime

        result = {"timestamp": datetime(2024, 1, 1, 12, 0, 0)}
        actions = []
        output = format_result_with_actions(result, actions)

        # Should not raise — datetime is handled by default=str
        assert "2024" in output


class TestFormatErrorWithActions:
    """Test format_error_with_actions with various inputs."""

    def test_error_with_actions(self):
        output = format_error_with_actions(
            "Action not valid",
            [ActionDef(name="retry", description="Try again", params={})],
        )

        assert '"error"' in output
        assert "Action not valid" in output
        assert "retry: Try again" in output

    def test_error_with_no_actions(self):
        output = format_error_with_actions("Something broke", [])

        assert '"error"' in output
        assert "Something broke" in output
        assert "Available actions" not in output

    def test_error_message_with_special_characters(self):
        output = format_error_with_actions(
            'Can\'t do that: value is <nil> & "missing"',
            [],
        )

        # The error message should be in JSON, so special chars are escaped
        parsed = json.loads(output)
        assert "Can't do that" in parsed["error"]

    def test_error_with_multiple_actions(self):
        output = format_error_with_actions(
            "Invalid state",
            [
                ActionDef(name="a1", description="Action 1", params={"x": "int"}),
                ActionDef(name="a2", description="Action 2", params={"y": "string"}),
            ],
        )

        assert "a1: Action 1" in output
        assert "a2: Action 2" in output

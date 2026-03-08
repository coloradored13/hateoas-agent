"""Tests for the validate_action function."""

import pytest

from hateoas_agent import InvalidActionError, validate_action
from hateoas_agent.types import ActionDef


class TestValidateAction:
    """Test validate_action directly."""

    def test_valid_action_returns_action_def(self):
        actions = [
            ActionDef(name="approve", description="Approve", params={}),
            ActionDef(name="cancel", description="Cancel", params={}),
        ]
        result = validate_action("approve", "pending", actions)
        assert result.name == "approve"

    def test_invalid_action_raises(self):
        actions = [
            ActionDef(name="approve", description="Approve", params={}),
        ]
        with pytest.raises(InvalidActionError) as exc_info:
            validate_action("ship", "pending", actions)

        assert exc_info.value.action == "ship"
        assert exc_info.value.state == "pending"
        assert exc_info.value.valid_actions == ["approve"]

    def test_empty_actions_list(self):
        with pytest.raises(InvalidActionError) as exc_info:
            validate_action("anything", "empty_state", [])

        assert exc_info.value.valid_actions == []

    def test_first_matching_action_returned(self):
        """When multiple actions exist, the first match is returned."""
        actions = [
            ActionDef(name="edit", description="Edit v1", params={}),
            ActionDef(name="delete", description="Delete", params={}),
        ]
        result = validate_action("delete", "active", actions)
        assert result.name == "delete"
        assert result.description == "Delete"

    def test_error_message_includes_context(self):
        actions = [
            ActionDef(name="a1", description="A1", params={}),
            ActionDef(name="a2", description="A2", params={}),
        ]
        with pytest.raises(InvalidActionError) as exc_info:
            validate_action("bad_action", "my_state", actions)

        msg = str(exc_info.value)
        assert "bad_action" in msg
        assert "my_state" in msg
        assert "a1" in msg
        assert "a2" in msg

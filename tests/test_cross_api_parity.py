"""Cross-API parity tests: StateMachine and Resource produce identical behavior."""

from hateoas_agent import Registry, Resource, StateMachine, action, gateway, state


def _build_state_machine():
    """Build a StateMachine with a known state graph."""
    sm = StateMachine("tickets", gateway_name="search_tickets")
    sm.gateway(description="Search tickets", params={"ticket_id": "string"})
    sm.state(
        "open",
        actions=[
            {
                "name": "assign_ticket",
                "description": "Assign",
                "params": {"ticket_id": "string", "user": "string"},
            },
            {"name": "close_ticket", "description": "Close", "params": {"ticket_id": "string"}},
        ],
    )
    sm.state(
        "assigned",
        actions=[
            {"name": "resolve_ticket", "description": "Resolve", "params": {"ticket_id": "string"}},
            {"name": "close_ticket", "description": "Close", "params": {"ticket_id": "string"}},
        ],
    )
    sm.state(
        "resolved",
        actions=[
            {"name": "reopen_ticket", "description": "Reopen", "params": {"ticket_id": "string"}},
        ],
    )

    db = {"T1": {"id": "T1", "status": "open", "title": "Bug"}}

    @sm.on_gateway
    def search(ticket_id=None, **kw):
        if ticket_id and ticket_id in db:
            t = db[ticket_id]
            return {"ticket": t, "_state": t["status"]}
        return {"message": "Not found"}

    @sm.on_action("assign_ticket")
    def assign(ticket_id, user=""):
        db[ticket_id]["status"] = "assigned"
        return {"assigned": True, "_state": "assigned"}

    @sm.on_action("close_ticket")
    def close(ticket_id):
        db[ticket_id]["status"] = "closed"
        return {"closed": True, "_state": "closed"}

    @sm.on_action("resolve_ticket")
    def resolve(ticket_id):
        db[ticket_id]["status"] = "resolved"
        return {"resolved": True, "_state": "resolved"}

    @sm.on_action("reopen_ticket")
    def reopen(ticket_id):
        db[ticket_id]["status"] = "open"
        return {"reopened": True, "_state": "open"}

    return sm


def _build_resource():
    """Build a Resource with the same state graph."""
    db = {"T1": {"id": "T1", "status": "open", "title": "Bug"}}

    class TicketResource(Resource):
        name = "tickets"

        @gateway(
            name="search_tickets", description="Search tickets", params={"ticket_id": "string"}
        )
        def search(self, ticket_id=None, **kw):
            if ticket_id and ticket_id in db:
                t = db[ticket_id]
                return {"ticket": t, "_state": t["status"]}
            return {"message": "Not found"}

        @action(
            name="assign_ticket",
            description="Assign",
            params={"ticket_id": "string", "user": "string"},
        )
        @state("open")
        def assign(self, ticket_id, user=""):
            db[ticket_id]["status"] = "assigned"
            return {"assigned": True, "_state": "assigned"}

        @action(name="close_ticket", description="Close", params={"ticket_id": "string"})
        @state("open", "assigned")
        def close(self, ticket_id):
            db[ticket_id]["status"] = "closed"
            return {"closed": True, "_state": "closed"}

        @action(name="resolve_ticket", description="Resolve", params={"ticket_id": "string"})
        @state("assigned")
        def resolve(self, ticket_id):
            db[ticket_id]["status"] = "resolved"
            return {"resolved": True, "_state": "resolved"}

        @action(name="reopen_ticket", description="Reopen", params={"ticket_id": "string"})
        @state("resolved")
        def reopen(self, ticket_id):
            db[ticket_id]["status"] = "open"
            return {"reopened": True, "_state": "open"}

    return TicketResource()


class TestCrossAPIParity:
    """Verify that StateMachine and Resource behave identically for the same state graph."""

    def test_gateway_name_matches(self):
        sm = _build_state_machine()
        res = _build_resource()
        assert sm.get_gateway().name == res.get_gateway().name

    def test_gateway_description_matches(self):
        sm = _build_state_machine()
        res = _build_resource()
        assert sm.get_gateway().description == res.get_gateway().description

    def test_gateway_params_match(self):
        sm = _build_state_machine()
        res = _build_resource()
        assert sm.get_gateway().params == res.get_gateway().params

    def test_all_action_names_match(self):
        sm = _build_state_machine()
        res = _build_resource()
        assert sm.get_all_action_names() == res.get_all_action_names()

    def test_open_state_actions_match(self):
        sm = _build_state_machine()
        res = _build_resource()
        sm_names = sorted(a.name for a in sm.get_actions_for_state("open"))
        res_names = sorted(a.name for a in res.get_actions_for_state("open"))
        assert sm_names == res_names

    def test_assigned_state_actions_match(self):
        sm = _build_state_machine()
        res = _build_resource()
        sm_names = sorted(a.name for a in sm.get_actions_for_state("assigned"))
        res_names = sorted(a.name for a in res.get_actions_for_state("assigned"))
        assert sm_names == res_names

    def test_resolved_state_actions_match(self):
        sm = _build_state_machine()
        res = _build_resource()
        sm_names = sorted(a.name for a in sm.get_actions_for_state("resolved"))
        res_names = sorted(a.name for a in res.get_actions_for_state("resolved"))
        assert sm_names == res_names

    def test_unknown_state_returns_empty_for_both(self):
        sm = _build_state_machine()
        res = _build_resource()
        assert sm.get_actions_for_state("nonexistent") == []
        assert res.get_actions_for_state("nonexistent") == []

    def test_gateway_handler_produces_same_data(self):
        sm = _build_state_machine()
        res = _build_resource()

        sm_result = sm.get_gateway().handler(ticket_id="T1")
        res_result = res.get_gateway().handler(ticket_id="T1")

        # Both should return the same ticket data and state
        assert sm_result["ticket"]["id"] == res_result["ticket"]["id"]
        assert sm_result["_state"] == res_result["_state"]

    def test_gateway_handler_not_found_same(self):
        sm = _build_state_machine()
        res = _build_resource()

        sm_result = sm.get_gateway().handler(ticket_id="MISSING")
        res_result = res.get_gateway().handler(ticket_id="MISSING")

        assert sm_result == res_result

    def test_registry_gateway_schema_matches(self):
        sm = _build_state_machine()
        res = _build_resource()

        sm_schema = Registry(sm).get_gateway_tool_schema()
        res_schema = Registry(res).get_gateway_tool_schema()

        assert sm_schema["name"] == res_schema["name"]
        assert sm_schema["description"] == res_schema["description"]
        assert sm_schema["input_schema"] == res_schema["input_schema"]

    def test_registry_handle_gateway_both_advertise_actions(self):
        """Both APIs should advertise the same actions after a gateway call."""
        sm = _build_state_machine()
        res = _build_resource()

        sm_reg = Registry(sm)
        res_reg = Registry(res)

        sm_result = sm_reg.handle_tool_call("search_tickets", {"ticket_id": "T1"})
        res_result = res_reg.handle_tool_call("search_tickets", {"ticket_id": "T1"})

        # Both should mention the same actions
        assert ("assign_ticket" in sm_result) == ("assign_ticket" in res_result)
        assert ("close_ticket" in sm_result) == ("close_ticket" in res_result)

    def test_registry_state_transition_same_behavior(self):
        """Full transition chain produces same behavior through both APIs."""
        sm = _build_state_machine()
        res = _build_resource()

        sm_reg = Registry(sm)
        res_reg = Registry(res)

        # Gateway -> open
        sm_reg.handle_tool_call("search_tickets", {"ticket_id": "T1"})
        res_reg.handle_tool_call("search_tickets", {"ticket_id": "T1"})

        assert sm_reg._last_state == res_reg._last_state == "open"

        # Assign -> assigned
        sm_reg.handle_tool_call("assign_ticket", {"ticket_id": "T1", "user": "alice"})
        res_reg.handle_tool_call("assign_ticket", {"ticket_id": "T1", "user": "alice"})

        assert sm_reg._last_state == res_reg._last_state == "assigned"

        # Resolve -> resolved
        sm_reg.handle_tool_call("resolve_ticket", {"ticket_id": "T1"})
        res_reg.handle_tool_call("resolve_ticket", {"ticket_id": "T1"})

        assert sm_reg._last_state == res_reg._last_state == "resolved"

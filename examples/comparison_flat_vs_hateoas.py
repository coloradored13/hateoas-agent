"""
Side-by-side comparison: Flat tool registration vs HATEOAS dynamic discovery.

Run with:
    python examples/comparison_flat_vs_hateoas.py

This script demonstrates the same database admin task using two approaches:

1. FLAT: All 15 tools registered upfront (traditional approach)
2. HATEOAS: Tools discovered dynamically based on state

The output shows exactly which tools Claude sees at each turn, making the
differences immediately visible.
"""

import json
import sqlite3
import os
import tempfile
from types import SimpleNamespace

import anthropic

# ============================================================
# Shared setup
# ============================================================

SAMPLE_DB_DIR = os.path.join(tempfile.gettempdir(), "hateoas_comparison_demo")
PROMPT = (
    "Connect to the company database, look at the employees table, "
    "then drop the projects table."
)
MODEL = "claude-sonnet-4-20250514"
MAX_TURNS = 15


def create_sample_db():
    os.makedirs(SAMPLE_DB_DIR, exist_ok=True)
    db_path = os.path.join(SAMPLE_DB_DIR, "company.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS departments")
    c.execute("DROP TABLE IF EXISTS employees")
    c.execute("DROP TABLE IF EXISTS projects")
    c.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL)")
    c.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, role TEXT, salary REAL)")
    c.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, status TEXT, deadline TEXT)")
    c.executemany("INSERT INTO departments VALUES (?, ?, ?)", [
        (1, "Engineering", 500000), (2, "Marketing", 200000),
    ])
    c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", [
        (1, "Alice Chen", 1, "Senior Engineer", 145000),
        (2, "Bob Martinez", 1, "Staff Engineer", 175000),
        (3, "Carol White", 2, "Marketing Manager", 110000),
    ])
    c.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", [
        (1, "Platform Rewrite", 1, "in_progress", "2026-06-30"),
        (2, "Q2 Campaign", 2, "planning", "2026-04-15"),
    ])
    conn.commit()
    conn.close()
    return db_path


# ============================================================
# APPROACH 1: Flat tool registration (all 15 tools upfront)
# ============================================================

FLAT_TOOLS = [
    {"name": "explore_database", "description": "Connect to a SQLite database or list available databases", "input_schema": {"type": "object", "properties": {"database_name": {"type": "string"}}}},
    {"name": "list_tables", "description": "List all tables in the connected database", "input_schema": {"type": "object", "properties": {}}},
    {"name": "select_table", "description": "Select a table to work with", "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "describe_table", "description": "Show the schema of the selected table", "input_schema": {"type": "object", "properties": {}}},
    {"name": "sample_rows", "description": "Preview rows from the selected table", "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "insert_row", "description": "Insert a row into the selected table", "input_schema": {"type": "object", "properties": {"values": {"type": "string", "description": "JSON object of column:value pairs"}}, "required": ["values"]}},
    {"name": "update_rows", "description": "Update rows matching a condition", "input_schema": {"type": "object", "properties": {"set_clause": {"type": "string"}, "where_clause": {"type": "string"}}, "required": ["set_clause", "where_clause"]}},
    {"name": "delete_rows", "description": "Delete rows matching a condition", "input_schema": {"type": "object", "properties": {"where_clause": {"type": "string"}}, "required": ["where_clause"]}},
    {"name": "add_column", "description": "Add a column to the selected table", "input_schema": {"type": "object", "properties": {"column_def": {"type": "string"}}, "required": ["column_def"]}},
    {"name": "create_table", "description": "Create a new table", "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "columns": {"type": "string"}}, "required": ["table_name", "columns"]}},
    {"name": "drop_table", "description": "Drop the selected table", "input_schema": {"type": "object", "properties": {}}},
    {"name": "confirm_drop", "description": "Confirm dropping a table", "input_schema": {"type": "object", "properties": {}}},
    {"name": "cancel_drop", "description": "Cancel a pending drop", "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_query", "description": "Run a read-only SQL query (SELECT only)", "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "disconnect", "description": "Disconnect from the current database", "input_schema": {"type": "object", "properties": {}}},
]

FLAT_SYSTEM = (
    "You are a database admin assistant. Use the provided tools to help "
    "the user manage their SQLite databases. Be careful with destructive "
    "operations like drop_table."
)


def run_flat(db_path):
    """Run with all 15 tools registered upfront — no state management."""
    client = anthropic.Anthropic()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    selected_table = None
    pending_drop = None
    connected = False

    tool_log = []
    tools_per_turn = []
    errors = []

    def handle_tool(name, inp):
        nonlocal selected_table, pending_drop, connected
        try:
            if name == "explore_database":
                db_name = inp.get("database_name", "")
                if not db_name:
                    return json.dumps({"databases": ["company.db"]})
                connected = True
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                return json.dumps({"connected": True, "tables": [r["name"] for r in tables]})
            elif name == "list_tables":
                if not connected:
                    errors.append(f"list_tables called before connecting")
                    return json.dumps({"error": "Not connected to a database"})
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                return json.dumps({"tables": [r["name"] for r in tables]})
            elif name == "select_table":
                if not connected:
                    errors.append(f"select_table called before connecting")
                    return json.dumps({"error": "Not connected to a database"})
                selected_table = inp["table_name"]
                cols = conn.execute(f'PRAGMA table_info("{selected_table}")').fetchall()
                return json.dumps({"selected": selected_table, "columns": [dict(c) for c in cols]})
            elif name == "describe_table":
                if not selected_table:
                    errors.append("describe_table called with no table selected")
                    return json.dumps({"error": "No table selected"})
                cols = conn.execute(f'PRAGMA table_info("{selected_table}")').fetchall()
                return json.dumps({"table": selected_table, "columns": [dict(c) for c in cols]})
            elif name == "sample_rows":
                if not selected_table:
                    errors.append("sample_rows called with no table selected")
                    return json.dumps({"error": "No table selected"})
                limit = inp.get("limit", 10)
                rows = conn.execute(f'SELECT * FROM "{selected_table}" LIMIT ?', (limit,)).fetchall()
                return json.dumps({"rows": [dict(r) for r in rows]})
            elif name == "drop_table":
                if not selected_table:
                    errors.append("drop_table called with no table selected")
                    return json.dumps({"error": "No table selected"})
                pending_drop = selected_table
                count = conn.execute(f'SELECT COUNT(*) as c FROM "{selected_table}"').fetchone()
                return json.dumps({"warning": f"About to drop {selected_table} ({count['c']} rows). Call confirm_drop to proceed."})
            elif name == "confirm_drop":
                if not pending_drop:
                    errors.append("confirm_drop called with no pending drop")
                    return json.dumps({"error": "No pending drop operation"})
                conn.execute(f'DROP TABLE "{pending_drop}"')
                conn.commit()
                dropped = pending_drop
                pending_drop = None
                selected_table = None
                return json.dumps({"dropped": dropped})
            elif name == "cancel_drop":
                if not pending_drop:
                    errors.append("cancel_drop with no pending drop")
                pending_drop = None
                return json.dumps({"cancelled": True})
            elif name == "run_query":
                if not connected:
                    errors.append("run_query before connecting")
                    return json.dumps({"error": "Not connected"})
                sql = inp["sql"].strip()
                if not sql.upper().startswith("SELECT"):
                    return json.dumps({"error": "Only SELECT allowed"})
                rows = conn.execute(sql).fetchall()
                return json.dumps({"rows": [dict(r) for r in rows]})
            elif name == "disconnect":
                connected = False
                selected_table = None
                return json.dumps({"disconnected": True})
            else:
                return json.dumps({"result": "ok"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    msgs = [{"role": "user", "content": PROMPT}]

    for turn in range(MAX_TURNS):
        tools_per_turn.append({
            "turn": turn + 1,
            "tool_count": len(FLAT_TOOLS),
            "tools": [t["name"] for t in FLAT_TOOLS],
        })

        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=FLAT_TOOLS,
            system=FLAT_SYSTEM, messages=msgs,
        )
        msgs.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            tool_log.append(tu.name)
            result = handle_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
        msgs.append({"role": "user", "content": tool_results})

    conn.close()
    return {
        "tool_log": tool_log,
        "tools_per_turn": tools_per_turn,
        "errors": errors,
        "total_calls": len(tool_log),
    }


# ============================================================
# APPROACH 2: HATEOAS dynamic tool discovery
# ============================================================

def run_hateoas(db_path):
    """Run with HATEOAS — tools discovered dynamically per state."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from hateoas_agent import StateMachine, Runner
    from hateoas_agent.registry import Registry

    # Build the same database admin state machine
    sm = StateMachine("database", gateway_name="explore_database")
    sm.gateway(description="Connect to a SQLite database or list available databases", params={"database_name": "string"})

    sm.state("connected", actions=[
        {"name": "list_tables", "description": "List all tables", "params": {}},
        {"name": "select_table", "description": "Select a table", "params": {"table_name": "string"}, "required": ["table_name"]},
        {"name": "create_table", "description": "Create a table", "params": {"table_name": "string", "columns": "string"}, "required": ["table_name", "columns"]},
        {"name": "run_query", "description": "Run SELECT query", "params": {"sql": "string"}, "required": ["sql"]},
        {"name": "disconnect", "description": "Disconnect", "params": {}},
    ])

    sm.state("table_selected", actions=[
        {"name": "describe_table", "description": "Show table schema", "params": {}},
        {"name": "sample_rows", "description": "Preview rows", "params": {"limit": "integer"}},
        {"name": "insert_row", "description": "Insert a row", "params": {"values": "string"}, "required": ["values"]},
        {"name": "update_rows", "description": "Update rows", "params": {"set_clause": "string", "where_clause": "string"}, "required": ["set_clause", "where_clause"]},
        {"name": "delete_rows", "description": "Delete rows", "params": {"where_clause": "string"}, "required": ["where_clause"]},
        {"name": "add_column", "description": "Add column", "params": {"column_def": "string"}, "required": ["column_def"]},
        {"name": "drop_table", "description": "Drop table (requires confirmation)", "params": {}},
        {"name": "select_table", "description": "Switch table", "params": {"table_name": "string"}, "required": ["table_name"]},
        {"name": "list_tables", "description": "Back to table list", "params": {}},
        {"name": "run_query", "description": "Run SELECT query", "params": {"sql": "string"}, "required": ["sql"]},
    ])

    sm.state("confirm_drop", actions=[
        {"name": "confirm_drop", "description": "Confirm and permanently drop", "params": {}},
        {"name": "cancel_drop", "description": "Cancel the drop", "params": {}},
    ])

    sm.state("disconnected")

    # Session state
    session = {"conn": None, "selected_table": None, "pending_drop": None}

    @sm.on_gateway
    def handle_explore(database_name=None):
        if not database_name:
            return {"databases": ["company.db"]}
        path = db_path
        session["conn"] = sqlite3.connect(path)
        session["conn"].row_factory = sqlite3.Row
        tables = session["conn"].execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {"connected": True, "tables": [r["name"] for r in tables], "_state": "connected"}

    @sm.on_action("list_tables")
    def handle_list():
        tables = session["conn"].execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        session["selected_table"] = None
        return {"tables": [r["name"] for r in tables], "_state": "connected"}

    @sm.on_action("select_table")
    def handle_select(table_name):
        session["selected_table"] = table_name
        cols = session["conn"].execute(f'PRAGMA table_info("{table_name}")').fetchall()
        return {"selected": table_name, "columns": [dict(c) for c in cols], "_state": "table_selected"}

    @sm.on_action("describe_table")
    def handle_describe():
        t = session["selected_table"]
        cols = session["conn"].execute(f'PRAGMA table_info("{t}")').fetchall()
        return {"table": t, "columns": [dict(c) for c in cols], "_state": "table_selected"}

    @sm.on_action("sample_rows")
    def handle_sample(limit=10):
        t = session["selected_table"]
        if isinstance(limit, str):
            limit = int(limit)
        rows = session["conn"].execute(f'SELECT * FROM "{t}" LIMIT ?', (limit,)).fetchall()
        return {"rows": [dict(r) for r in rows], "_state": "table_selected"}

    @sm.on_action("insert_row")
    def handle_insert(values):
        import json as jm
        t = session["selected_table"]
        if isinstance(values, str):
            values = jm.loads(values)
        cols = ", ".join(f'"{k}"' for k in values.keys())
        phs = ", ".join("?" for _ in values)
        session["conn"].execute(f'INSERT INTO "{t}" ({cols}) VALUES ({phs})', list(values.values()))
        session["conn"].commit()
        return {"inserted": True, "_state": "table_selected"}

    @sm.on_action("update_rows")
    def handle_update(set_clause, where_clause):
        t = session["selected_table"]
        cur = session["conn"].execute(f'UPDATE "{t}" SET {set_clause} WHERE {where_clause}')
        session["conn"].commit()
        return {"rows_affected": cur.rowcount, "_state": "table_selected"}

    @sm.on_action("delete_rows")
    def handle_delete(where_clause):
        t = session["selected_table"]
        cur = session["conn"].execute(f'DELETE FROM "{t}" WHERE {where_clause}')
        session["conn"].commit()
        return {"rows_deleted": cur.rowcount, "_state": "table_selected"}

    @sm.on_action("add_column")
    def handle_add_col(column_def):
        t = session["selected_table"]
        session["conn"].execute(f'ALTER TABLE "{t}" ADD COLUMN {column_def}')
        session["conn"].commit()
        return {"added": column_def, "_state": "table_selected"}

    @sm.on_action("create_table")
    def handle_create(table_name, columns):
        session["conn"].execute(f'CREATE TABLE "{table_name}" ({columns})')
        session["conn"].commit()
        return {"created": table_name, "_state": "connected"}

    @sm.on_action("run_query")
    def handle_query(sql):
        if not sql.strip().upper().startswith("SELECT"):
            state = "table_selected" if session["selected_table"] else "connected"
            return {"error": "Only SELECT allowed", "_state": state}
        rows = session["conn"].execute(sql).fetchall()
        state = "table_selected" if session["selected_table"] else "connected"
        return {"rows": [dict(r) for r in rows], "_state": state}

    @sm.on_action("drop_table")
    def handle_drop():
        t = session["selected_table"]
        session["pending_drop"] = t
        count = session["conn"].execute(f'SELECT COUNT(*) as c FROM "{t}"').fetchone()
        return {"warning": f"About to drop {t} ({count['c']} rows). Confirm?", "_state": "confirm_drop"}

    @sm.on_action("confirm_drop")
    def handle_confirm():
        t = session["pending_drop"]
        session["conn"].execute(f'DROP TABLE "{t}"')
        session["conn"].commit()
        session["selected_table"] = None
        session["pending_drop"] = None
        return {"dropped": t, "_state": "connected"}

    @sm.on_action("cancel_drop")
    def handle_cancel():
        t = session["pending_drop"]
        session["pending_drop"] = None
        return {"cancelled": True, "table": t, "_state": "table_selected"}

    @sm.on_action("disconnect")
    def handle_disconnect():
        if session["conn"]:
            session["conn"].close()
        return {"disconnected": True, "_state": "disconnected"}

    # Track tools per turn
    tool_log = []
    tools_per_turn = []
    reg = Registry(sm)

    def on_tool(name, inp):
        tool_log.append(name)

    # We need to intercept the runner to log tools per turn.
    # Use the runner but also track tool schemas.
    original_run = Runner.run

    runner = Runner(
        sm, model=MODEL,
        system="You are a database admin assistant. Help users manage SQLite databases safely.",
        on_tool_call=on_tool,
    )

    # Patch to capture tools per turn
    original_create = runner._client.messages.create
    turn_counter = [0]

    def tracking_create(**kwargs):
        turn_counter[0] += 1
        tools_in_call = kwargs.get("tools", [])
        tools_per_turn.append({
            "turn": turn_counter[0],
            "tool_count": len(tools_in_call),
            "tools": [t["name"] for t in tools_in_call],
        })
        return original_create(**kwargs)

    runner._client.messages.create = tracking_create

    result = runner.run(PROMPT)

    if session["conn"]:
        session["conn"].close()

    return {
        "tool_log": tool_log,
        "tools_per_turn": tools_per_turn,
        "errors": [],
        "total_calls": len(tool_log),
    }


# ============================================================
# Report
# ============================================================

def print_report(flat_result, hateoas_result):
    print("\n" + "=" * 70)
    print("COMPARISON: Flat Tool Registration vs HATEOAS Dynamic Discovery")
    print("=" * 70)
    print(f"\nPrompt: \"{PROMPT}\"")

    print("\n" + "-" * 70)
    print("FLAT APPROACH (all 15 tools registered upfront)")
    print("-" * 70)
    print(f"Total tool calls: {flat_result['total_calls']}")
    print(f"Tool call sequence: {' -> '.join(flat_result['tool_log'])}")
    if flat_result["errors"]:
        print(f"Out-of-order errors: {len(flat_result['errors'])}")
        for e in flat_result["errors"]:
            print(f"  ! {e}")
    print("\nTools available to Claude per turn:")
    for t in flat_result["tools_per_turn"]:
        print(f"  Turn {t['turn']}: {t['tool_count']} tools (always all 15)")

    print("\n" + "-" * 70)
    print("HATEOAS APPROACH (tools discovered dynamically)")
    print("-" * 70)
    print(f"Total tool calls: {hateoas_result['total_calls']}")
    print(f"Tool call sequence: {' -> '.join(hateoas_result['tool_log'])}")
    print("\nTools available to Claude per turn:")
    for t in hateoas_result["tools_per_turn"]:
        print(f"  Turn {t['turn']}: {t['tool_count']} tools -> [{', '.join(t['tools'])}]")

    print("\n" + "-" * 70)
    print("KEY DIFFERENCES")
    print("-" * 70)

    flat_max = max(t["tool_count"] for t in flat_result["tools_per_turn"]) if flat_result["tools_per_turn"] else 0
    hateoas_avg = sum(t["tool_count"] for t in hateoas_result["tools_per_turn"]) / len(hateoas_result["tools_per_turn"]) if hateoas_result["tools_per_turn"] else 0
    hateoas_max = max(t["tool_count"] for t in hateoas_result["tools_per_turn"]) if hateoas_result["tools_per_turn"] else 0

    print(f"  Tools per turn (flat):    always {flat_max}")
    print(f"  Tools per turn (HATEOAS): avg {hateoas_avg:.1f}, max {hateoas_max}")
    print(f"  Flat out-of-order errors: {len(flat_result['errors'])}")
    print(f"  HATEOAS impossible errors: 0 (invalid actions rejected by framework)")
    print()
    print("  With FLAT registration:")
    print("    - Claude sees confirm_drop before any drop is pending")
    print("    - Claude sees insert_row/update_rows before any table is selected")
    print("    - Claude sees disconnect before connecting")
    print("    - Nothing prevents calling tools in the wrong order")
    print()
    print("  With HATEOAS:")
    print("    - Claude only sees tools valid for the current state")
    print("    - drop_table only appears after selecting a table")
    print("    - confirm_drop only appears after requesting a drop")
    print("    - State transitions are enforced server-side")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Setting up sample databases...")
    db_path = create_sample_db()

    print(f"\nRunning FLAT approach (all 15 tools)...")
    flat_result = run_flat(db_path)

    # Recreate DB for clean comparison
    db_path = create_sample_db()

    print(f"Running HATEOAS approach (dynamic tools)...")
    hateoas_result = run_hateoas(db_path)

    print_report(flat_result, hateoas_result)

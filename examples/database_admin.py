"""
HATEOAS Agent — Database Admin example using the declarative StateMachine API.

Demonstrates dynamic tool discovery for database administration with SQLite.
The agent navigates: connect → list tables → select table → operate,
with destructive actions (DROP TABLE) requiring explicit confirmation.

This showcases why HATEOAS matters: 15 possible actions would be unwieldy
as a flat tool list, but state-gating makes them discoverable and safe.
"""

import sqlite3
import os
import tempfile

from hateoas_agent import StateMachine, Runner

# ============================================================
# State machine definition
# ============================================================

db = StateMachine("database", gateway_name="explore_database")

db.gateway(
    description="Connect to a SQLite database. Lists available databases if no name given.",
    params={"database_name": "string"},
)

# --- Connected to a database, can browse tables ---

db.state("connected", actions=[
    {"name": "list_tables", "description": "List all tables in the connected database"},
    {"name": "select_table", "description": "Select a table to work with", "params": {"table_name": "string"}, "required": ["table_name"]},
    {"name": "create_table", "description": "Create a new table", "params": {"table_name": "string", "columns": "string (comma-separated column definitions, e.g. 'name TEXT, age INTEGER')"}, "required": ["table_name", "columns"]},
    {"name": "run_query", "description": "Run a read-only SQL query (SELECT only)", "params": {"sql": "string"}, "required": ["sql"]},
    {"name": "disconnect", "description": "Disconnect from the current database"},
])

# --- Working with a specific table ---

db.state("table_selected", actions=[
    {"name": "describe_table", "description": "Show the schema and column info for the selected table"},
    {"name": "sample_rows", "description": "Preview rows from the selected table", "params": {"limit": "integer"}},
    {"name": "insert_row", "description": "Insert a new row into the selected table", "params": {"values": "string (JSON object of column:value pairs, e.g. '{\"name\": \"Alice\", \"age\": 30}')"}, "required": ["values"]},
    {"name": "update_rows", "description": "Update rows matching a condition", "params": {"set_clause": "string (e.g. 'salary = 75000')", "where_clause": "string (e.g. 'id = 3')"}, "required": ["set_clause", "where_clause"]},
    {"name": "delete_rows", "description": "Delete rows matching a condition", "params": {"where_clause": "string (e.g. 'id = 3')"}, "required": ["where_clause"]},
    {"name": "add_column", "description": "Add a new column to the selected table", "params": {"column_def": "string (e.g. 'email TEXT')"}, "required": ["column_def"]},
    {"name": "drop_table", "description": "Request to drop (delete) the selected table. Requires confirmation."},
    {"name": "select_table", "description": "Switch to a different table", "params": {"table_name": "string"}, "required": ["table_name"]},
    {"name": "list_tables", "description": "Go back to the table list"},
    {"name": "run_query", "description": "Run a read-only SQL query (SELECT only)", "params": {"sql": "string"}, "required": ["sql"]},
])

# --- Confirmation before destructive DROP ---

db.state("confirm_drop", actions=[
    {"name": "confirm_drop", "description": "Confirm and permanently drop the table. This cannot be undone."},
    {"name": "cancel_drop", "description": "Cancel the drop and go back to the table"},
])

# --- Disconnected (must call gateway again) ---

db.state("disconnected")

# ============================================================
# Session state
# ============================================================

session = {
    "conn": None,
    "db_path": None,
    "db_name": None,
    "selected_table": None,
    "pending_drop_table": None,
}

# ============================================================
# Sample database setup
# ============================================================

SAMPLE_DB_DIR = os.path.join(tempfile.gettempdir(), "hateoas_db_demo")


def create_sample_database():
    """Create a sample company database for the demo."""
    os.makedirs(SAMPLE_DB_DIR, exist_ok=True)
    db_path = os.path.join(SAMPLE_DB_DIR, "company.db")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS departments")
    c.execute("DROP TABLE IF EXISTS employees")
    c.execute("DROP TABLE IF EXISTS projects")

    c.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            role TEXT NOT NULL,
            salary REAL NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            status TEXT NOT NULL,
            deadline TEXT
        )
    """)

    c.executemany("INSERT INTO departments VALUES (?, ?, ?)", [
        (1, "Engineering", 500000),
        (2, "Marketing", 200000),
        (3, "Sales", 300000),
        (4, "HR", 150000),
    ])

    c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", [
        (1, "Alice Chen", 1, "Senior Engineer", 145000),
        (2, "Bob Martinez", 1, "Staff Engineer", 175000),
        (3, "Carol White", 2, "Marketing Manager", 110000),
        (4, "Dave Kim", 3, "Sales Lead", 95000),
        (5, "Eve Johnson", 1, "Junior Engineer", 85000),
        (6, "Frank Lee", 4, "HR Director", 130000),
        (7, "Grace Park", 2, "Content Strategist", 88000),
        (8, "Hank Brown", 3, "Account Executive", 92000),
    ])

    c.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", [
        (1, "Platform Rewrite", 1, "in_progress", "2026-06-30"),
        (2, "Q2 Campaign", 2, "planning", "2026-04-15"),
        (3, "CRM Integration", 3, "in_progress", "2026-05-01"),
        (4, "New Hire Onboarding v2", 4, "completed", "2026-02-28"),
        (5, "Mobile App", 1, "planning", "2026-09-01"),
    ])

    conn.commit()
    conn.close()
    return db_path


# ============================================================
# Handlers
# ============================================================

@db.on_gateway
def handle_explore(database_name=None):
    if not database_name:
        # List available databases
        databases = []
        if os.path.isdir(SAMPLE_DB_DIR):
            databases = [f for f in os.listdir(SAMPLE_DB_DIR) if f.endswith(".db")]
        if not databases:
            return {"message": "No databases found. Create a .db file first."}
        return {"available_databases": databases, "directory": SAMPLE_DB_DIR}

    # Connect to specified database
    db_path = database_name
    if not os.path.isabs(database_name):
        db_path = os.path.join(SAMPLE_DB_DIR, database_name)
    if not db_path.endswith(".db"):
        db_path += ".db"

    if not os.path.exists(db_path):
        return {"error": f"Database not found: {db_path}"}

    session["conn"] = sqlite3.connect(db_path)
    session["conn"].row_factory = sqlite3.Row
    session["db_path"] = db_path
    session["db_name"] = os.path.basename(db_path)
    session["selected_table"] = None

    tables = session["conn"].execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    return {
        "connected": True,
        "database": session["db_name"],
        "tables": [r["name"] for r in tables],
        "table_count": len(tables),
        "_state": "connected",
    }


@db.on_action("list_tables")
def handle_list_tables():
    tables = session["conn"].execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_info = []
    for t in tables:
        count = session["conn"].execute(f'SELECT COUNT(*) as cnt FROM "{t["name"]}"').fetchone()
        table_info.append({"name": t["name"], "row_count": count["cnt"]})

    session["selected_table"] = None
    return {
        "database": session["db_name"],
        "tables": table_info,
        "_state": "connected",
    }


@db.on_action("select_table")
def handle_select_table(table_name):
    # Verify table exists
    exists = session["conn"].execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    if not exists:
        tables = session["conn"].execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {
            "error": f"Table '{table_name}' not found",
            "available_tables": [r["name"] for r in tables],
            "_state": "connected",
        }

    session["selected_table"] = table_name

    # Get schema info
    columns = session["conn"].execute(f'PRAGMA table_info("{table_name}")').fetchall()
    count = session["conn"].execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"').fetchone()

    return {
        "selected_table": table_name,
        "columns": [{"name": c["name"], "type": c["type"], "notnull": bool(c["notnull"]), "pk": bool(c["pk"])} for c in columns],
        "row_count": count["cnt"],
        "_state": "table_selected",
    }


@db.on_action("describe_table")
def handle_describe():
    table_name = session["selected_table"]
    columns = session["conn"].execute(f'PRAGMA table_info("{table_name}")').fetchall()
    fk = session["conn"].execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
    indexes = session["conn"].execute(f'PRAGMA index_list("{table_name}")').fetchall()

    return {
        "table": table_name,
        "columns": [{"name": c["name"], "type": c["type"], "notnull": bool(c["notnull"]), "pk": bool(c["pk"]), "default": c["dflt_value"]} for c in columns],
        "foreign_keys": [{"column": f["from"], "references": f"{f['table']}({f['to']})"} for f in fk],
        "indexes": [{"name": i["name"], "unique": bool(i["unique"])} for i in indexes],
        "_state": "table_selected",
    }


@db.on_action("sample_rows")
def handle_sample(limit=10):
    table_name = session["selected_table"]
    if isinstance(limit, str):
        limit = int(limit)
    limit = min(limit, 100)

    rows = session["conn"].execute(f'SELECT * FROM "{table_name}" LIMIT ?', (limit,)).fetchall()
    columns = [desc[0] for desc in session["conn"].execute(f'SELECT * FROM "{table_name}" LIMIT 0').description]

    return {
        "table": table_name,
        "columns": columns,
        "rows": [dict(r) for r in rows],
        "row_count": len(rows),
        "_state": "table_selected",
    }


@db.on_action("insert_row")
def handle_insert(values):
    import json as json_mod
    table_name = session["selected_table"]

    if isinstance(values, str):
        values = json_mod.loads(values)

    columns = ", ".join(f'"{k}"' for k in values.keys())
    placeholders = ", ".join("?" for _ in values)
    sql = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

    cursor = session["conn"].execute(sql, list(values.values()))
    session["conn"].commit()

    return {
        "success": True,
        "table": table_name,
        "inserted_id": cursor.lastrowid,
        "values": values,
        "_state": "table_selected",
    }


@db.on_action("update_rows")
def handle_update(set_clause, where_clause):
    table_name = session["selected_table"]

    sql = f'UPDATE "{table_name}" SET {set_clause} WHERE {where_clause}'
    cursor = session["conn"].execute(sql)
    session["conn"].commit()

    return {
        "success": True,
        "table": table_name,
        "rows_affected": cursor.rowcount,
        "set": set_clause,
        "where": where_clause,
        "_state": "table_selected",
    }


@db.on_action("delete_rows")
def handle_delete(where_clause):
    table_name = session["selected_table"]

    # Safety: count affected rows first
    count = session["conn"].execute(
        f'SELECT COUNT(*) as cnt FROM "{table_name}" WHERE {where_clause}'
    ).fetchone()

    cursor = session["conn"].execute(f'DELETE FROM "{table_name}" WHERE {where_clause}')
    session["conn"].commit()

    return {
        "success": True,
        "table": table_name,
        "rows_deleted": cursor.rowcount,
        "where": where_clause,
        "_state": "table_selected",
    }


@db.on_action("add_column")
def handle_add_column(column_def):
    table_name = session["selected_table"]
    session["conn"].execute(f'ALTER TABLE "{table_name}" ADD COLUMN {column_def}')
    session["conn"].commit()

    columns = session["conn"].execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return {
        "success": True,
        "table": table_name,
        "added": column_def,
        "columns": [{"name": c["name"], "type": c["type"]} for c in columns],
        "_state": "table_selected",
    }


@db.on_action("create_table")
def handle_create_table(table_name, columns):
    session["conn"].execute(f'CREATE TABLE "{table_name}" ({columns})')
    session["conn"].commit()

    return {
        "success": True,
        "created_table": table_name,
        "columns": columns,
        "_state": "connected",
    }


@db.on_action("run_query")
def handle_run_query(sql):
    # Safety: only allow SELECT statements
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed. Use the dedicated actions for INSERT, UPDATE, and DELETE.",
            "_state": session.get("_current_state", "connected"),
        }

    rows = session["conn"].execute(sql).fetchall()
    columns = [desc[0] for desc in session["conn"].execute(sql).description]

    # Preserve current state (connected or table_selected)
    current_state = "table_selected" if session["selected_table"] else "connected"
    return {
        "columns": columns,
        "rows": [dict(r) for r in rows],
        "row_count": len(rows),
        "_state": current_state,
    }


@db.on_action("drop_table")
def handle_drop_table():
    table_name = session["selected_table"]
    session["pending_drop_table"] = table_name

    count = session["conn"].execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"').fetchone()
    return {
        "warning": f"You are about to permanently drop table '{table_name}' with {count['cnt']} rows. This cannot be undone.",
        "table": table_name,
        "row_count": count["cnt"],
        "action_required": "Use confirm_drop to proceed or cancel_drop to abort.",
        "_state": "confirm_drop",
    }


@db.on_action("confirm_drop")
def handle_confirm_drop():
    table_name = session["pending_drop_table"]
    session["conn"].execute(f'DROP TABLE "{table_name}"')
    session["conn"].commit()

    session["selected_table"] = None
    session["pending_drop_table"] = None

    return {
        "success": True,
        "dropped_table": table_name,
        "message": f"Table '{table_name}' has been permanently dropped.",
        "_state": "connected",
    }


@db.on_action("cancel_drop")
def handle_cancel_drop():
    table_name = session["pending_drop_table"]
    session["pending_drop_table"] = None

    return {
        "cancelled": True,
        "table": table_name,
        "message": f"Drop cancelled. Still working with table '{table_name}'.",
        "_state": "table_selected",
    }


@db.on_action("disconnect")
def handle_disconnect():
    if session["conn"]:
        session["conn"].close()
    db_name = session["db_name"]
    session["conn"] = None
    session["db_path"] = None
    session["db_name"] = None
    session["selected_table"] = None

    return {
        "disconnected": True,
        "message": f"Disconnected from {db_name}.",
        "_state": "disconnected",
    }


# ============================================================
# Run scenarios
# ============================================================

if __name__ == "__main__":
    import json

    # Set up sample data
    print("Setting up sample database...")
    sample_path = create_sample_database()
    print(f"Created: {sample_path}\n")

    def on_tool(name, inp):
        print(f"  -> {name}({json.dumps(inp)[:120]})")

    def on_text(text):
        print(f"  Agent: {text[:300]}")

    runner = Runner(
        db,
        model="claude-sonnet-4-20250514",
        system=(
            "You are a database administration assistant. "
            "Help users explore and manage their SQLite databases safely."
        ),
        on_tool_call=on_tool,
        on_text=on_text,
    )

    # Scenario 1: Explore and query
    print("=" * 60)
    print("SCENARIO 1: Explore database structure")
    print("=" * 60)
    result = runner.run(
        "Connect to the company database. Show me all tables, then "
        "look at the employees table — show me the schema and a few sample rows."
    )
    print(f"\nTool calls: {len(result.tool_calls)} "
          f"(gateway: {result.gateway_calls}, dynamic: {result.dynamic_calls})")
    print(f"Tools used: {result.unique_tools}")

    # Scenario 2: Cross-table query
    print("\n" + "=" * 60)
    print("SCENARIO 2: Cross-table analysis")
    print("=" * 60)
    result = runner.run(
        "Show me all engineers and their department budgets. "
        "Use a JOIN query."
    )
    print(f"\nTool calls: {len(result.tool_calls)} "
          f"(gateway: {result.gateway_calls}, dynamic: {result.dynamic_calls})")

    # Scenario 3: Modifications
    print("\n" + "=" * 60)
    print("SCENARIO 3: Data modification")
    print("=" * 60)
    result = runner.run(
        "Add a new employee: Iris Wang, Engineering department (id 1), "
        "role 'ML Engineer', salary 135000. Then show me the updated "
        "employees table."
    )
    print(f"\nTool calls: {len(result.tool_calls)} "
          f"(gateway: {result.gateway_calls}, dynamic: {result.dynamic_calls})")

    # Scenario 4: Destructive action with confirmation
    print("\n" + "=" * 60)
    print("SCENARIO 4: Drop table (confirmation flow)")
    print("=" * 60)
    result = runner.run(
        "I want to drop the projects table. Go ahead and do it."
    )
    print(f"\nTool calls: {len(result.tool_calls)} "
          f"(gateway: {result.gateway_calls}, dynamic: {result.dynamic_calls})")
    print(f"Tools used: {result.unique_tools}")

    # Cleanup
    if session["conn"]:
        session["conn"].close()
    print("\nDone.")

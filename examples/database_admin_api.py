"""
HATEOAS Agent — Database Admin as a FastAPI service.

Demonstrates how to wrap a hateoas-agent in a real HTTP API with session
management. Each conversation gets its own isolated state machine, SQLite
connection, and message history.

Run:
    pip install fastapi uvicorn
    ANTHROPIC_API_KEY="sk-..." python examples/database_admin_api.py

Then in another terminal:
    # Start a conversation
    curl -X POST localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"message": "Connect to the company database and show me the tables"}'

    # Continue it (use session_id from the response)
    curl -X POST localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"message": "Select the employees table and show me the schema", "session_id": "..."}'

    # List active sessions
    curl localhost:8000/sessions

    # Clean up
    curl -X DELETE localhost:8000/sessions/<session_id>
"""

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hateoas_agent import Runner, StateMachine

# ============================================================
# Sample database setup
# ============================================================

SAMPLE_DB_DIR = os.path.join(tempfile.gettempdir(), "hateoas_api_demo")


def create_sample_database():
    os.makedirs(SAMPLE_DB_DIR, exist_ok=True)
    db_path = os.path.join(SAMPLE_DB_DIR, "company.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS departments")
    c.execute("DROP TABLE IF EXISTS employees")
    c.execute("DROP TABLE IF EXISTS projects")
    c.execute("""CREATE TABLE departments (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, budget REAL NOT NULL
    )""")
    c.execute("""CREATE TABLE employees (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL,
        department_id INTEGER REFERENCES departments(id),
        role TEXT NOT NULL, salary REAL NOT NULL
    )""")
    c.execute("""CREATE TABLE projects (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL,
        department_id INTEGER REFERENCES departments(id),
        status TEXT NOT NULL, deadline TEXT
    )""")
    c.executemany("INSERT INTO departments VALUES (?, ?, ?)", [
        (1, "Engineering", 500000), (2, "Marketing", 200000),
        (3, "Sales", 300000), (4, "HR", 150000),
    ])
    c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", [
        (1, "Alice Chen", 1, "Senior Engineer", 145000),
        (2, "Bob Martinez", 1, "Staff Engineer", 175000),
        (3, "Carol White", 2, "Marketing Manager", 110000),
        (4, "Dave Kim", 3, "Sales Lead", 95000),
        (5, "Eve Johnson", 1, "Junior Engineer", 85000),
        (6, "Frank Lee", 4, "HR Director", 130000),
    ])
    c.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", [
        (1, "Platform Rewrite", 1, "in_progress", "2026-06-30"),
        (2, "Q2 Campaign", 2, "planning", "2026-04-15"),
        (3, "CRM Integration", 3, "in_progress", "2026-05-01"),
    ])
    conn.commit()
    conn.close()
    return db_path


# ============================================================
# Session factory — each session gets its own state machine
# ============================================================


def create_session():
    """Create an isolated state machine + runner for a new conversation."""

    sm = StateMachine("database", gateway_name="explore_database")
    sm.gateway(
        description="Connect to a SQLite database or list available databases.",
        params={"database_name": "string"},
    )

    sm.state("connected", actions=[
        {"name": "list_tables", "description": "List all tables in the database"},
        {"name": "select_table", "description": "Select a table to work with",
         "params": {"table_name": "string"}, "required": ["table_name"]},
        {"name": "create_table", "description": "Create a new table",
         "params": {"table_name": "string", "columns": "string"},
         "required": ["table_name", "columns"]},
        {"name": "run_query", "description": "Run a read-only SQL query (SELECT only)",
         "params": {"sql": "string"}, "required": ["sql"]},
        {"name": "disconnect", "description": "Disconnect from the database"},
    ])

    sm.state("table_selected", actions=[
        {"name": "describe_table", "description": "Show schema and column info"},
        {"name": "sample_rows", "description": "Preview rows from the table",
         "params": {"limit": "integer"}},
        {"name": "insert_row", "description": "Insert a new row",
         "params": {"values": "string"}, "required": ["values"]},
        {"name": "update_rows", "description": "Update rows matching a condition",
         "params": {"set_clause": "string", "where_clause": "string"},
         "required": ["set_clause", "where_clause"]},
        {"name": "delete_rows", "description": "Delete rows matching a condition",
         "params": {"where_clause": "string"}, "required": ["where_clause"]},
        {"name": "add_column", "description": "Add a column to the table",
         "params": {"column_def": "string"}, "required": ["column_def"]},
        {"name": "drop_table", "description": "Drop the table (requires confirmation)"},
        {"name": "select_table", "description": "Switch to a different table",
         "params": {"table_name": "string"}, "required": ["table_name"]},
        {"name": "list_tables", "description": "Go back to the table list"},
        {"name": "run_query", "description": "Run a read-only SQL query (SELECT only)",
         "params": {"sql": "string"}, "required": ["sql"]},
    ])

    sm.state("confirm_drop", actions=[
        {"name": "confirm_drop", "description": "Confirm and permanently drop the table"},
        {"name": "cancel_drop", "description": "Cancel and go back"},
    ])

    sm.state("disconnected")

    # Per-session state
    ctx = {"conn": None, "selected_table": None, "pending_drop": None}

    @sm.on_gateway
    def handle_explore(database_name=None):
        if not database_name:
            dbs = [f for f in os.listdir(SAMPLE_DB_DIR) if f.endswith(".db")]
            return {"available_databases": dbs}
        path = database_name
        if not os.path.isabs(database_name):
            path = os.path.join(SAMPLE_DB_DIR, database_name)
        if not path.endswith(".db"):
            path += ".db"
        if not os.path.exists(path):
            return {"error": f"Database not found: {path}"}
        ctx["conn"] = sqlite3.connect(path)
        ctx["conn"].row_factory = sqlite3.Row
        tables = ctx["conn"].execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {"connected": True, "tables": [r["name"] for r in tables], "_state": "connected"}

    @sm.on_action("list_tables")
    def handle_list():
        tables = ctx["conn"].execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        info = []
        for t in tables:
            count = ctx["conn"].execute(f'SELECT COUNT(*) as c FROM "{t["name"]}"').fetchone()
            info.append({"name": t["name"], "row_count": count["c"]})
        ctx["selected_table"] = None
        return {"tables": info, "_state": "connected"}

    @sm.on_action("select_table")
    def handle_select(table_name):
        exists = ctx["conn"].execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()
        if not exists:
            return {"error": f"Table '{table_name}' not found", "_state": "connected"}
        ctx["selected_table"] = table_name
        cols = ctx["conn"].execute(f'PRAGMA table_info("{table_name}")').fetchall()
        count = ctx["conn"].execute(f'SELECT COUNT(*) as c FROM "{table_name}"').fetchone()
        return {
            "selected_table": table_name,
            "columns": [{"name": c["name"], "type": c["type"], "pk": bool(c["pk"])} for c in cols],
            "row_count": count["c"],
            "_state": "table_selected",
        }

    @sm.on_action("describe_table")
    def handle_describe():
        t = ctx["selected_table"]
        cols = ctx["conn"].execute(f'PRAGMA table_info("{t}")').fetchall()
        return {
            "table": t,
            "columns": [{"name": c["name"], "type": c["type"],
                         "notnull": bool(c["notnull"]), "pk": bool(c["pk"])} for c in cols],
            "_state": "table_selected",
        }

    @sm.on_action("sample_rows")
    def handle_sample(limit=10):
        t = ctx["selected_table"]
        if isinstance(limit, str):
            limit = int(limit)
        rows = ctx["conn"].execute(f'SELECT * FROM "{t}" LIMIT ?', (min(limit, 100),)).fetchall()
        return {"table": t, "rows": [dict(r) for r in rows], "_state": "table_selected"}

    @sm.on_action("insert_row")
    def handle_insert(values):
        import json
        t = ctx["selected_table"]
        if isinstance(values, str):
            values = json.loads(values)
        cols = ", ".join(f'"{k}"' for k in values.keys())
        phs = ", ".join("?" for _ in values)
        cur = ctx["conn"].execute(
            f'INSERT INTO "{t}" ({cols}) VALUES ({phs})', list(values.values())
        )
        ctx["conn"].commit()
        return {"inserted_id": cur.lastrowid, "_state": "table_selected"}

    @sm.on_action("update_rows")
    def handle_update(set_clause, where_clause):
        t = ctx["selected_table"]
        cur = ctx["conn"].execute(f'UPDATE "{t}" SET {set_clause} WHERE {where_clause}')
        ctx["conn"].commit()
        return {"rows_affected": cur.rowcount, "_state": "table_selected"}

    @sm.on_action("delete_rows")
    def handle_delete(where_clause):
        t = ctx["selected_table"]
        cur = ctx["conn"].execute(f'DELETE FROM "{t}" WHERE {where_clause}')
        ctx["conn"].commit()
        return {"rows_deleted": cur.rowcount, "_state": "table_selected"}

    @sm.on_action("add_column")
    def handle_add_col(column_def):
        t = ctx["selected_table"]
        ctx["conn"].execute(f'ALTER TABLE "{t}" ADD COLUMN {column_def}')
        ctx["conn"].commit()
        return {"added": column_def, "_state": "table_selected"}

    @sm.on_action("create_table")
    def handle_create(table_name, columns):
        ctx["conn"].execute(f'CREATE TABLE "{table_name}" ({columns})')
        ctx["conn"].commit()
        return {"created": table_name, "_state": "connected"}

    @sm.on_action("run_query")
    def handle_query(sql):
        if not sql.strip().upper().startswith("SELECT"):
            state = "table_selected" if ctx["selected_table"] else "connected"
            return {"error": "Only SELECT queries are allowed", "_state": state}
        rows = ctx["conn"].execute(sql).fetchall()
        state = "table_selected" if ctx["selected_table"] else "connected"
        return {"rows": [dict(r) for r in rows], "_state": state}

    @sm.on_action("drop_table")
    def handle_drop():
        t = ctx["selected_table"]
        ctx["pending_drop"] = t
        count = ctx["conn"].execute(f'SELECT COUNT(*) as c FROM "{t}"').fetchone()
        return {
            "warning": f"About to drop '{t}' ({count['c']} rows). This cannot be undone.",
            "_state": "confirm_drop",
        }

    @sm.on_action("confirm_drop")
    def handle_confirm():
        t = ctx["pending_drop"]
        ctx["conn"].execute(f'DROP TABLE "{t}"')
        ctx["conn"].commit()
        ctx["selected_table"] = None
        ctx["pending_drop"] = None
        return {"dropped": t, "_state": "connected"}

    @sm.on_action("cancel_drop")
    def handle_cancel():
        t = ctx["pending_drop"]
        ctx["pending_drop"] = None
        return {"cancelled": True, "table": t, "_state": "table_selected"}

    @sm.on_action("disconnect")
    def handle_disconnect():
        if ctx["conn"]:
            ctx["conn"].close()
            ctx["conn"] = None
        return {"disconnected": True, "_state": "disconnected"}

    tool_log = []
    runner = Runner(
        sm,
        model="claude-sonnet-4-20250514",
        system="You are a database admin assistant. Help users explore and manage SQLite databases safely.",
        on_tool_call=lambda name, inp: tool_log.append({"tool": name, "input": inp}),
    )

    return {
        "sm": sm,
        "runner": runner,
        "ctx": ctx,
        "tool_log": tool_log,
        "messages": [],
        "turns": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="HATEOAS Database Admin Agent",
    description="AI-powered database administration with dynamic tool discovery",
)

sessions: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: list
    state: Optional[str]
    available_actions: list[str]
    turn_count: int


@app.on_event("startup")
def startup():
    print("Creating sample database...")
    path = create_sample_database()
    print(f"Sample database ready: {path}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message to the database admin agent."""
    # Get or create session
    if req.session_id and req.session_id in sessions:
        session = sessions[req.session_id]
        session_id = req.session_id
    elif req.session_id and req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    else:
        session_id = str(uuid.uuid4())
        session = create_session()
        sessions[session_id] = session

    # Clear tool log for this turn
    session["tool_log"].clear()

    # Run the agent (continue from existing message history)
    result = session["runner"].run(
        req.message,
        messages=session["messages"] if session["messages"] else None,
    )

    # Update session state
    session["messages"] = result.messages
    session["turns"] += 1

    # Get current state info from the runner's internal registry
    state = session["runner"]._registry._last_state
    available = []
    if state:
        actions = session["sm"].get_actions_for_state(state)
        available = [a.name for a in actions]

    return ChatResponse(
        session_id=session_id,
        response=result.text,
        tool_calls=list(session["tool_log"]),
        state=state,
        available_actions=available,
        turn_count=session["turns"],
    )


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return [
        {
            "session_id": sid,
            "state": s["runner"]._registry._last_state,
            "turns": s["turns"],
            "created_at": s["created_at"],
        }
        for sid, s in sessions.items()
    ]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Clean up a session and its database connection."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions.pop(session_id)
    if session["ctx"]["conn"]:
        session["ctx"]["conn"].close()
    return {"deleted": session_id}


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

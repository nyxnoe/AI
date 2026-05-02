"""
database.py
-----------
SQLite helper layer for the AI Decision Simulator.
Single source of truth for all DB logic — never duplicated in app.py.
"""

import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT    NOT NULL,
    scenario    TEXT    NOT NULL,
    budget      INTEGER NOT NULL,
    risk        INTEGER NOT NULL,
    time_weeks  INTEGER NOT NULL,
    priority    TEXT    NOT NULL,
    engine_used TEXT    NOT NULL DEFAULT 'anthropic',
    result_json TEXT,
    risk_level  TEXT,
    confidence  INTEGER,
    latency_ms  INTEGER DEFAULT 0,
    created_at  REAL    DEFAULT (strftime('%s','now'))
)
"""

# Migration: add latency_ms to existing DBs that lack it
_MIGRATE_LATENCY = """
ALTER TABLE simulations ADD COLUMN latency_ms INTEGER DEFAULT 0
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
    # Safe migration for existing databases
    cols = [r[1] for r in conn.execute("PRAGMA table_info(simulations)").fetchall()]
    if "latency_ms" not in cols:
        try:
            conn.execute(_MIGRATE_LATENCY)
            conn.commit()
            logger.info("Migrated: added latency_ms column")
        except Exception as e:
            logger.warning(f"Migration skipped: {e}")
    logger.info("Database schema ready.")


def insert_simulation(
    conn: sqlite3.Connection,
    params: dict,
    result: dict | None,
    engine_used: str,
    latency_ms: int = 0,
) -> int:
    cursor = conn.execute(
        """INSERT INTO simulations
           (domain, scenario, budget, risk, time_weeks, priority,
            engine_used, result_json, risk_level, confidence, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            params["domain"],
            params["scenario"],
            params["budget"],
            params["risk"],
            params["time"],
            params["priority"],
            engine_used,
            json.dumps(result) if result else None,
            result.get("riskLevel")       if result else None,
            result.get("confidenceScore") if result else None,
            latency_ms,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def fetch_history(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, domain, scenario, budget, risk, time_weeks, priority,
                  engine_used, risk_level, confidence, latency_ms, created_at
           FROM simulations
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_simulation(conn: sqlite3.Connection, sim_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM simulations WHERE id = ?", (sim_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except Exception:
            d["result"] = None
    return d


def delete_simulation(conn: sqlite3.Connection, sim_id: int) -> bool:
    cursor = conn.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
    conn.commit()
    return cursor.rowcount > 0
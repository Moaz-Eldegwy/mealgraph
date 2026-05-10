"""Long-term memory layer (semantic / procedural / episodic).

Phase 3 deliberately uses stdlib ``sqlite3`` rather than Mem0 / Letta /
sqlite-vec so the demo ships with zero extra dependencies. The interface,
however, mirrors the modern three-tier taxonomy from the 2026 agent-memory
literature so a later phase can swap the backend without touching call sites.

Tiers
-----
* **Working** — kept in the LangGraph state (untouched by this module).
* **Semantic** — atomic facts about the user (likes, dislikes, hard
  constraints, lab results). Survives across sessions.
* **Procedural** — verdicts the validator produced. Lets the system learn
  "this user rejected high-carb breakfasts twice" without re-asking.
* **Episodic** — JSON snapshot of past sessions for replay / audit.

Schema is intentionally tiny — three tables, one row per fact / verdict /
session. Vector search is *not* needed for this demo; SQL ``LIKE`` over
short text is good enough and adds zero dependencies. Phase 6 evals will
make the case for upgrading.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,         -- e.g. 'dislike', 'allergy', 'preference'
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',  -- e.g. 'user_stated', 'inferred', 'validator'
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON semantic_facts(user_id, fact_type);

CREATE TABLE IF NOT EXISTS procedural_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    plan_summary TEXT NOT NULL,
    verdict TEXT NOT NULL,            -- 'pass' | 'revise' | 'reject'
    issues_json TEXT NOT NULL,        -- JSON list of ValidationIssue
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proc_user ON procedural_records(user_id, created_at);

CREATE TABLE IF NOT EXISTS episodic_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,       -- JSON snapshot of session state
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_sessions(user_id, created_at);
"""


class LongTermMemory:
    """SQLite-backed three-tier long-term memory.

    Pass a file path for persistence across runs, or ``None`` (default) for an
    in-memory database useful in tests / ephemeral demos.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or ":memory:"
        # SQLite connections are not thread-safe by default; one connection per
        # thread is the standard pattern. The demo is single-process so a single
        # connection + lock is enough.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------
    # Semantic facts
    # ------------------------------------------------------------------
    def remember_fact(
        self,
        user_id: str,
        fact_type: str,
        content: str,
        source: str = "user_stated",
    ) -> int:
        """Insert a semantic fact. Returns the row id."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO semantic_facts (user_id, fact_type, content, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, fact_type, content, source, now),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def recall_facts(
        self,
        user_id: str,
        fact_type: Optional[str] = None,
        contains: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List facts for a user, optionally filtered by type / substring."""
        sql = "SELECT * FROM semantic_facts WHERE user_id = ?"
        params: List[Any] = [user_id]
        if fact_type:
            sql += " AND fact_type = ?"
            params.append(fact_type)
        if contains:
            sql += " AND content LIKE ?"
            params.append(f"%{contains}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self.conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def forget_fact(self, fact_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM semantic_facts WHERE id = ?", (fact_id,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Procedural records (validator history)
    # ------------------------------------------------------------------
    def remember_validation(
        self,
        user_id: str,
        plan_summary: str,
        verdict: str,
        issues: List[Dict[str, Any]],
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO procedural_records (user_id, plan_summary, verdict, issues_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, plan_summary, verdict, json.dumps(issues), now),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def recall_validations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM procedural_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [
                {**dict(row), "issues": json.loads(row["issues_json"])}
                for row in cur.fetchall()
            ]

    # ------------------------------------------------------------------
    # Episodic sessions
    # ------------------------------------------------------------------
    def remember_session(self, user_id: str, session_id: str, payload: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO episodic_sessions (user_id, session_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, session_id, json.dumps(payload, default=str), now),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def recall_sessions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM episodic_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [
                {**dict(row), "payload": json.loads(row["payload_json"])}
                for row in cur.fetchall()
            ]


__all__ = ["LongTermMemory"]

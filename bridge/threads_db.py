"""threads.sqlite — bridge-owned state, schema per spec 13.4.

The bridge is the only writer; the pipeline never reads this DB (FS mailbox is the
only pipeline interface). Fully recoverable from Slack history + FS (bridge/reconcile.py).
Run `python -m bridge.threads_db [path]` to create/migrate the DB standalone.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# PRAGMA user_version-based migrations; append new steps, never edit old ones.
MIGRATIONS: list[str] = [
    # v1 — spec 13.4 schema
    """
    CREATE TABLE runs (
        run_id           TEXT PRIMARY KEY,
        slack_thread_ts  TEXT UNIQUE NOT NULL,
        slack_channel_id TEXT NOT NULL,
        launched_at      TEXT NOT NULL,
        status           TEXT NOT NULL
    );
    CREATE TABLE escalations (
        escalation_id    TEXT PRIMARY KEY,
        run_id           TEXT NOT NULL REFERENCES runs(run_id),
        slack_msg_ts     TEXT NOT NULL,
        posted_at        TEXT NOT NULL,
        resolved_at      TEXT,
        reply_method     TEXT,
        reply_payload    TEXT
    );
    CREATE INDEX idx_esc_run_open ON escalations(run_id, resolved_at);
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, step in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(step)
        conn.execute(f"PRAGMA user_version = {i}")
    conn.commit()


class ThreadsDB:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        migrate(self.conn)

    # -- runs ------------------------------------------------------------
    def insert_run(self, run_id: str, thread_ts: str, channel_id: str,
                   status: str = "running", launched_at: str | None = None) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?)",
            (run_id, thread_ts, channel_id, launched_at or _now(), status))
        self.conn.commit()
        return cur.rowcount > 0

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def run_for_thread(self, thread_ts: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE slack_thread_ts=?", (thread_ts,)).fetchone()

    def set_run_status(self, run_id: str, status: str) -> None:
        self.conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
        self.conn.commit()

    # -- escalations -----------------------------------------------------
    def insert_escalation(self, escalation_id: str, run_id: str, slack_msg_ts: str,
                          posted_at: str | None = None) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO escalations (escalation_id, run_id, slack_msg_ts, posted_at)"
            " VALUES (?,?,?,?)", (escalation_id, run_id, slack_msg_ts, posted_at or _now()))
        self.conn.commit()
        return cur.rowcount > 0

    def get_escalation(self, escalation_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM escalations WHERE escalation_id=?", (escalation_id,)).fetchone()

    def newest_unresolved(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM escalations WHERE run_id=? AND resolved_at IS NULL"
            " ORDER BY posted_at DESC, escalation_id DESC LIMIT 1", (run_id,)).fetchone()

    def unresolved(self, run_id: str) -> list[sqlite3.Row]:
        """ALL open escalations for a run, newest first (free-text routing must disambiguate)."""
        return self.conn.execute(
            "SELECT * FROM escalations WHERE run_id=? AND resolved_at IS NULL"
            " ORDER BY posted_at DESC, escalation_id DESC", (run_id,)).fetchall()

    def mark_resolved(self, escalation_id: str, method: str, payload: str) -> bool:
        """Atomically resolve; returns False if it was already resolved (race-safe)."""
        cur = self.conn.execute(
            "UPDATE escalations SET resolved_at=?, reply_method=?, reply_payload=?"
            " WHERE escalation_id=? AND resolved_at IS NULL",
            (_now(), method, payload[:2000], escalation_id))
        self.conn.commit()
        return cur.rowcount > 0


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "threads.sqlite")
    ThreadsDB(target)
    print(f"migrated {target}")

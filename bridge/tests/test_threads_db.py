"""threads_db.py — spec 13.4 schema, migration idempotence, race-safe resolution."""
import sqlite3

from bridge.threads_db import ThreadsDB, migrate


def test_schema_columns(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(runs)")}
    assert cols == {"run_id", "slack_thread_ts", "slack_channel_id", "launched_at", "status"}
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(escalations)")}
    assert cols == {"escalation_id", "run_id", "slack_msg_ts", "posted_at",
                    "resolved_at", "reply_method", "reply_payload"}


def test_migration_idempotent(cfg):
    a = ThreadsDB(cfg.db_path)
    migrate(a.conn)  # re-running is a no-op
    b = ThreadsDB(cfg.db_path)  # reopening migrates again harmlessly
    assert b.conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_run_roundtrip(db):
    assert db.insert_run("r1", "111.0", "C_REQ")
    assert not db.insert_run("r1", "222.0", "C_REQ")  # dup run_id ignored
    assert db.run_for_thread("111.0")["run_id"] == "r1"
    assert db.run_for_thread("999.9") is None
    db.set_run_status("r1", "escalated")
    assert db.get_run("r1")["status"] == "escalated"


def test_newest_unresolved_ordering(db):
    db.insert_run("r1", "1.0", "C")
    db.insert_escalation("ingest_1_a", "r1", "2.0", posted_at="2026-06-09T01:00:00Z")
    db.insert_escalation("filter_2_b", "r1", "3.0", posted_at="2026-06-09T02:00:00Z")
    assert db.newest_unresolved("r1")["escalation_id"] == "filter_2_b"
    db.mark_resolved("filter_2_b", "cli", "1")
    assert db.newest_unresolved("r1")["escalation_id"] == "ingest_1_a"
    db.mark_resolved("ingest_1_a", "slack_button", "2")
    assert db.newest_unresolved("r1") is None


def test_mark_resolved_race_safe(db):
    db.insert_run("r1", "1.0", "C")
    db.insert_escalation("e_1_a", "r1", "2.0")
    assert db.mark_resolved("e_1_a", "file_drop", "1")
    assert not db.mark_resolved("e_1_a", "slack_button", "2")  # second writer loses
    esc = db.get_escalation("e_1_a")
    assert esc["reply_method"] == "file_drop" and esc["reply_payload"] == "1"


def test_thread_ts_unique(db):
    db.insert_run("r1", "1.0", "C")
    try:
        db.conn.execute("INSERT INTO runs VALUES ('r2','1.0','C','t','running')")
        assert False, "UNIQUE constraint should fire"
    except sqlite3.IntegrityError:
        pass

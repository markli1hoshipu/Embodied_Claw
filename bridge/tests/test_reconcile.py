"""reconcile.py — rebuild threads.sqlite from mocked Slack history + FS walk (spec 13.9/13.11)."""
from bridge.reconcile import reconcile, reconcile_from_fs, reconcile_from_slack
from bridge.tests.conftest import write_question


class FakeHistoryClient:
    """Mocked WebClient: parent messages in history, threaded replies per thread_ts."""

    def __init__(self, history, threads):
        self.history, self.threads = history, threads

    def conversations_history(self, channel, **kwargs):
        return {"messages": self.history, "response_metadata": {"next_cursor": ""}}

    def conversations_replies(self, channel, ts, **kwargs):
        return {"messages": self.threads.get(ts, [])}


def _launch_msg(run_id, ts, thread_ts=None, reply_count=0):
    msg = {"ts": ts, "reply_count": reply_count,
           "metadata": {"event_type": "run_launched", "event_payload": {"run_id": run_id}}}
    if thread_ts:
        msg["thread_ts"] = thread_ts
    return msg


def test_rebuild_from_slack_history(db):
    history = [
        # a colleague's mention with a bot reply inside the thread
        {"ts": "10.0", "reply_count": 2, "text": "<@U_BOT> train ..."},
        # a bridge-created anchor for a CLI-launched run (bot message at top level)
        _launch_msg("cli_run", "20.0"),
    ]
    threads = {"10.0": [
        {"ts": "10.0", "text": "<@U_BOT> train ..."},
        _launch_msg("slack_run", "10.1", thread_ts="10.0"),
        {"ts": "10.2", "thread_ts": "10.0",
         "metadata": {"event_type": "escalation_posted",
                      "event_payload": {"run_id": "slack_run",
                                        "escalation_id": "filter_build_1_aa"}}},
    ]}
    reconcile_from_slack(db, FakeHistoryClient(history, threads), "C_REQ")
    assert db.get_run("slack_run")["slack_thread_ts"] == "10.0"
    assert db.get_run("cli_run")["slack_thread_ts"] == "20.0"
    esc = db.get_escalation("filter_build_1_aa")
    assert esc["run_id"] == "slack_run" and esc["slack_msg_ts"] == "10.2"


def test_slack_scan_failure_is_nonfatal(db):
    class Boom:
        def conversations_history(self, **kwargs):
            raise RuntimeError("no network in tests")

    reconcile_from_slack(db, Boom(), "C_REQ")  # must not raise


def test_fs_walk_resolves_answered_escalations(cfg, db):
    db.insert_run("r1", "1.0", "C_REQ")
    db.insert_escalation("filter_1_aa", "r1", "2.0")
    db.insert_escalation("train_2_bb", "r1", "3.0")
    write_question(cfg.runs_root, "r1", "filter_1_aa")
    write_question(cfg.runs_root, "r1", "train_2_bb")
    (cfg.runs_root / "r1" / "escalations" / "filter_1_aa.reply.txt").write_text("1\n")
    reconcile_from_fs(db, cfg.runs_root)
    assert db.get_escalation("filter_1_aa")["reply_method"] == "file_drop"
    assert db.get_escalation("filter_1_aa")["reply_payload"] == "1"
    assert db.get_escalation("train_2_bb")["resolved_at"] is None  # still open


def test_fs_walk_updates_run_status_from_transitions(cfg, db):
    db.insert_run("r1", "1.0", "C_REQ")
    (cfg.runs_root / "r1").mkdir(exist_ok=True)
    (cfg.runs_root / "r1" / "transitions.jsonl").write_text(
        '{"ts":"1","run_id":"r1","node":"ingest","status":"succeeded"}\n'
        '{"ts":"2","run_id":"r1","status":"failed","detail":"oom"}\n')
    reconcile_from_fs(db, cfg.runs_root)
    assert db.get_run("r1")["status"] == "failed"


def test_full_reconcile_idempotent(cfg, db):
    client = FakeHistoryClient([_launch_msg("r1", "1.0")], {})
    write_question(cfg.runs_root, "r1", "e_1_a")
    reconcile(db, client, cfg.runs_root, "C_REQ")
    reconcile(db, client, cfg.runs_root, "C_REQ")  # second run is a no-op
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

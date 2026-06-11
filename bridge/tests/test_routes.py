"""routes.py — inbound routing (13.5), mailbox handling (13.6), security (13.8), races (13.9)."""
import json

from bridge.tests.conftest import click_payload, mention_event, write_question


# ---------- mentions / operator threads ----------
def test_mention_opens_train_operator_thread(cfg, db, gateway, router, spawned):
    op_spawned = []
    router._spawn_operator = op_spawned.append
    router.handle_event(mention_event("Train pi0.5 on the radio task, PCA p98"))
    assert spawned == [] and op_spawned == ["1.000001"]  # no direct FSM launch anymore
    d = cfg.runs_root / "_operator" / "1.000001"
    assert (d / ".module").read_text().strip() == "pipeline.operator"
    inbox = list((d / "inbox").glob("*.txt"))
    assert len(inbox) == 1 and "radio task" in inbox[0].read_text()
    assert "<@U_BOT>" not in inbox[0].read_text()  # bot mention stripped
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_wrong_channel_mention_redirected(cfg, db, gateway, router, spawned):
    router.handle_event(mention_event(channel="C_RANDOM"))
    assert spawned == [] and db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert gateway.posts[0]["channel"] == "C_RANDOM"
    assert "<#C_REQ>" in gateway.posts[0]["text"]


def test_allowed_user_list_enforced(cfg, db, gateway, spawned):
    from bridge.routes import Router
    cfg.allowed_users = {"U_BOSS"}
    router = Router(cfg, db, gateway, spawn=spawned.append)
    op_spawned = []
    router._spawn_operator = op_spawned.append
    router.handle_event(mention_event(user="U_EVE"))
    assert op_spawned == [] and "not on the allowed-user list" in gateway.posts[0]["text"]
    router.handle_event(mention_event(user="U_BOSS", ts="2.0"))
    assert op_spawned == ["2.0"]


def test_empty_mention_asks_for_text(gateway, router, spawned):
    router.handle_event(mention_event(""))
    assert spawned == [] and "Tell me what to run" in gateway.posts[0]["text"]


def test_spawns_use_subprocess_not_import(cfg, db, gateway, monkeypatch):
    """13.1: the bridge never imports pipeline/operator code — subprocess only."""
    from bridge.routes import Router
    calls = []
    monkeypatch.setattr("bridge.routes.subprocess.Popen", lambda *a, **k: calls.append((a, k)))
    router = Router(cfg, db, gateway)
    router.handle_event(mention_event())  # train channel -> pipeline.operator handle <ts>
    (argv,), kwargs = calls[0]
    assert argv[1:5] == ["-m", "pipeline.operator", "handle", "1.000001"]
    assert kwargs["cwd"] == str(cfg.repo_root) and kwargs["start_new_session"]
    router._spawn_pipeline("rid1")  # FSM spawn helper keeps the same guarantee
    (argv,), kwargs = calls[1]
    assert argv[1:4] == ["-m", "pipeline.cli", "run"] and argv[4] == "--detach"
    assert kwargs["cwd"] == str(cfg.repo_root) and kwargs["start_new_session"]


# ---------- question files ----------
def test_question_file_posted_in_thread(cfg, db, gateway, router):
    db.insert_run("r1", "5.0", "C_REQ")
    qfile = write_question(cfg.runs_root, "r1", "filter_build_1718_a1b2")
    router.handle_question_file(qfile)
    post = gateway.posts[-1]
    assert post["thread_ts"] == "5.0" and post["blocks"][0]["type"] == "header"
    esc = db.get_escalation("filter_build_1718_a1b2")
    assert esc["run_id"] == "r1" and esc["resolved_at"] is None
    assert db.get_run("r1")["status"] == "escalated"
    n = len(gateway.posts)
    router.handle_question_file(qfile)  # duplicate FS event -> no double post
    assert len(gateway.posts) == n


def test_question_for_cli_launched_run_creates_thread(cfg, db, gateway, router):
    qfile = write_question(cfg.runs_root, "cli_run", "ingest_1_aa")
    router.handle_question_file(qfile)
    anchor, question = gateway.posts[0], gateway.posts[1]
    assert "cli_run" in anchor["text"] and anchor["thread_ts"] is None
    assert question["thread_ts"] == anchor["ts"]
    assert db.get_run("cli_run") is not None


# ---------- replies: buttons, thread text, file drops ----------
def _setup_escalation(cfg, db, gateway, router, esc_id="filter_build_1718_a1b2"):
    db.insert_run("r1", "5.0", "C_REQ")
    router.handle_question_file(write_question(cfg.runs_root, "r1", esc_id))
    return esc_id


def test_button_click_writes_reply_file(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0"))
    reply = cfg.runs_root / "r1" / "escalations" / f"{esc_id}.reply.txt"
    assert reply.read_text().strip() == "1"  # single int = option id (spec 7.2B)
    esc = db.get_escalation(esc_id)
    assert esc["reply_method"] == "slack_button" and esc["reply_payload"] == "1"
    assert "Reply recorded" in gateway.posts[-1]["text"]
    assert gateway.posts[-1]["thread_ts"] == "5.0"


def test_thread_text_reply_freeform(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_event({"type": "message", "channel": "C_REQ", "user": "U_ALICE",
                         "ts": "6.0", "thread_ts": "5.0",
                         "text": "Use p98 but force-drop ep280"})
    reply = cfg.runs_root / "r1" / "escalations" / f"{esc_id}.reply.txt"
    assert reply.read_text().strip() == "Use p98 but force-drop ep280"
    assert db.get_escalation(esc_id)["reply_method"] == "slack_text"


def test_freeform_reply_with_two_open_escalations_disambiguates(cfg, db, gateway, router):
    """Nodes 3+4 escalate concurrently in one thread: free text must NOT auto-route to the
    newest question — ask the user to disambiguate, route again once only one is open."""
    db.insert_run("r1", "5.0", "C_REQ")
    router.handle_question_file(write_question(cfg.runs_root, "r1", "upload_dataset_100_aa"))
    router.handle_question_file(write_question(cfg.runs_root, "r1", "norm_stats_101_bb",
                                               question="norm stats look odd — proceed?"))
    router.handle_event({"type": "message", "channel": "C_REQ", "user": "U_ALICE",
                         "ts": "6.0", "thread_ts": "5.0", "text": "yes overwrite"})
    assert not list((cfg.runs_root / "r1" / "escalations").glob("*.reply.*"))  # nothing routed
    assert db.get_escalation("upload_dataset_100_aa")["resolved_at"] is None
    assert db.get_escalation("norm_stats_101_bb")["resolved_at"] is None
    assert len(gateway.ephemerals) == 1
    eph = gateway.ephemerals[0]["text"]
    assert "2 questions are open" in eph and "buttons" in eph
    # button click (carries esc_id) still resolves precisely
    router.handle_block_actions(click_payload("upload_dataset_100_aa", 1, thread_ts="5.0"))
    # now exactly one open question -> free text auto-routes to it
    router.handle_event({"type": "message", "channel": "C_REQ", "user": "U_ALICE",
                         "ts": "7.0", "thread_ts": "5.0", "text": "yes proceed"})
    reply = cfg.runs_root / "r1" / "escalations" / "norm_stats_101_bb.reply.txt"
    assert reply.read_text().strip() == "yes proceed"


def test_allowed_users_also_gates_escalation_answers(cfg, db, gateway, spawned):
    """13.8 tightened: when the allow-list is set, thread replies and button clicks from
    non-listed users are rejected ephemerally instead of resolving escalations."""
    from bridge.routes import Router
    cfg.allowed_users = {"U_BOSS"}
    router = Router(cfg, db, gateway, spawn=spawned.append)
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0", user="U_EVE"))
    router.handle_event({"type": "message", "channel": "C_REQ", "user": "U_EVE",
                         "ts": "6.0", "thread_ts": "5.0", "text": "1"})
    assert db.get_escalation(esc_id)["resolved_at"] is None
    assert not list((cfg.runs_root / "r1" / "escalations").glob("*.reply.*"))
    assert all("not on the allowed-user list" in e["text"] for e in gateway.ephemerals)
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0", user="U_BOSS"))
    assert db.get_escalation(esc_id)["reply_method"] == "slack_button"


def test_cli_reply_first_race_rejects_button(cfg, db, gateway, router):
    """Spec 13.9: file lands before the click -> resolve from file, reject click ephemerally."""
    esc_id = _setup_escalation(cfg, db, gateway, router)
    fs_reply = cfg.runs_root / "r1" / "escalations" / f"{esc_id}.reply.txt"
    fs_reply.write_text("2\n")  # CLI got there first
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0"))
    assert fs_reply.read_text().strip() == "2"  # button did NOT overwrite
    esc = db.get_escalation(esc_id)
    assert esc["reply_method"] == "file_drop" and esc["reply_payload"] == "2"
    assert len(gateway.ephemerals) == 1
    assert "already answered" in gateway.ephemerals[0]["text"]


def test_second_button_click_rejected(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0"))
    router.handle_block_actions(click_payload(esc_id, 3, thread_ts="5.0", user="U_BOB"))
    assert db.get_escalation(esc_id)["reply_payload"] == "1"
    assert gateway.ephemerals[0]["user"] == "U_BOB"
    assert "already answered" in gateway.ephemerals[0]["text"]


def test_reply_file_event_acks_in_thread(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    reply = cfg.runs_root / "r1" / "escalations" / f"{esc_id}.reply.txt"
    reply.write_text("Use p98 but force-drop 280\n")
    router.handle_reply_file(reply)
    assert db.get_escalation(esc_id)["reply_method"] == "file_drop"
    last = gateway.posts[-1]
    assert "CLI/file drop" in last["text"] and last["thread_ts"] == "5.0"
    n = len(gateway.posts)
    router.handle_reply_file(reply)  # second event for same file -> no double ack
    assert len(gateway.posts) == n


def test_own_reply_write_does_not_reack(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_block_actions(click_payload(esc_id, 1, thread_ts="5.0"))
    n = len(gateway.posts)
    reply = cfg.runs_root / "r1" / "escalations" / f"{esc_id}.reply.txt"
    router.handle_reply_file(reply)  # watcher sees our own write
    assert len(gateway.posts) == n


def test_unknown_thread_and_unknown_escalation_ignored(cfg, db, gateway, router):
    router.handle_event({"type": "message", "channel": "C_REQ", "user": "U_X",
                         "ts": "6.0", "thread_ts": "404.0", "text": "hello?"})
    router.handle_block_actions(click_payload("ghost_1_zz", 1))
    router.handle_reply_file(cfg.runs_root / "r1" / "escalations" / "ghost_1_zz.reply.txt")
    assert gateway.posts == [] and gateway.ephemerals == []


def test_bot_and_edited_messages_ignored(cfg, db, gateway, router):
    esc_id = _setup_escalation(cfg, db, gateway, router)
    router.handle_event({"type": "message", "bot_id": "B1", "thread_ts": "5.0",
                         "ts": "6.0", "text": "1"})
    router.handle_event({"type": "message", "subtype": "message_changed",
                         "channel": "C_REQ", "ts": "6.1"})
    assert db.get_escalation(esc_id)["resolved_at"] is None


# ---------- transitions ----------
def test_transition_posts_status_and_updates_run(cfg, db, gateway, router):
    db.insert_run("r1", "5.0", "C_REQ")
    router.handle_transition({"ts": "t", "run_id": "r1", "node": "ingest",
                              "status": "succeeded", "detail": "144 parquets",
                              "future_field": 1})
    assert gateway.posts[-1]["channel"] == "C_STATUS" and "ingest" in gateway.posts[-1]["text"]
    router.handle_transition({"ts": "t", "run_id": "r1", "node": "train", "status": "failed"})
    assert db.get_run("r1")["status"] == "failed"
    router.handle_transition({"ts": "t", "run_id": "r1", "status": "done"})
    assert db.get_run("r1")["status"] == "done"


def test_transition_without_status_channel(cfg, db, gateway, spawned):
    from bridge.routes import Router
    cfg.status_channel = None
    router = Router(cfg, db, gateway, spawn=spawned.append)
    router.handle_transition({"run_id": "r9", "node": "x", "status": "running"})
    assert gateway.posts == []  # no status channel configured -> silent


def test_secrets_redacted_in_question_post(cfg, db, gateway, router):
    db.insert_run("r1", "5.0", "C_REQ")
    qfile = write_question(cfg.runs_root, "r1", "upload_1_zz",
                           question="Use token hf_" + "S" * 30 + " for the upload?",
                           options=None, recommendation=None)
    router.handle_question_file(qfile)
    assert "hf_S" not in json.dumps(gateway.posts[-1]["blocks"]) + gateway.posts[-1]["text"]

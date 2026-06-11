"""slack_bridge.py — path classification, transitions tailing, gateway retry/queue, socket ack."""
import queue
from types import SimpleNamespace

import pytest
from slack_sdk.errors import SlackApiError

from bridge.routes import Router
from bridge.slack_bridge import Bridge, MailboxHandler, SlackGateway, classify_path


def make_bridge(cfg, db, gateway, spawned=None):
    router = Router(cfg, db, gateway, spawn=(spawned.append if spawned is not None else lambda r: None))
    return Bridge(cfg, db, gateway, router, socket_client=None)


# ---------- path classification / watchdog handler ----------
def test_classify_path():
    assert classify_path("/r/runs/r1/escalations/a_1_b.question.json") == "question"
    assert classify_path("/r/runs/r1/escalations/a_1_b.reply.txt") == "reply"
    assert classify_path("/r/runs/r1/escalations/a_1_b.reply.json") == "reply"
    assert classify_path("/r/runs/r1/transitions.jsonl") == "transitions"
    assert classify_path("/r/runs/r1/escalations/.a_1_b.tmp") is None  # atomic-write temp
    assert classify_path("/r/runs/r1/config.json") is None
    assert classify_path("/r/runs/r1/request.txt") is None


def test_mailbox_handler_enqueues():
    q = queue.Queue()
    h = MailboxHandler(q)
    ev = SimpleNamespace(is_directory=False, src_path="/x/runs/r/escalations/e_1_a.question.json")
    h.on_created(ev)
    h.on_moved(SimpleNamespace(is_directory=False, src_path="/t",
                               dest_path="/x/runs/r/escalations/e_1_a.reply.txt"))
    h.on_modified(SimpleNamespace(is_directory=False, src_path="/x/runs/r/transitions.jsonl"))
    h.on_modified(SimpleNamespace(is_directory=False, src_path="/x/runs/r/state.sqlite"))
    kinds = [q.get_nowait()[0] for _ in range(3)]
    assert kinds == ["fs_question", "fs_reply", "fs_transitions"]
    assert q.empty()


# ---------- transitions tailing ----------
def test_drain_transitions_incremental_and_tolerant(cfg, db, gateway):
    bridge = make_bridge(cfg, db, gateway)
    run_dir = cfg.runs_root / "r1"
    run_dir.mkdir()
    t = run_dir / "transitions.jsonl"
    t.write_text('{"ts":"1","run_id":"r1","node":"ingest","status":"running"}\n'
                 'not json at all\n'
                 '{"ts":"2","node":"ingest","status":"succeeded","detail":"144 files","new_f":[]}\n')
    bridge.drain_transitions(t)
    assert len(gateway.posts) == 2  # bad line skipped, missing run_id defaulted from path
    assert "`r1`" in gateway.posts[1]["text"] and "144 files" in gateway.posts[1]["text"]
    with open(t, "a") as fh:
        fh.write('{"ts":"3","run_id":"r1","node":"train","status":"running"}\n')
        fh.write('{"partial line without newline')
    bridge.drain_transitions(t)
    assert len(gateway.posts) == 3  # only the complete new line; no re-posting of old ones
    assert "train" in gateway.posts[2]["text"]


def test_initial_scan_skips_transition_history(cfg, db, gateway):
    run_dir = cfg.runs_root / "r1"
    run_dir.mkdir()
    t = run_dir / "transitions.jsonl"
    t.write_text('{"ts":"1","run_id":"r1","node":"ingest","status":"running"}\n')
    bridge = make_bridge(cfg, db, gateway)
    bridge.initial_scan()
    bridge.drain_transitions(t)
    assert gateway.posts == []  # pre-existing lines not re-broadcast after restart


# ---------- gateway: 429 Retry-After, network queueing, redaction ----------
class FakeWebClient:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []

    def chat_postMessage(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return {"ts": "42.0"}


def _rate_limit_error():
    resp = SimpleNamespace(status_code=429, headers={"Retry-After": "3"})
    return SlackApiError("ratelimited", resp)


def test_gateway_honors_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr("bridge.slack_bridge.time.sleep", sleeps.append)
    web = FakeWebClient([_rate_limit_error(), _rate_limit_error()])
    gw = SlackGateway(web)
    assert gw.post_message("C", "hello") == "42.0"
    assert sleeps == [3, 3] and len(web.calls) == 3


def test_gateway_queues_on_network_error_and_flushes():
    web = FakeWebClient([ConnectionError("down")])
    gw = SlackGateway(web)
    assert gw.post_message("C", "hello") == ""  # queued, not raised
    assert len(gw.pending) == 1
    gw.flush_pending()
    assert len(gw.pending) == 0 and web.calls[-1]["text"] == "hello"


def test_gateway_redacts_outbound_text_and_blocks():
    web = FakeWebClient([])
    gw = SlackGateway(web)
    gw.post_message("C", "tok hf_" + "K" * 25,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn",
                             "text": "xoxb-111-secret"}}])
    sent = web.calls[0]
    assert "hf_K" not in sent["text"] and "xoxb-111" not in str(sent["blocks"])


def test_gateway_redacts_metadata_too():
    """run_id slugs derive from raw user text — a pasted token must not survive in metadata."""
    web = FakeWebClient([])
    gw = SlackGateway(web)
    gw.post_message("C", "ok", metadata={"event_type": "run_launched",
                                         "event_payload": {"run_id": "use_hf_" + "k" * 24}})
    sent = web.calls[0]["metadata"]
    assert "hf_k" not in str(sent) and "[REDACTED]" in str(sent)


# ---------- socket request ack + enqueue ----------
def test_socket_request_acked_and_enqueued(cfg, db, gateway):
    bridge = make_bridge(cfg, db, gateway)
    acks = []
    client = SimpleNamespace(send_socket_mode_response=acks.append)
    req = SimpleNamespace(type="events_api", envelope_id="env1",
                          payload={"event": {"type": "app_mention", "text": "hi"}})
    bridge._on_socket_request(client, req)
    req2 = SimpleNamespace(type="interactive", envelope_id="env2",
                           payload={"type": "block_actions", "actions": []})
    bridge._on_socket_request(client, req2)
    assert [a.envelope_id for a in acks] == ["env1", "env2"]
    assert bridge.events.get_nowait() == ("slack_event", {"type": "app_mention", "text": "hi"})
    assert bridge.events.get_nowait()[0] == "block_actions"


def test_dispatch_routes_kinds(cfg, db, gateway, spawned):
    bridge = make_bridge(cfg, db, gateway, spawned)
    op_spawned = []
    bridge.router._spawn_operator = op_spawned.append
    bridge.dispatch("slack_event", {"type": "app_mention", "text": "<@U_B> go",
                                    "channel": "C_REQ", "user": "U_A", "ts": "1.0"})
    assert op_spawned == ["1.0"] and spawned == []  # mention -> operator, not direct launch


def test_reconnect_backoff(cfg, db, gateway, monkeypatch):
    sleeps = []
    monkeypatch.setattr("bridge.slack_bridge.time.sleep", sleeps.append)
    bridge = make_bridge(cfg, db, gateway)
    bridge.socket_client = SimpleNamespace(
        is_connected=lambda: False,
        connect=lambda: (_ for _ in ()).throw(OSError("net down")))
    for _ in range(4):
        bridge._ensure_connected()
    assert sleeps == [1, 2, 4, 8]  # exponential backoff (13.9)

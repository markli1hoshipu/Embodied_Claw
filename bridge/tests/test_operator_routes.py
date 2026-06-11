"""Operator-thread routing: eval mention -> inbox+spawn, thread reply -> inbox,
outbox -> thread post, terminal transition -> operator notification."""
from __future__ import annotations

import json

import pytest

from bridge.config import BridgeConfig
from bridge.routes import Router
from bridge.threads_db import ThreadsDB

from .conftest import FakeGateway


@pytest.fixture
def op_setup(tmp_path):
    cfg = BridgeConfig(bot_token="x", app_token="x", requests_channel="C_TRAIN",
                       status_channel=None, repo_root=tmp_path, runs_root=tmp_path / "runs",
                       db_path=tmp_path / "t.sqlite", eval_channel="C_EVAL")
    cfg.runs_root.mkdir(parents=True)
    db = ThreadsDB(cfg.db_path)
    gw = FakeGateway()
    spawned: list[str] = []
    router = Router(cfg, db, gw)
    router._spawn_operator = spawned.append
    return cfg, db, gw, router, spawned


def _mention(channel, ts, text, user="U1", thread_ts=None):
    e = {"type": "app_mention", "channel": channel, "user": user, "ts": ts,
         "text": f"<@BOT> {text}"}
    if thread_ts:
        e["thread_ts"] = thread_ts
    return e


def test_eval_mention_opens_operator_thread(op_setup):
    cfg, db, gw, router, spawned = op_setup
    router.handle_app_mention(_mention("C_EVAL", "111.222", "eval pi05 on domino"))
    assert spawned == ["111.222"]
    d = cfg.runs_root / "_operator" / "111.222"
    inbox = list((d / "inbox").glob("*.txt"))
    assert len(inbox) == 1 and "eval pi05 on domino" in inbox[0].read_text()
    assert (d / ".module").read_text().strip() == "eval_domino.operator"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["channel"] == "C_EVAL"
    # no FSM run was spawned directly and nothing posted yet (operator will reply via outbox)
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_train_channel_opens_pipeline_operator(op_setup):
    cfg, db, gw, router, spawned = op_setup
    fsm_spawned = []
    router.spawn = fsm_spawned.append
    router.handle_app_mention(
        _mention("C_TRAIN", "333.444", "what is the current status of checkpoints"))
    assert fsm_spawned == [] and spawned == ["333.444"]
    d = cfg.runs_root / "_operator" / "333.444"
    assert (d / ".module").read_text().strip() == "pipeline.operator"
    inbox = list((d / "inbox").glob("*.txt"))
    assert len(inbox) == 1 and "status of checkpoints" in inbox[0].read_text()
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_spawn_operator_dispatches_by_module_file(op_setup, monkeypatch):
    cfg, db, gw, router, spawned = op_setup
    calls = []
    monkeypatch.setattr("bridge.routes.subprocess.Popen", lambda *a, **k: calls.append((a, k)))
    import bridge.routes as routes_mod
    real = routes_mod.Router._spawn_operator
    for ts, channel, module in (("1.1", "C_TRAIN", "pipeline.operator"),
                                ("2.2", "C_EVAL", "eval_domino.operator")):
        router.handle_app_mention(_mention(channel, ts, "hello"))
        real(router, ts)
        (argv,), kwargs = calls[-1]
        assert argv[1:5] == ["-m", module, "handle", ts]
        assert kwargs["start_new_session"]
    # legacy thread without a .module file falls back to the eval operator
    legacy = cfg.runs_root / "_operator" / "9.9"
    legacy.mkdir(parents=True)
    real(router, "9.9")
    (argv,), _ = calls[-1]
    assert argv[1:3] == ["-m", "eval_domino.operator"]


def test_thread_reply_routes_to_operator(op_setup):
    cfg, db, gw, router, spawned = op_setup
    router.handle_app_mention(_mention("C_EVAL", "111.222", "start"))
    router.handle_message({"type": "message", "channel": "C_EVAL", "user": "U2",
                           "ts": "111.999", "thread_ts": "111.222", "text": "use 2 gpus"})
    inbox = sorted((cfg.runs_root / "_operator" / "111.222" / "inbox").glob("*.txt"))
    assert len(inbox) == 2 and "use 2 gpus" in inbox[-1].read_text()
    assert spawned == ["111.222", "111.222"]


def test_operator_outbox_posts_to_thread(op_setup):
    cfg, db, gw, router, spawned = op_setup
    router.handle_app_mention(_mention("C_EVAL", "111.222", "start"))
    out = cfg.runs_root / "_operator" / "111.222" / "outbox" / "1.md"
    out.write_text("Confirmed: 1 GPU, adjust_bottle, 10 eps. Launching.")
    router.handle_operator_out(out)
    assert gw.posts and gw.posts[-1]["thread_ts"] == "111.222"
    assert "Launching" in gw.posts[-1]["text"]
    assert not out.exists() and out.with_suffix(".posted").exists()


def test_terminal_transition_notifies_operator(op_setup):
    cfg, db, gw, router, spawned = op_setup
    router.handle_app_mention(_mention("C_EVAL", "111.222", "start"))
    rd = cfg.runs_root / "my_eval"
    rd.mkdir()
    (rd / ".operator_thread").write_text("111.222")
    router.handle_transition({"run_id": "my_eval", "node": "aggregate_report",
                              "status": "succeeded", "detail": "done"})
    inbox = sorted((cfg.runs_root / "_operator" / "111.222" / "inbox").glob("*.txt"))
    assert any("my_eval" in p.read_text() for p in inbox)
    # non-terminal events do not notify
    n = len(inbox)
    router.handle_transition({"run_id": "my_eval", "node": "run_matrix", "status": "running"})
    assert len(sorted((cfg.runs_root / "_operator" / "111.222" / "inbox").glob("*.txt"))) == n

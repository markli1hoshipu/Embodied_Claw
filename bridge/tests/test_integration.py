"""Integration round-trip (spec 13.11 / addendum B1), fully mocked, tmpdir mailbox:

mention -> operator thread (inbox + spawn) -> operator outbox -> posted in thread;
FSM run escalates -> question.json -> bridge renders Block Kit in (tracking) thread ->
button click -> reply file written -> ack; plus the CLI-reply-first race and the
service's no-token setup exit (addenda B4/B5).
"""
import json
import os
import subprocess
import sys

from bridge.routes import Router
from bridge.slack_bridge import Bridge
from bridge.tests.conftest import click_payload, mention_event, write_question


def make_bridge(cfg, db, gateway, spawned):
    router = Router(cfg, db, gateway, spawn=spawned.append)
    return Bridge(cfg, db, gateway, router, socket_client=None)


def drain(bridge):
    while not bridge.events.empty():
        bridge.dispatch(*bridge.events.get_nowait())


def test_full_round_trip(cfg, db, gateway, spawned):
    bridge = make_bridge(cfg, db, gateway, spawned)

    # 1. colleague mentions the bot in the training channel -> operator thread opens
    op_spawned = []
    bridge.router._spawn_operator = op_spawned.append
    bridge.dispatch("slack_event", mention_event("what is the current status of checkpoints"))
    assert op_spawned == ["1.000001"] and spawned == []  # conversational, no direct launch
    opd = cfg.runs_root / "_operator" / "1.000001"
    assert (opd / ".module").read_text().strip() == "pipeline.operator"
    assert "status of checkpoints" in next(iter((opd / "inbox").glob("*.txt"))).read_text()

    # 2. the operator replies via its outbox -> bridge posts into the thread
    (opd / "outbox" / "1.md").write_text("```config | steps | live```\nAll checkpoints healthy.")
    bridge.initial_scan()
    drain(bridge)
    assert any(p["thread_ts"] == "1.000001" and "checkpoints healthy" in p["text"]
               for p in gateway.posts)

    # 3. an FSM run (launched via CLI/operator) escalates -> tracking thread + Block Kit
    run_id = "task0_perturb_v4"
    esc_id = "filter_build_1718312537_a1b2"
    write_question(cfg.runs_root, run_id, esc_id)
    bridge.initial_scan()  # same code path the watchdog events feed
    drain(bridge)
    tracking = next(p for p in gateway.posts if "Tracking run" in p["text"])
    question_post = [p for p in gateway.posts if p["blocks"]][-1]
    assert question_post["thread_ts"] == tracking["ts"]
    buttons = [el for b in question_post["blocks"] if b["type"] == "actions"
               for el in b["elements"]]
    assert buttons[0]["action_id"] == f"esc:{esc_id}:opt:1"
    assert "✓" in buttons[0]["text"]["text"]  # recommendation marked

    # 3b. colleague clicks the recommended button
    bridge.dispatch("block_actions", click_payload(esc_id, 1, thread_ts=tracking["ts"]))
    reply_file = cfg.runs_root / run_id / "escalations" / f"{esc_id}.reply.txt"
    assert reply_file.read_text().strip() == "1"  # single int = option id (spec 7.2)
    assert db.get_escalation(esc_id)["reply_method"] == "slack_button"
    assert "Reply recorded" in gateway.posts[-1]["text"]

    # 4. pipeline appends transitions; bridge broadcasts to #pi05-status
    tfile = cfg.runs_root / run_id / "transitions.jsonl"
    tfile.write_text("")
    bridge._offsets[tfile] = 0
    with open(tfile, "a") as fh:
        fh.write(json.dumps({"ts": "x", "run_id": run_id, "node": "filter_build",
                             "status": "succeeded", "detail": "836 episodes"}) + "\n")
        fh.write(json.dumps({"ts": "y", "run_id": run_id, "status": "done"}) + "\n")
    bridge.dispatch("fs_transitions", tfile)
    status_posts = [p for p in gateway.posts if p["channel"] == "C_STATUS"]
    assert any("836 episodes" in p["text"] for p in status_posts)
    assert db.get_run(run_id)["status"] == "done"

    # 5. a second click on the long-resolved button is rejected ephemerally
    bridge.dispatch("block_actions", click_payload(esc_id, 2, thread_ts="1.000001", user="U_BOB"))
    assert "already answered" in gateway.ephemerals[-1]["text"]
    assert reply_file.read_text().strip() == "1"


def test_round_trip_cli_reply_wins_race(cfg, db, gateway, spawned):
    bridge = make_bridge(cfg, db, gateway, spawned)
    run_id = "race_run"  # launched outside Slack; bridge creates a tracking thread on contact
    esc_id = "norm_stats_1_zz"
    write_question(cfg.runs_root, run_id, esc_id)
    bridge.initial_scan()
    drain(bridge)
    # CLI reply lands BEFORE anyone clicks (spec 13.9)
    reply = cfg.runs_root / run_id / "escalations" / f"{esc_id}.reply.txt"
    reply.write_text("3\n")
    bridge.dispatch("fs_reply", reply)
    assert db.get_escalation(esc_id)["reply_method"] == "file_drop"
    assert any("CLI/file drop" in p["text"] for p in gateway.posts)  # in-thread ack
    bridge.dispatch("block_actions", click_payload(esc_id, 1, thread_ts="1.000001"))
    assert reply.read_text().strip() == "3"
    assert "already answered" in gateway.ephemerals[-1]["text"]


def test_service_without_tokens_exits_cleanly(tmp_path):
    """Addendum B5 / acceptance 2: clear setup message, exit code 2, no traceback."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLACK_")}
    env["EMBODIED_CLAW_ROOT"] = str(tmp_path)  # no .env there
    proc = subprocess.run(
        [sys.executable, "-m", "bridge.service"], env=env, capture_output=True,
        text=True, timeout=60, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    assert proc.returncode == 2
    assert "SLACK_BOT_TOKEN" in proc.stderr and "bridge/README.md" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_load_config_reports_missing(tmp_path):
    import pytest
    from bridge.config import MissingConfig, load_config
    with pytest.raises(MissingConfig) as exc:
        load_config({"SLACK_BOT_TOKEN": "xoxb-REPLACE-ME"}, tmp_path)
    assert set(exc.value.missing) == {"SLACK_APP_TOKEN", "SLACK_REQUESTS_CHANNEL_ID"}


def test_runs_root_falls_back_to_pipeline_env(tmp_path):
    """One env var must not split the shared mailbox: BRIDGE_RUNS_ROOT > EMBODIED_CLAW_RUNS
    > <repo>/runs."""
    from bridge.config import load_config
    base = {"SLACK_BOT_TOKEN": "x", "SLACK_APP_TOKEN": "y", "SLACK_REQUESTS_CHANNEL_ID": "C"}
    assert load_config(base, tmp_path).runs_root == tmp_path / "runs"
    cfg = load_config({**base, "EMBODIED_CLAW_RUNS": "/elsewhere/runs"}, tmp_path)
    assert str(cfg.runs_root) == "/elsewhere/runs"
    cfg = load_config({**base, "EMBODIED_CLAW_RUNS": "/elsewhere/runs",
                       "BRIDGE_RUNS_ROOT": "/bridge/runs"}, tmp_path)
    assert str(cfg.runs_root) == "/bridge/runs"


def test_allowed_users_toml_tolerates_absence_and_garbage(tmp_path):
    from bridge.config import load_allowed_users
    assert load_allowed_users(tmp_path) is None  # no config dir at all
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "slack.toml").write_text("not [ valid toml ===")
    assert load_allowed_users(tmp_path) is None  # malformed -> disabled, no crash
    (cfg_dir / "slack.toml").write_text('allowed_users = ["U_BOSS", "U_ALICE"]\n')
    assert load_allowed_users(tmp_path) == {"U_BOSS", "U_ALICE"}

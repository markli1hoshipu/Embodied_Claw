"""CLI surface: inbox, reply, run guards, --detach."""
import json

import pytest

from pipeline import cli, tools


def test_inbox_empty_clean_output(env, capsys):
    cli.main(["inbox"])
    assert capsys.readouterr().out.strip() == "inbox empty — no pending escalations."


def test_help_usable(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "run" in out and "reply" in out and "inbox" in out


def test_reply_by_option_and_node(run_dir, capsys):
    eid = tools.new_escalation_id("filter_build")
    tools.write_question(run_dir, eid, node="filter_build", agent="data_agent", question="q?",
                         options=[{"id": 1, "label": "p98"}])
    cli.main(["reply", "--run-id", "tr1", "--node", "filter_build", "--option", "1"])
    assert "reply recorded for [tr1:filter_build]" in capsys.readouterr().out
    p = run_dir / "escalations" / f"{eid}.reply.txt"
    assert p.read_text() == "1"
    assert tools.read_reply(run_dir, eid) == {"type": "option", "option": 1}


def test_reply_requires_target_and_payload(env, run_dir):
    with pytest.raises(SystemExit, match="no pending escalation"):
        cli.main(["reply", "--run-id", "tr1", "--node", "train", "--option", "1"])
    with pytest.raises(SystemExit, match="--option N or --message"):
        cli.main(["reply", "--latest"])


def test_inbox_lists_pending_with_run_node_prefix(run_dir, capsys):
    eid = tools.new_escalation_id("train")
    tools.write_question(run_dir, eid, node="train", agent="training_agent", question="loss ok?")
    cli.main(["inbox"])
    out = capsys.readouterr().out
    assert "[tr1:train]" in out and "loss ok?" in out and eid in out


def test_run_requires_request_txt(env):
    with pytest.raises(SystemExit, match="no request found"):
        cli.main(["run", "ghost_run"])


def test_run_detach_relaunches_via_setsid_session(run_dir, monkeypatch, capsys):
    import subprocess
    seen = {}

    def popen(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return None
    monkeypatch.setattr(subprocess, "Popen", popen)
    cli.main(["run", "tr1", "--detach"])
    assert seen["cmd"][1:] == ["-m", "pipeline.cli", "run", "tr1"]
    assert seen["kw"]["start_new_session"] is True
    assert "detached driver started" in capsys.readouterr().out


def test_run_writes_request_from_flag(env, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: None)
    cli.main(["run", "fresh1", "--request", "train pi05 on task-0", "--detach"])
    assert (tools.run_dir("fresh1") / "request.txt").read_text() == "train pi05 on task-0"


def test_single_instance_lock(graph_env, run_dir, monkeypatch):
    import fcntl
    lock = open(run_dir / ".driver.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(SystemExit, match="refusing to run"):
            cli.main(["run", "tr1"])
    finally:
        lock.close()


def test_cmd_run_full_cycle_and_rerun_noop(graph_env, run_dir, capsys):
    graph_env()  # patches the anthropic client; cmd_run builds its own graph on the same env
    cli.main(["run", "tr1", "--poll", "0.01"])
    out = capsys.readouterr().out
    assert "=== tr1 ===" in out and out.count("succeeded") == 7
    assert json.loads((run_dir / "artifacts.json").read_text())
    cli.main(["run", "tr1"])  # all succeeded -> explicit no-op path
    assert "already completed" in capsys.readouterr().out

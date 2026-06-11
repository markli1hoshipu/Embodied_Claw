"""Training operator (pipeline.operator): pure impl functions + launch gating.
No claude-agent-sdk session is exercised here — the impls are plain functions; the SDK
loop is the same battle-tested shape as eval_domino.operator."""
from __future__ import annotations

import json

from pipeline import operator as op
from pipeline import tools
from pipeline.tests.fakes import MINIMAL_CONFIG


# ------------------------------------------------------------------ checkpoint_status
def _mk_ckpts(root):
    exp = root / "pi05_b1k_demo_task0" / "curated_lr2.5e5"
    for step in (10000, 20000, 59999):
        (exp / str(step)).mkdir(parents=True)
    (exp / "59999" / "_CHECKPOINT_METADATA").write_text("{}")
    half = root / "pi05_b1k_other_task0" / "curated_lr2.5e5"
    (half / "10000").mkdir(parents=True)  # no metadata: incomplete final
    return exp


def test_checkpoint_status_reports_steps_and_completeness(tmp_path):
    _mk_ckpts(tmp_path)
    rows = op.checkpoint_status_impl(ckpt_root=tmp_path, logs_dir=tmp_path, live=lambda c: [])
    by = {r["train_config"]: r for r in rows}
    demo = by["pi05_b1k_demo_task0"]
    assert demo["steps"] == [10000, 20000, 59999]
    assert demo["final_step"] == 59999 and demo["final_complete"] is True
    assert demo["live_pids"] == [] and "last_progress" not in demo
    other = by["pi05_b1k_other_task0"]
    assert other["final_step"] == 10000 and other["final_complete"] is False


def test_checkpoint_status_live_run_includes_progress(tmp_path):
    _mk_ckpts(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pi05_b1k_demo_task0_20260611_000000.log").write_text(
        "boot\r18:41 [I] Progress on: 7.24kit/60.0kit rate:3.4it/s\rnoise\n")
    rows = op.checkpoint_status_impl(
        ckpt_root=tmp_path, logs_dir=logs,
        live=lambda c: [4242] if c == "pi05_b1k_demo_task0" else [])
    demo = next(r for r in rows if r["train_config"] == "pi05_b1k_demo_task0")
    assert demo["live_pids"] == [4242]
    assert "Progress on: 7.24kit/60.0kit" in demo["last_progress"]
    # a live() failure must degrade to [] (status tool never raises), not propagate
    rows = op.checkpoint_status_impl(ckpt_root=tmp_path, logs_dir=logs,
                                     live=lambda c: (_ for _ in ()).throw(RuntimeError("x")))
    assert all(r["live_pids"] == [] for r in rows)


# ------------------------------------------------------------------ dataset_status
def test_dataset_status_lists_lerobot_dirs(tmp_path):
    d = tmp_path / "b1k_demo_task0_curated"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps(
        {"total_episodes": 836, "total_frames": 530000}))
    (d / "norm_stats.json").write_text("{}")
    (tmp_path / "not_a_dataset").mkdir()  # no meta/info.json -> skipped
    rows = op.dataset_status_impl(root=tmp_path)
    assert rows == [{"dataset": "b1k_demo_task0_curated", "episodes": 836,
                     "frames": 530000, "norm_stats": True}]


# ------------------------------------------------------------------ runs
def test_run_status_and_list_runs(env):
    rd = tools.run_dir("tr9")
    rd.mkdir(parents=True)
    (rd / "request.txt").write_text("train something")
    (rd / "config.json").write_text(json.dumps({**MINIMAL_CONFIG, "run_id": "tr9"}))
    tools.log_transition("tr9", "ingest", "succeeded")
    tools.log_transition("tr9", "filter_build", "running")
    st = op.run_status_impl("tr9")
    assert st["run_id"] == "tr9" and len(st["last_transitions"]) == 2
    assert st["pending_escalations"] == []
    assert op.run_status_impl("nope")["error"].startswith("unknown run")
    runs = op.list_runs_impl()
    assert {"run_id": "tr9", "kind": "train", "last": runs[0]["last"]} == runs[0]
    # operator threads are invisible
    (tools.runs_root() / "_operator" / "1.2" / "inbox").mkdir(parents=True)
    assert all(r["run_id"] != "_operator" for r in op.list_runs_impl())


# ------------------------------------------------------------------ launch gating
def test_launch_requires_api_key(env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = op.launch_train_run_impl("111.222", "tr_new", "req", json.dumps(MINIMAL_CONFIG),
                                   spawn=lambda rid: (_ for _ in ()).throw(AssertionError))
    assert "ANTHROPIC_API_KEY" in res["error"]
    assert not (tools.run_dir("tr_new") / "request.txt").exists()


def test_launch_writes_confirmed_config_and_spawns(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    spawned = []
    res = op.launch_train_run_impl("111.222", "tr_new", "train it please",
                                   json.dumps(MINIMAL_CONFIG), spawn=spawned.append)
    assert res["ok"] and spawned == ["tr_new"]
    rd = tools.run_dir("tr_new")
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["_confirmed"] is True and cfg["run_id"] == "tr_new"
    assert (rd / "request.txt").read_text().startswith("train it please")
    assert (rd / ".operator_thread").read_text() == "111.222"
    # second launch with the same run_id refuses
    res2 = op.launch_train_run_impl("111.222", "tr_new", "again", json.dumps(MINIMAL_CONFIG),
                                    spawn=spawned.append)
    assert "already exists" in res2["error"] and spawned == ["tr_new"]


def test_launch_rejects_invalid_config(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    bad = {"description": "x"}  # missing data_request/train_request/outputs
    res = op.launch_train_run_impl("1.2", "tr_bad", "req", json.dumps(bad),
                                   spawn=lambda rid: None)
    assert res["error"] == "config invalid" and "missing" in json.dumps(res["details"])
    res2 = op.launch_train_run_impl("1.2", "tr_bad2", "req", "{not json",
                                    spawn=lambda rid: None)
    assert "not valid JSON" in res2["error"]


# ------------------------------------------------------------------ intake fast path
def test_intake_short_circuits_on_operator_confirmed_config(env, monkeypatch):
    from pipeline.nodes import intake
    rd = tools.run_dir("tr_op")
    rd.mkdir(parents=True)
    cfg = {**MINIMAL_CONFIG, "run_id": "tr_op", "_confirmed": True}
    (rd / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr("pipeline.nodes.intake.make_agent",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no agent!")))
    out = intake.node({"run_id": "tr_op", "intake": {"status": "pending"}})
    assert out["intake"]["status"] == "succeeded" and out["config"]["_confirmed"]
    # unconfirmed config still goes to the agent (which our stub makes explode)
    (rd / "config.json").write_text(json.dumps({**cfg, "_confirmed": False}))
    try:
        intake.node({"run_id": "tr_op", "intake": {"status": "pending"}})
        raise SystemExit("agent path not taken")
    except AssertionError:
        pass

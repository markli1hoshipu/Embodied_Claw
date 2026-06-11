"""Smoke tests for every section-5 training skill — subprocess fully mocked, no GPUs touched."""
import json
import types

from pipeline import tools
from pipeline.skills import training_skills as t

CONFIG_PY = """_CONFIGS = [
    TrainConfig(
        name="existing_cfg",
    ),
]
"""


def _rc(code=0, out="", err=""):
    return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_ensure_train_config_present_inserts_once(env, tmp_path):
    cp = tmp_path / "config.py"
    cp.write_text(CONFIG_PY)
    out = t.ensure_train_config_present("pi05_b1k_new", "/data/lerobot/new", batch_size=32,
                                        peak_lr=2.5e-5, num_train_steps=60000,
                                        config_path=str(cp))
    assert out["added"]
    text = cp.read_text()
    assert 'name="pi05_b1k_new"' in text and 'repo_id="/data/lerobot/new"' in text
    assert "num_workers=48" in text and "wandb_enabled=False" in text
    assert "decay_lr=2.5e-06" in text and "num_train_steps=60000" in text
    assert "fsdp_devices=1" in text and "save_interval=1000" in text  # openpi defaults
    assert "checkpoints/pi05_base/params" in text
    assert text.index('name="pi05_b1k_new"') < text.rindex("]")  # inside _CONFIGS
    assert 'name="existing_cfg"' in text
    again = t.ensure_train_config_present("pi05_b1k_new", "x", config_path=str(cp))
    assert again == {"added": False, "config_block": ""}


def test_ensure_train_config_honors_structured_train_request(env, tmp_path):
    """Spec section 1: wandb_enabled/fsdp_devices/save_interval/base_model are exact values —
    the user's confirmed intent must land in the TrainConfig verbatim, never be overridden."""
    cp = tmp_path / "config.py"
    cp.write_text(CONFIG_PY)
    out = t.ensure_train_config_present("pi05_b1k_w", "/d/lerobot/w", wandb_enabled=True,
                                        fsdp_devices=8, save_interval=10000,
                                        base_model="pi05_droid", config_path=str(cp))
    block = out["config_block"]
    assert "wandb_enabled=True" in block and "fsdp_devices=8" in block
    assert "save_interval=10000" in block
    assert "checkpoints/pi05_droid/params" in block
    assert "num_workers=48" in block  # learned gotcha stays


def test_compute_norm_stats_fast_always_caps_frames(env, tmp_path, monkeypatch):
    seen = {}

    def sh(cmd, env=None, timeout=None):
        seen["cmd"] = cmd
        return _rc(0)
    monkeypatch.setattr(tools, "sh", sh)
    (tmp_path / "norm_stats.json").write_text(json.dumps({"pad": "x" * 2000}))
    out = t.compute_norm_stats_fast("cfg_x", dataset_dir=str(tmp_path))
    assert "--max-frames 50000" in seen["cmd"] and "compute_norm_stats.py" in seen["cmd"]
    assert out["norm_stats_path"] == str(tmp_path / "norm_stats.json")
    assert out["sample_size"] == 50000
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: _rc(1, err="decode boom"))
    assert "decode boom" in t.compute_norm_stats_fast("cfg_x", dataset_dir=str(tmp_path))["error"]


def test_verify_norm_stats_sane(env, tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({"norm_stats": {"actions": {"mean": [0.1]}, "state": {"std": [1.0]}}}))
    assert t.verify_norm_stats_sane(str(p)) == {"ok": True, "nan_keys": [], "missing_keys": []}
    p.write_text('{"norm_stats": {"actions": {"mean": [NaN]}, "state": {"q01": [0.0]}}}')
    out = t.verify_norm_stats_sane(str(p))
    assert not out["ok"] and out["nan_keys"] == ["norm_stats.actions.mean[0]"]
    p.write_text(json.dumps({"actions": {}}))
    assert t.verify_norm_stats_sane(str(p))["missing_keys"] == ["state"]
    assert not t.verify_norm_stats_sane(str(tmp_path / "missing.json"))["ok"]


def test_preflight_nccl_check(env, monkeypatch):
    monkeypatch.setattr(tools, "live_pids", lambda c: [])
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: _rc(0, out="NCCL OK\n"))
    assert t.preflight_nccl_check("cfg") == {"ok": True, "error": None, "suggested_env": {}}
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: _rc(1, err="cuda failure 401 nvls.cc"))
    out = t.preflight_nccl_check("cfg")
    assert not out["ok"] and out["suggested_env"] == {"NCCL_NVLS_ENABLE": "0"}


def test_preflight_suppressed_while_matching_run_live(env, monkeypatch):
    """The 95%-mem GPU probe must NEVER run under a live matching train (attach scenario)."""
    monkeypatch.setattr(tools, "live_pids", lambda c: [200738, 200750])
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("probe must not touch GPUs while the run is live")))
    out = t.preflight_nccl_check("pi05_cfg")
    assert out["ok"] and "suppressed" in out["skipped"]


def test_clone_train_script_if_needed(env, tmp_path):
    tpl = tmp_path / "train_old.sh"
    tpl.write_text("#!/bin/bash\nuv run scripts/train.py --config-name old_cfg --overwrite\n")
    out = t.clone_train_script_if_needed(str(tpl), "train_new.sh", "new_cfg",
                                         scripts_dir=str(tmp_path))
    text = (tmp_path / "train_new.sh").read_text()
    assert out["created"] and "--config-name new_cfg" in text and "old_cfg" not in text
    again = t.clone_train_script_if_needed(str(tpl), "train_new.sh", "other",
                                           scripts_dir=str(tmp_path))
    assert not again["created"]  # existing script untouched


def test_launch_attaches_instead_of_launching(env, monkeypatch):
    monkeypatch.setattr(tools, "live_pids", lambda c: [200738, 200750])
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("must not shell out when a live run matches")))
    out = t.launch_detached_train("s.sh", "/tmp/x.log", "pi05_cfg")
    assert out == {"attached": True, "pid": 200738, "pids": [200738, 200750],
                   "log_path": "/tmp/x.log"}


def test_launch_fresh_uses_setsid_and_waits_for_log(env, tmp_path, monkeypatch):
    log = tmp_path / "train.log"
    state = {"launched": False}

    def live(c):
        return [777] if state["launched"] else []

    def sh(cmd, env=None, timeout=None):
        assert "setsid bash" in cmd and "disown" in cmd and "s.sh" in cmd
        log.write_text("")
        state["launched"] = True
        return _rc(0)
    monkeypatch.setattr(tools, "live_pids", live)
    monkeypatch.setattr(tools, "sh", sh)
    out = t.launch_detached_train("s.sh", str(log), "cfg")
    assert out == {"attached": False, "pid": 777, "log_path": str(log)}


def test_launch_never_returns_none_pid_during_pgrep_blind_window(env, tmp_path, monkeypatch):
    """Between `setsid bash` and the python exec, pgrep matches nothing: launch must keep
    polling (and error after the startup deadline) — never report success with pid=None,
    which would invite a duplicate --overwrite launch."""
    log = tmp_path / "train.log"
    monkeypatch.setattr(tools, "live_pids", lambda c: [])  # python never appears
    monkeypatch.setattr(tools, "sh", lambda *a, **kw: (log.write_text("uv resolving...\n"),
                                                       _rc(0))[1])
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    out = t.launch_detached_train("s.sh", str(log), "cfg", startup_timeout_s=0)
    assert "no matching train process appeared" in out["error"]
    assert "pid" not in out  # an error dict, not a half-success


def test_monitor_done_and_crash(env, tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    final = ckpt / "59999"
    (final / "params").mkdir(parents=True)
    (final / "_CHECKPOINT_METADATA").write_text("")
    log = tmp_path / "t.log"
    log.write_text("old stuff\n")
    out = t.monitor_train_progress(str(log), str(ckpt), 60000, "cfg", poll_interval=0)
    assert out["status"] == "done" and out["last_step"] == 59999
    # crash path: process gone, final ckpt absent, evidence only from bytes AFTER the mark
    import shutil
    shutil.rmtree(final)
    (ckpt / "30000").mkdir()
    mark = log.stat().st_size
    calls = {"n": 0}

    def live_then_die(c):  # the crash lands in the log after monitor records its mark
        calls["n"] += 1
        with open(log, "a") as f:
            f.write("E0610 cuda failure 401 at nvls.cc\n")
        return []
    monkeypatch.setattr(tools, "live_pids", live_then_die)
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    out = t.monitor_train_progress(str(log), str(ckpt), 60000, "cfg", poll_interval=0)
    assert out["status"] == "crashed" and out["last_step"] == 30000
    assert out["since_byte"] == mark and "401" in out["log_tail"]
    assert out["final_loss"] is None  # spec crashed shape carries final_loss
    assert calls["n"] == 3  # 3 consecutive empty-pid polls before declaring death


def _live_writer(log, line, pids):
    """live_pids stub that appends `line` to the log on its first call — loss evidence must
    land AFTER the monitor records its stale-log byte mark."""
    state = {"written": False}

    def live(c):
        if not state["written"]:
            state["written"] = True
            with open(log, "a") as f:
                f.write(line)
        return pids
    return live


def test_monitor_returns_running_periodically_and_on_deadline(env, tmp_path, monkeypatch):
    """Spec section 5: the monitor yields progress events — a live slow run is 'running'
    (both at max_polls and at the deadline), NEVER 'crashed'."""
    ckpt = tmp_path / "ckpt"
    (ckpt / "10000").mkdir(parents=True)
    log = tmp_path / "t.log"
    log.write_text("")
    monkeypatch.setattr(tools, "live_pids",
                        _live_writer(log, "step 10000 loss=0.012\n", [200738]))
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    out = t.monitor_train_progress(str(log), str(ckpt), 60000, "cfg", poll_interval=0,
                                   max_polls=3)
    assert out["status"] == "running" and out["last_step"] == 10000
    assert out["recent_loss"] == 0.012
    monkeypatch.setattr(tools, "live_pids", lambda c: [200738])
    out = t.monitor_train_progress(str(log), str(ckpt), 60000, "cfg", poll_interval=0,
                                   max_polls=5, deadline_s=0)
    assert out["status"] == "running"  # deadline expiry with live pids is not a crash


def test_monitor_short_circuits_on_divergence(env, tmp_path, monkeypatch):
    """Node-5 ESCALATE trigger: NaN or >10x end-of-run loss must surface mid-training."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    log = tmp_path / "t.log"
    log.write_text("")
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    monkeypatch.setattr(tools, "live_pids", _live_writer(log, "step 500 loss=nan\n", [1]))
    out = t.monitor_train_progress(str(log), str(ckpt), 60000, "cfg", poll_interval=0)
    assert out["status"] == "diverged" and str(out["loss"]) == "nan"
    log2 = tmp_path / "t2.log"
    log2.write_text("")
    monkeypatch.setattr(tools, "live_pids", _live_writer(log2, "step 30000 loss=0.5\n", [1]))
    out = t.monitor_train_progress(str(log2), str(ckpt), 60000, "cfg", poll_interval=0,
                                   loss_ceiling=0.08)
    assert out["status"] == "diverged" and out["loss"] == 0.5


def test_classify_crash_categories_and_stale_gate(env, tmp_path):
    log = tmp_path / "t.log"
    log.write_text("boot\ncuda failure 401 at nvls.cc\n")
    assert t.classify_train_crash(str(log))["category"] == "nccl_nvls"
    # stale-log gate: same 401 BEFORE the mark must NOT classify as a fresh nccl crash
    assert t.classify_train_crash(str(log), since_byte=log.stat().st_size)["category"] == "unknown"
    log.write_text("RESOURCE_EXHAUSTED: out of memory allocating\n")
    assert t.classify_train_crash(str(log))["category"] == "oom"
    log.write_text("torchcodec decode error on episode 42\n")
    assert t.classify_train_crash(str(log))["category"] == "data_loader"
    log.write_text("Traceback: KeyError 'foo'\n")
    out = t.classify_train_crash(str(log))
    assert out["category"] == "unknown" and "KeyError" in out["evidence"]


def test_restart_with_workaround(env, monkeypatch):
    calls = {}
    rechecks = {"n": 0}

    def live(c):
        rechecks["n"] += 1
        return []
    monkeypatch.setattr(tools, "live_pids", live)
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    monkeypatch.setattr(t, "launch_detached_train",
                        lambda s, l, c: calls.update(args=(s, l, c)) or {"attached": False, "pid": 9})
    out = t.restart_with_workaround("nccl_nvls", "s.sh", "t.log", "cfg")
    assert out["pid"] == 9 and out["applied_env"] == {"NCCL_NVLS_ENABLE": "0"}
    assert calls["args"] == ("s.sh", "t.log", "cfg")
    assert rechecks["n"] == 3  # cooldown: liveness re-confirmed empty 3x before relaunching
    for cat in ("oom", "data_loader", "unknown"):
        assert "escalate" in t.restart_with_workaround(cat, "s.sh", "t.log", "cfg")["error"]


def test_restart_attaches_if_pid_reappears_during_cooldown(env, monkeypatch):
    """A 'crash' observed in the pgrep blind window resolves to attach, never double-launch."""
    seq = iter([[], [4242]])
    monkeypatch.setattr(tools, "live_pids", lambda c: next(seq))
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    monkeypatch.setattr(t, "launch_detached_train", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not relaunch — the run came back")))
    out = t.restart_with_workaround("nccl_nvls", "s.sh", "t.log", "cfg")
    assert out["attached"] and out["pid"] == 4242

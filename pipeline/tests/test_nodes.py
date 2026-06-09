import hashlib
import json
import re
import shlex
import shutil

import helpers as H
import pytest

from pipeline import nodes


def _boom(*a, **k):
    raise AssertionError("must not be called")


def _no_sleep(monkeypatch):
    monkeypatch.setattr(nodes.time, "sleep", lambda s: None)


# ---------------------------------------------------------------- ingest

def test_ingest_skips_when_sources_ready(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    monkeypatch.setattr(nodes, "_sh", _boom)
    out = nodes.ingest_source({"config": cfg})
    assert out["ingest"]["status"] == "skipped"
    assert out["ingest"]["artifact_paths"] == [s["local_dir"] for s in cfg["sources"]]


def test_ingest_fails_without_hf_token(cfg, sandbox):
    out = nodes.ingest_source({"config": cfg})
    assert out["ingest"]["status"] == "failed"
    assert "HF_TOKEN" in out["ingest"]["error"]


def test_ingest_429_raises_transient_for_graph_retry(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "_sh", lambda cmd, env=None, timeout=None: H.CP(1, "", "429 Too Many Requests"))
    with pytest.raises(nodes.TransientHFError):
        nodes.ingest_source({"config": cfg})


def test_ingest_fallback_429_raises_transient(cfg, sandbox, monkeypatch):
    """A 429 hit only by the per-file fallback must ALSO reach the graph RetryPolicy."""
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def fake_sh(cmd, env=None, timeout=None):
        if "snapshot_download" in cmd:
            return H.CP(1, "", "boom (connection reset, non-rate-limit)")
        return H.CP(1, "", "429 Too Many Requests")

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    with pytest.raises(nodes.TransientHFError):
        nodes.ingest_source({"config": cfg})


def test_ingest_downloads_with_env_and_verifies(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "_hf_api",
                        lambda token=None: H.FakeApi(files={"org/official": H.official_repo_files()}))
    cmds = []

    def fake_sh(cmd, env=None, timeout=None):
        cmds.append(cmd)
        assert env["HF_TOKEN"] == "hf_test" and env["HF_HUB_DISABLE_XET"] == "1"
        if "snapshot_download" in cmd:
            H.make_official(cfg)   # only the tree THIS command downloads
        else:
            H.make_perturb(cfg)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.ingest_source({"config": cfg})
    assert out["ingest"]["status"] == "succeeded"
    assert len(cmds) == 2  # one download per source — the zip flavor must actually execute
    assert "snapshot_download" in cmds[0] and "conda activate xvla-stable" in cmds[0]
    assert "hf_hub_download" in cmds[1] and "out.zip" in cmds[1] and "unzip -q -o" in cmds[1]
    # expected counts persisted for both sources so later partial states are detectable
    for src in cfg["sources"]:
        assert nodes._read_manifest(nodes._source_root(src))["parquets"] >= 3


def test_ingest_snapshot_failure_uses_per_file_fallback(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "_hf_api",
                        lambda token=None: H.FakeApi(files={"org/official": H.official_repo_files()}))
    cmds = []

    def fake_sh(cmd, env=None, timeout=None):
        cmds.append(cmd)
        if "snapshot_download" in cmd:
            return H.CP(1, "", "boom (connection reset, non-rate-limit)")
        if "list_repo_files" in cmd:   # the per-file fallback
            H.make_official(cfg)
            return H.CP(0)
        H.make_perturb(cfg)            # the zip source
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.ingest_source({"config": cfg})
    assert out["ingest"]["status"] == "succeeded"
    assert len(cmds) == 3  # snapshot (failed) + fallback + zip
    assert any("hf_hub_download" in c and "list_repo_files" in c for c in cmds)
    assert "unzip -q -o" in cmds[2]


def test_ingest_fails_when_download_short_of_repo_listing(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "_hf_api",
                        lambda token=None: H.FakeApi(files={"org/official": H.official_repo_files(9)}))

    def fake_sh(cmd, env=None, timeout=None):
        if "snapshot_download" in cmd:
            H.make_official(cfg)  # lands only 4 of the 9 repo-side parquets
        else:
            H.make_perturb(cfg)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.ingest_source({"config": cfg})
    assert out["ingest"]["status"] == "failed"
    assert "incomplete" in out["ingest"]["error"]


def test_partial_download_detected_via_manifest(cfg, sandbox):
    """A truncated tree must not pass source_ready once expected counts are pinned."""
    H.make_sources(cfg)  # 4 official parquets on disk
    root = nodes._source_root(cfg["sources"][0])
    assert nodes.source_ready(cfg["sources"][0])  # pre-pipeline data, no manifest -> weak fallback
    nodes._manifest_path(root).write_text(json.dumps({"parquets": 9, "mp4s": 0, "annotations": 0}))
    assert not nodes.source_ready(cfg["sources"][0])  # 4 < 9 -> not ready
    assert nodes.needs_ingest(cfg)


# ---------------------------------------------------------------- filter_build

def test_build_skips_when_info_matches_expected(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    monkeypatch.setattr(nodes, "_sh", _boom)  # destructive builder must NOT run
    monkeypatch.setattr(nodes, "find_train_pids", _boom)
    out = nodes.filter_and_build({"config": cfg})
    assert out["filter_build"]["status"] == "skipped"
    assert "episodes=13" in out["filter_build"]["artifact_paths"][1]


def test_build_refuses_destructive_rebuild_while_training_live(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)  # no info.json -> needs_build is True
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [4242])
    monkeypatch.setattr(nodes, "_sh", _boom)  # the builder must NOT run
    out = nodes.filter_and_build({"config": cfg})
    assert out["filter_build"]["status"] == "failed"
    assert "refusing destructive rebuild" in out["filter_build"]["error"]


def test_build_runs_and_verifies(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])

    def fake_sh(cmd, env=None, timeout=None):
        if "find -L" in cmd:
            return H.CP(0, "", "")  # no broken symlinks
        assert "python3" in cmd and cfg["builder_script"] in cmd
        H.make_built(nodes, cfg, episodes=13)
        return H.CP(0, "[done]")

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.filter_and_build({"config": cfg})
    assert out["filter_build"]["status"] == "succeeded"


def test_build_fails_on_broken_video_symlinks(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])

    def fake_sh(cmd, env=None, timeout=None):
        if "find -L" in cmd:
            return H.CP(0, "/x/ep0.mp4\n")  # one broken symlink
        H.make_built(nodes, cfg, episodes=13)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.filter_and_build({"config": cfg})
    assert out["filter_build"]["status"] == "failed"
    assert "broken video symlinks" in out["filter_build"]["error"]


def test_build_fails_on_episode_mismatch(cfg, sandbox, monkeypatch):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])

    def fake_sh(cmd, env=None, timeout=None):
        H.make_built(nodes, cfg, episodes=12)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.filter_and_build({"config": cfg})
    assert out["filter_build"]["status"] == "failed"
    assert "expected 13" in out["filter_build"]["error"]


# ---------------------------------------------------------------- norm_stats

def test_norm_skips_when_present(cfg, sandbox, monkeypatch):
    H.make_norm(nodes, cfg)
    monkeypatch.setattr(nodes, "_sh", _boom)
    assert nodes.compute_norm_stats({"config": cfg})["norm_stats"]["status"] == "skipped"


def test_norm_runs_fast_path(cfg, sandbox, monkeypatch):
    def fake_sh(cmd, env=None, timeout=None):
        assert "compute_norm_stats.py" in cmd and "--max-frames 50000" in cmd and "uv run" in cmd
        assert "ffmpeg7/lib" in env["LD_LIBRARY_PATH"]
        H.make_norm(nodes, cfg, size=2048)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.compute_norm_stats({"config": cfg})
    assert out["norm_stats"]["status"] == "succeeded"


def test_norm_fails_when_output_too_small(cfg, sandbox, monkeypatch):
    def fake_sh(cmd, env=None, timeout=None):
        H.make_norm(nodes, cfg, size=10)
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    out = nodes.compute_norm_stats({"config": cfg})
    assert out["norm_stats"]["status"] == "failed"


# ---------------------------------------------------------------- train helpers (pgrep/launch/preflight)

def test_find_train_pids_pgrep_invocation_and_parsing(monkeypatch):
    seen = {}

    def fake_run(argv, capture_output=None, text=None):
        seen["argv"] = argv
        return H.CP(0, "123\n456\n")

    monkeypatch.setattr(nodes.subprocess, "run", fake_run)
    assert nodes.find_train_pids("pi05_b1k_test") == [123, 456]
    assert seen["argv"][:2] == ["pgrep", "-f"]
    pat = seen["argv"][2]
    assert pat.startswith(r"python.*scripts/train\.py")  # never a bare substring match
    assert pat.endswith("( |$)")                          # trailing boundary (prefix collisions)


def test_find_train_pids_no_match_vs_pgrep_error(monkeypatch):
    monkeypatch.setattr(nodes.subprocess, "run", lambda *a, **k: H.CP(1, "", ""))
    assert nodes.find_train_pids("cfg_x") == []     # rc=1: genuinely no match
    monkeypatch.setattr(nodes.subprocess, "run", lambda *a, **k: H.CP(2, "", "pgrep: regex error"))
    with pytest.raises(RuntimeError):               # rc>=2: pgrep itself failed, NOT 'no process'
        nodes.find_train_pids("cfg_x")
    assert nodes._live_pids("cfg_x") == [-1]        # fail-safe: read as 'assume alive'


def test_pgrep_pattern_regex_semantics():
    name = "pi05_b1k_perturb_recovery3_task0_curated"
    pat = rf"python.*scripts/train\.py.*{re.escape(name)}( |$)"
    assert re.search(pat, f"uv run python scripts/train.py {name} --exp-name x")
    assert re.search(pat, f"python scripts/train.py {name}")  # end-of-line boundary
    # bash watcher carrying the literal script path must NOT match (spec failure mode 3)
    assert not re.search(pat, f"bash -c tail -f /tmp/x | grep Progress scripts/train.py {name} log")
    # prefix-sibling config names must NOT cross-match
    short = rf"python.*scripts/train\.py.*{re.escape('pi05_recovery')}( |$)"
    assert not re.search(short, "python scripts/train.py pi05_recovery3 --exp-name x")


def test_launch_detached_command_env_and_log(cfg, sandbox, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])
    seen = {}

    def fake_sh(cmd, env=None, timeout=None):
        seen["cmd"], seen["env"] = cmd, env
        nodes._tmp_log(cfg).write_text("")  # the >log redirect would create this for real
        return H.CP(0)

    monkeypatch.setattr(nodes, "_sh", fake_sh)
    nodes._launch(cfg)
    cmd = seen["cmd"]
    assert "setsid bash" in cmd and cfg["train_script"] in cmd          # mandatory detach pattern
    assert "</dev/null" in cmd and "2>&1" in cmd and "& disown" in cmd
    assert str(nodes._tmp_log(cfg)) in cmd                              # log redirect target
    assert "ffmpeg7/lib" in cmd and ".local/bin" in cmd                 # always-set env block
    assert seen["env"]["NCCL_NVLS_ENABLE"] == "0"


def test_launch_refuses_when_process_alive(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [4242])
    monkeypatch.setattr(nodes, "_sh", _boom)  # must never reach the shell
    with pytest.raises(RuntimeError, match="refusing to launch"):
        nodes._launch(cfg)


def test_preflight_psum_command_and_classification(monkeypatch):
    seen = {}

    def ok_sh(cmd, env=None, timeout=None):
        seen["cmd"], seen["env"] = cmd, env
        return H.CP(0, "NCCL OK\n")

    monkeypatch.setattr(nodes, "_sh", ok_sh)
    assert nodes._preflight() is None
    assert "uv run python -c" in seen["cmd"] and "psum" in seen["cmd"]
    assert seen["env"]["NCCL_NVLS_ENABLE"] == "0"
    monkeypatch.setattr(nodes, "_sh", lambda c, env=None, timeout=None: H.CP(0, ""))
    err = nodes._preflight()  # exit 0 but no 'NCCL OK' is still a failure
    assert err and "NCCL_NVLS_ENABLE=0" in err


# ---------------------------------------------------------------- train node

def test_train_skips_when_final_ckpt_exists(cfg, sandbox, monkeypatch):
    H.make_final_ckpt(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", _boom)
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "skipped"


def test_train_attach_never_launches(cfg, sandbox, monkeypatch, capsys):
    _no_sleep(monkeypatch)
    (nodes.ckpt_dir(cfg) / "10000").mkdir(parents=True)  # heartbeat's 2nd progress signal
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [4242])
    monkeypatch.setattr(nodes, "last_progress", lambda c: "Progress on: 2.45kit/60.0kit")
    monkeypatch.setattr(nodes, "_preflight", _boom)
    monkeypatch.setattr(nodes, "_launch", _boom)
    dones = iter([False, False, True])
    monkeypatch.setattr(nodes, "train_done", lambda c: next(dones))
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "succeeded"
    assert "ckpt_step=10000" in capsys.readouterr().out  # checkpoint dir is watched while polling


def test_train_dead_process_non_nccl_fails(cfg, sandbox, monkeypatch):
    _no_sleep(monkeypatch)
    pids = iter([[4242], []])
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: next(pids))
    monkeypatch.setattr(nodes, "last_progress", lambda c: None)
    monkeypatch.setattr(nodes, "_log_tail", lambda c, marks=None: "Traceback: some other crash")
    monkeypatch.setattr(nodes, "_launch", _boom)
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "failed"
    assert "died before final checkpoint" in out["train"]["error"]


def test_train_nccl_401_relaunches_then_succeeds(cfg, sandbox, monkeypatch):
    _no_sleep(monkeypatch)
    launches = []
    pids = iter([[4242], [], [5555]])
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: next(pids))
    monkeypatch.setattr(nodes, "last_progress", lambda c: None)
    monkeypatch.setattr(nodes, "_log_tail", lambda c, marks=None: "NCCL WARN ncclGroupEnd: Cuda failure 401")
    monkeypatch.setattr(nodes, "_launch", lambda c: launches.append(c["run_id"]))
    dones = iter([False, False, False, False, True])
    monkeypatch.setattr(nodes, "train_done", lambda c: next(dones))
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "succeeded"
    assert launches == ["testrun"]


def test_train_stale_401_log_not_reclassified(cfg, sandbox, monkeypatch):
    """A stale /tmp/train_<run_id>.log containing an old 401 must NOT trigger an auto-relaunch:
    classification only reads bytes written after the attach mark."""
    _no_sleep(monkeypatch)
    nodes._tmp_log(cfg).write_text("old run: NCCL WARN Cuda failure 401 at nvls.cc")
    pids = iter([[4242], []])
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: next(pids))
    monkeypatch.setattr(nodes, "last_progress", lambda c: None)
    monkeypatch.setattr(nodes, "_launch", _boom)  # relaunch must not happen
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "failed"
    assert "died before final checkpoint" in out["train"]["error"]


def test_train_fresh_launch_runs_preflight_first(cfg, sandbox, monkeypatch):
    _no_sleep(monkeypatch)
    calls = []
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])
    monkeypatch.setattr(nodes, "last_progress", lambda c: None)  # never probe real logs
    monkeypatch.setattr(nodes, "_sh", _boom)
    monkeypatch.setattr(nodes, "_preflight", lambda: (calls.append("preflight"), None)[1])
    monkeypatch.setattr(nodes, "_launch", lambda c: calls.append("launch"))
    dones = iter([False, True])
    monkeypatch.setattr(nodes, "train_done", lambda c: next(dones))
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "succeeded"
    assert calls == ["preflight", "launch"]


def test_train_preflight_failure_aborts(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])
    monkeypatch.setattr(nodes, "_preflight", lambda: "psum failed — NCCL_NVLS_ENABLE=0 hint")
    monkeypatch.setattr(nodes, "_launch", _boom)
    out = nodes.train({"config": cfg})
    assert out["train"]["status"] == "failed" and "psum" in out["train"]["error"]


# ---------------------------------------------------------------- upload

def test_upload_skips_when_repos_exist(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=True))
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "skipped"
    assert out["upload"]["artifact_paths"][0].endswith(cfg["hf_model_repo"])


def test_upload_fails_without_hf_token(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "failed" and "HF_TOKEN" in out["upload"]["error"]


def _fake_cp_al(cmd, env=None, timeout=None):
    if cmd.startswith("cp -al"):
        parts = shlex.split(cmd)
        shutil.copytree(parts[2], parts[3])
    return H.CP(0)


def _ulf_calls(api):
    return {c[1]["repo_id"]: c[1] for c in api.calls if c[0] == "upload_large_folder"}


def test_upload_stages_rounds_and_uploads(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    api = H.FakeApi(exists=True)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: api)
    monkeypatch.setattr(nodes, "_sh", _fake_cp_al)
    H.make_norm(nodes, cfg)
    for step in (30000, 40000, 50000, 59999):
        (nodes.ckpt_dir(cfg) / str(step) / "params").mkdir(parents=True)
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "succeeded"
    staged = sorted(p.name for p in (nodes.STAGING_ROOT / cfg["run_id"]).iterdir())
    assert staged == ["ckpt-30000", "ckpt-40000", "ckpt-50000", "ckpt-60000"]  # 59999 rounded
    ulf = _ulf_calls(api)
    assert ulf["Org/model"]["repo_type"] == "model"
    assert "ckpt-*/train_state/*" in ulf["Org/model"]["ignore_patterns"]   # bulk bytes excluded
    assert "ckpt-*/train_state/**" in ulf["Org/model"]["ignore_patterns"]
    assert "ckpt-*/assets/*" in ulf["Org/model"]["ignore_patterns"]
    assert ulf["Org/dataset"]["repo_type"] == "dataset"
    assert "ignore_patterns" not in ulf["Org/dataset"]                     # dataset uploads everything
    assert ("upload_file", "ckpt-60000/assets/norm_stats.json") in api.calls
    assert ("model_info", "Org/model") in api.calls  # end-of-node verification


def test_upload_env_is_restored_after_node(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "keepme")
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=True))
    monkeypatch.setattr(nodes, "_sh", _fake_cp_al)
    H.make_norm(nodes, cfg)
    (nodes.ckpt_dir(cfg) / "59999" / "params").mkdir(parents=True)
    assert nodes.upload_to_hf({"config": cfg})["upload"]["status"] == "succeeded"
    import os
    assert os.environ["HF_HUB_DISABLE_XET"] == "keepme"          # restored, not leaked
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ


def test_upload_stall_clears_cache_and_retries(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    api = H.FakeApi(exists=True, fail_uploads=1)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: api)
    monkeypatch.setattr(nodes, "_sh", _fake_cp_al)
    H.make_norm(nodes, cfg)
    (nodes.ckpt_dir(cfg) / "59999" / "params").mkdir(parents=True)
    nodes.UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    (nodes.UPLOAD_CACHE / "sentinel").write_text("stale resumable-upload state")
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "succeeded"
    assert not nodes.UPLOAD_CACHE.exists()  # spec failure mode 8: cache cleared between attempts
    assert "Org/model" in _ulf_calls(api)


def test_upload_sha_mismatch_fails_stage(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    api = H.FakeApi(exists=True, path_shas={"ckpt-60000/params/params.bin": "0" * 64})
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: api)
    monkeypatch.setattr(nodes, "_sh", _fake_cp_al)
    H.make_norm(nodes, cfg)
    p = nodes.ckpt_dir(cfg) / "59999" / "params"
    p.mkdir(parents=True)
    (p / "params.bin").write_bytes(b"localbytes")
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "failed"
    assert "mismatch" in out["upload"]["error"]


def test_upload_sha_verification_passes_and_covers_dataset(cfg, sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(nodes, "needs_upload", lambda c: True)
    H.make_norm(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    p = nodes.ckpt_dir(cfg) / "59999" / "params"
    p.mkdir(parents=True)
    (p / "params.bin").write_bytes(b"localbytes")
    info_json = nodes.LEROBOT_ROOT / cfg["dataset_name"] / "meta" / "info.json"
    api = H.FakeApi(exists=True, path_shas={
        "ckpt-60000/params/params.bin": hashlib.sha256(b"localbytes").hexdigest(),
        "meta/info.json": hashlib.sha256(info_json.read_bytes()).hexdigest()})
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: api)
    monkeypatch.setattr(nodes, "_sh", _fake_cp_al)
    out = nodes.upload_to_hf({"config": cfg})
    assert out["upload"]["status"] == "succeeded"
    checked = [c for c in api.calls if c[0] == "get_paths_info"]
    assert ("get_paths_info", "Org/model", ("ckpt-60000/params/params.bin",), "model") in checked
    assert ("get_paths_info", "Org/dataset", ("meta/info.json",), "dataset") in checked

"""Smoke tests for every section-5 hf skill — HfApi and child processes fully mocked."""
import os
import types

import pytest

from pipeline import tools
from pipeline.skills import hf_skills as h
from pipeline.tests.fakes import FakeHfApi


def test_lookup_and_propose_names(env):
    out = h.lookup_past_naming_conventions()
    assert "Hoshipu/example-hf_agent" in out["examples"]
    assert out["conventions"]["dataset"].startswith("Hoshipu/b1k_")
    p = h.propose_repo_name("dataset", "perturb recovery3 task0 curated")
    assert p["proposals"][0] == "Hoshipu/b1k_perturb_recovery3_task0_curated"
    m = h.propose_repo_name("model", "perturb rec3")
    assert m["proposals"][0].startswith("Hoshipu/pi05-b1kt0-")


def test_verify_repo_three_states(env, monkeypatch):
    fake = FakeHfApi(files={
        "Hoshipu/full_model": ["ckpt-60000/params/x", "README.md"],
        "Hoshipu/empty_model": [".gitattributes"],
        "Hoshipu/full_ds": ["meta/info.json"],
    })
    monkeypatch.setattr(tools, "hf_api", lambda token=None: fake)
    assert h.verify_repo_doesnt_exist_or_confirm_overwrite("Hoshipu/none", "model")["status"] == "new"
    assert h.verify_repo_doesnt_exist_or_confirm_overwrite(
        "Hoshipu/full_model", "model")["status"] == "exists_conflict"
    assert h.verify_repo_doesnt_exist_or_confirm_overwrite(
        "Hoshipu/empty_model", "model")["status"] == "exists_safe"
    assert h.verify_repo_doesnt_exist_or_confirm_overwrite(
        "Hoshipu/full_ds", "dataset")["status"] == "exists_conflict"


def test_hardlink_stage_rounds_final_step(env, tmp_path):
    ckpt = tmp_path / "ckpt"
    for step in ("30000", "59999"):  # final save on disk at N-1
        (ckpt / step / "params").mkdir(parents=True)
        (ckpt / step / "params" / "w.bin").write_bytes(b"W")
        (ckpt / step / "train_state").mkdir()
    staging = tmp_path / "staging"
    out = h.hardlink_stage_checkpoints(str(ckpt), str(staging), [30000, 50000, 60000])
    assert sorted(os.path.basename(p) for p in out["staged"]) == ["ckpt-30000", "ckpt-60000"]
    assert out["missing_steps"] == [50000]
    src = ckpt / "59999/params/w.bin"
    dst = staging / "ckpt-60000/params/w.bin"
    assert dst.stat().st_ino == src.stat().st_ino  # hardlink, not copy
    # stale staging re-hardlinked fresh
    (staging / "ckpt-60000" / "stale.txt").write_text("x")
    h.hardlink_stage_checkpoints(str(ckpt), str(staging), [60000])
    assert not (staging / "ckpt-60000" / "stale.txt").exists()
    bad = h.hardlink_stage_checkpoints(str(ckpt), str(staging), [90000])
    assert "no step dirs" in bad["error"]


class FakePopen:
    def __init__(self, behavior):
        self.behavior, self.polls, self.pid, self.returncode = behavior, 0, 4321, None

    def poll(self):
        self.polls += 1
        if self.behavior == "hang":
            return None
        self.returncode = 0
        return 0


def test_upload_resilient_stall_kill_clear_retry(env, tmp_path, monkeypatch):
    cache = tmp_path / "upload_cache"
    cache.mkdir()
    (cache / "shard").write_bytes(b"x")
    monkeypatch.setattr(tools, "UPLOAD_CACHE", cache)
    fake_api = FakeHfApi()
    monkeypatch.setattr(tools, "hf_api", lambda token=None: fake_api)
    killed = {}
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.setdefault("pid", pid))
    procs = iter([FakePopen("hang"), FakePopen("ok")])
    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **kw: next(procs))
    folder = tmp_path / "stage"
    folder.mkdir()
    out = h.upload_large_folder_resilient(str(folder), "Hoshipu/m", "model",
                                          ignore_patterns=["ckpt-*/train_state/*"],
                                          stall_minutes=0.0001, poll_s=0.01)
    assert out == {"ok": True, "attempts": 2}
    assert killed["pid"] == 4321                       # stalled child killed by process group
    assert not (cache / "shard").exists()              # resumable cache cleared between attempts
    assert ("create_repo", "Hoshipu/m", "model") in fake_api.calls


def test_upload_resilient_exhausts_and_escalates(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "UPLOAD_CACHE", tmp_path / "nope")
    monkeypatch.setattr(tools, "hf_api", lambda token=None: FakeHfApi())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **kw: FakePopen("hang"))
    out = h.upload_large_folder_resilient(str(tmp_path), "Hoshipu/m", "model",
                                          attempts=2, stall_minutes=0.0001, poll_s=0.01)
    assert not out["ok"] and "escalate" in out["error"]


def test_upload_norm_stats_per_checkpoint(env, tmp_path, monkeypatch):
    fake = FakeHfApi()
    monkeypatch.setattr(tools, "hf_api", lambda token=None: fake)
    for s in ("ckpt-30000", "ckpt-60000"):
        (tmp_path / s).mkdir()
    out = h.upload_norm_stats_per_checkpoint(str(tmp_path), "/ds/norm_stats.json", "Hoshipu/m")
    assert out["uploaded"] == {"ckpt-30000": "ok", "ckpt-60000": "ok"}
    assert ("upload_file", "ckpt-60000/assets/norm_stats.json") in fake.calls
    assert "error" in h.upload_norm_stats_per_checkpoint(str(tmp_path / "x"), "n", "r")


def test_confirm_upload_complete(env, tmp_path, monkeypatch):
    fake = FakeHfApi(files={"Hoshipu/m": ["ckpt-60000/params/w.bin",
                                          "ckpt-60000/assets/norm_stats.json"]})
    monkeypatch.setattr(tools, "hf_api", lambda token=None: fake)
    (tmp_path / "ckpt-60000" / "params").mkdir(parents=True)
    (tmp_path / "ckpt-60000" / "params" / "w.bin").write_bytes(b"W")
    out = h.confirm_upload_complete("Hoshipu/m", "model",
                                    ["ckpt-60000/params/w.bin", "ckpt-60000/assets/norm_stats.json"],
                                    local_root=str(tmp_path))
    assert out["ok"] and out["missing"] == [] and out["sha_errors"] == []
    assert any(c[0] == "get_paths_info" for c in fake.calls)  # SHA spot check exercised
    out = h.confirm_upload_complete("Hoshipu/m", "model", ["ckpt-30000/params/w.bin"])
    assert not out["ok"] and out["missing"] == ["ckpt-30000/params/w.bin"]


def test_sha_helpers(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert h._sha256(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert h._git_blob_sha1(p) == hashlib.sha1(b"blob 5\0hello").hexdigest()

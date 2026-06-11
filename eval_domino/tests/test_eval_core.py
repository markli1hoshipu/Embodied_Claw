"""Deterministic-core tests: GPU leases, config validation, task plan, error classification.
Agent behavior is exercised by the live dry-run, not here."""
import json

import pytest

from eval_domino import gpu
from eval_domino.benchmarks import domino
from eval_domino.skills import benchmark_skills, model_skills


@pytest.fixture
def lease_root(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "LEASE_ROOT", tmp_path / "leases")
    return tmp_path / "leases"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIED_CLAW_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


def _fake_snapshot(monkeypatch, free_ids):
    monkeypatch.setattr(gpu, "snapshot", lambda: [
        {"id": i, "mem_free_mb": 140_000 if i in free_ids else 3_000, "leased_by": None}
        for i in range(8)])


def test_acquire_release_roundtrip(lease_root, monkeypatch):
    _fake_snapshot(monkeypatch, free_ids={2, 5})
    got = gpu.acquire("run_a", 2)
    assert sorted(got) == [2, 5]
    # second run sees nothing leasable
    monkeypatch.setattr(gpu, "free_gpus",
                        lambda min_free_mb=0: [g for g in (2, 5)
                                               if not (lease_root / f"gpu{g}.lease").exists()])
    assert gpu.acquire("run_b", 1) == []
    assert sorted(gpu.release("run_a")) == [2, 5]


def test_stale_lease_reaped(lease_root):
    lease_root.mkdir(parents=True)
    (lease_root / "gpu3.lease").write_text(json.dumps({"run_id": "dead", "pid": 99999999}))
    assert gpu.reap_stale() == [3]
    assert not (lease_root / "gpu3.lease").exists()


def test_save_eval_config_validates(runs):
    (runs / "r1").mkdir()
    bad = model_skills.save_eval_config("r1", json.dumps({"model": {"family": "pi05"}}))
    assert not bad["ok"] and any("train_config_name" in e for e in bad["errors"])
    good = model_skills.save_eval_config("r1", json.dumps({
        "model": {"family": "pi05", "train_config_name": "c", "model_name": "m",
                  "checkpoint_step": 30000},
        "benchmark": {"name": "domino", "tasks": ["adjust_bottle"],
                      "task_config": "demo_clean_dynamic"},
        "resources": {"gpus_requested": 1}}))
    assert good["ok"] and good["n_tasks"] == 1


def test_save_eval_config_rejects_unknown_task(runs):
    (runs / "r2").mkdir()
    r = model_skills.save_eval_config("r2", json.dumps({
        "model": {"family": "pi05", "train_config_name": "c", "model_name": "m"},
        "benchmark": {"name": "domino", "tasks": ["adjust_botle"],
                      "task_config": "demo_clean_dynamic"},
        "resources": {"gpus_requested": 1}}))
    assert not r["ok"] and "adjust_bottle" in str(r["errors"])


def test_task_plan_idempotent(runs):
    (runs / "r3").mkdir()
    (runs / "r3" / "config.json").write_text(json.dumps({
        "model": {"family": "pi05", "train_config_name": "c", "model_name": "m"},
        "benchmark": {"name": "domino", "tasks": ["adjust_bottle", "place_shoe"],
                      "task_config": "demo_clean_dynamic", "seed": 0},
        "resources": {"gpus_requested": 1, "gpu_ids": [0]}}))
    first = benchmark_skills.build_task_plan("r3")
    assert first["n_shards"] == 2 and not first["existing"]
    again = benchmark_skills.build_task_plan("r3")
    assert again["existing"]


def test_classify_log():
    assert domino.classify_log("blah\nCUDA error: illegal instruction\n") == "program_error"
    assert domino.classify_log("episode 4 success! score 0.4") == "ok"


def test_retry_and_skip_shard(runs):
    (runs / "r4").mkdir()
    (runs / "r4" / "config.json").write_text(json.dumps({
        "model": {"family": "pi05", "train_config_name": "c", "model_name": "m"},
        "benchmark": {"name": "domino", "tasks": ["adjust_bottle", "place_shoe"],
                      "task_config": "demo_clean_dynamic", "seed": 0},
        "resources": {"gpus_requested": 1, "gpu_ids": [0]}}))
    benchmark_skills.build_task_plan("r4")
    plan = benchmark_skills._load_plan("r4")
    plan["shards"][1]["status"] = "error_paused"
    plan["paused"] = True
    benchmark_skills._save_plan("r4", plan)
    r = benchmark_skills.retry_shard("r4", "place_shoe-s0")
    assert r["ok"]
    plan = benchmark_skills._load_plan("r4")
    assert plan["shards"][0]["id"] == "place_shoe-s0"      # probe moved to front
    assert plan["shards"][0]["status"] == "pending" and not plan["paused"]
    s = benchmark_skills.skip_shard("r4", "place_shoe-s0", "asset corrupted")
    assert s["ok"]
    assert benchmark_skills._load_plan("r4")["shards"][0]["status"] == "failed"

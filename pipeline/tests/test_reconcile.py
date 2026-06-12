"""Reconcile (enter-the-FSM-anywhere): name derivation, evidence, seeding, the train-node
artifact guard, and the cli `node` single-node graph."""
from __future__ import annotations

import json

import pytest

from pipeline import reconcile, tools
from pipeline.state import STAGES
from pipeline.tests.fakes import MINIMAL_CONFIG


@pytest.fixture
def artifacts(env, monkeypatch):
    """Fabricated openpi+lerobot trees matching MINIMAL_CONFIG's derived names, with
    everything 'done' on disk; tests then knock pieces out."""
    root = env / "fakefs"
    monkeypatch.setattr(tools, "OPENPI", root / "openpi")
    monkeypatch.setattr(tools, "LEROBOT_ROOT", root / "lerobot")
    cfg = {**MINIMAL_CONFIG, "run_id": "tr_rec",
           "dataset_name": "b1k_demo", "train_config_name": "pi05_b1k_demo"}
    src = root / "src_a"
    src.mkdir(parents=True)
    (src / "x.parquet").write_text("x")
    cfg["data_request"] = {**cfg["data_request"],
                           "sources": [{"description": "a", "local_dir": str(src)}]}
    dd = root / "lerobot" / "b1k_demo"
    (dd / "meta").mkdir(parents=True)
    (dd / "meta" / "info.json").write_text(json.dumps({"total_episodes": 836}))
    (dd / "norm_stats.json").write_text("x" * 2048)
    final = root / "openpi" / "checkpoints" / "pi05_b1k_demo" / "exp1" / "59999"
    (final / "params").mkdir(parents=True)
    (final / "_CHECKPOINT_METADATA").write_text("{}")
    return cfg, root


def test_derive_names_conventions():
    assert reconcile.derive_names({"dataset_name": "b1k_x"})["train_config_name"] == "pi05_b1k_x"
    assert reconcile.derive_names({"dataset_name": "rmb_y"})["train_config_name"] == "pi05_b1k_rmb_y"
    d = reconcile.derive_names({"outputs": {"hf_dataset_repo": "Hoshipu/b1k_z_curated"}})
    assert d["dataset_name"] == "b1k_z_curated" and d["train_config_name"] == "pi05_b1k_z_curated"
    assert reconcile.derive_names({"train_config_name": "explicit",
                                   "dataset_name": "d"})["train_config_name"] == "explicit"
    assert reconcile.derive_names({})["dataset_name"] is None


def test_evidence_all_done(artifacts):
    cfg, _ = artifacts
    ev = reconcile.evidence(cfg)
    assert all(ev[s]["done"] for s in reconcile.AUTO_SEEDABLE), ev
    assert "836" in ev["filter_build"]["why"]


def test_evidence_degrades_per_stage(artifacts):
    cfg, root = artifacts
    (root / "openpi" / "checkpoints" / "pi05_b1k_demo" / "exp1" / "59999"
     / "_CHECKPOINT_METADATA").unlink()
    ev = reconcile.evidence(cfg)
    assert not ev["train"]["done"] and ev["norm_stats"]["done"]
    (root / "lerobot" / "b1k_demo" / "norm_stats.json").unlink()
    ev = reconcile.evidence(cfg)
    assert not ev["norm_stats"]["done"] and ev["filter_build"]["done"]
    # episodes=0 means a torn build — not done
    (root / "lerobot" / "b1k_demo" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0}))
    assert not reconcile.evidence(cfg)["filter_build"]["done"]
    # sources without local_dir are never claimed done
    cfg2 = {**cfg, "data_request": {**cfg["data_request"], "sources": [{"description": "a"}]}}
    assert not reconcile.evidence(cfg2)["ingest"]["done"]


def test_seed_auto_mode(artifacts):
    cfg, root = artifacts
    seeds, report, conflicts = reconcile.seed(cfg)
    assert set(seeds) == set(reconcile.AUTO_SEEDABLE) and conflicts == []
    assert all(seeds[s]["status"] == "succeeded" for s in seeds)
    assert all("reconciled from disk" in seeds[s]["artifact_paths"][0] for s in seeds)
    # uploads and intake are never auto-seeded
    assert "upload_dataset" not in seeds and "upload_model" not in seeds and "intake" not in seeds
    # knock out the checkpoint -> train goes live
    (root / "openpi" / "checkpoints" / "pi05_b1k_demo" / "exp1" / "59999"
     / "_CHECKPOINT_METADATA").unlink()
    seeds, report, _ = reconcile.seed(cfg)
    assert "train" not in seeds
    assert ("train", "live", reconcile.evidence(cfg)["train"]["why"]) in report


def test_seed_from_node(artifacts):
    cfg, _ = artifacts
    # --from upload_model: everything before it must seed; upload_dataset has no offline
    # evidence -> conflict without force
    seeds, report, conflicts = reconcile.seed(cfg, from_node="upload_model")
    assert [c[0] for c in conflicts] == ["upload_dataset"]
    seeds, report, conflicts = reconcile.seed(cfg, from_node="upload_model", force=True)
    assert conflicts == [] and set(seeds) == set(STAGES) - {"upload_model"}
    assert "FORCED" in seeds["upload_dataset"]["artifact_paths"][0]
    assert "config.json present" in seeds["intake"]["artifact_paths"][0]
    with pytest.raises(ValueError):
        reconcile.seed(cfg, from_node="nope")


def test_train_node_guard_skips_agent(artifacts, monkeypatch):
    cfg, _ = artifacts
    from pipeline.nodes import train
    monkeypatch.setattr("pipeline.nodes.train.make_agent",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no agent!")))
    out = train.node({"run_id": "tr_rec", "config": cfg, "train": {"status": "pending"},
                      "norm_stats": {"status": "succeeded", "artifact_paths": []}})
    assert out["train"]["status"] == "succeeded"
    assert "59999" in out["train"]["artifact_paths"][0]


def test_train_node_guard_falls_through_without_artifact(env, monkeypatch):
    from pipeline.nodes import train
    monkeypatch.setattr(tools, "OPENPI", env / "nothing_here")
    called = []
    monkeypatch.setattr("pipeline.nodes.train._agent_node",
                        lambda state: called.append(1) or {"train": {"status": "failed"}})
    out = train.node({"run_id": "tr_rec", "config": dict(MINIMAL_CONFIG),
                      "train": {"status": "pending"}})
    assert called == [1] and out["train"]["status"] == "failed"


def test_single_node_graph_runs_only_that_node(artifacts, monkeypatch):
    """cli `node norm_stats`: one-node graph, upstream satisfied by seeds, agent mocked."""
    from pipeline.graph import build_single_node_graph
    from pipeline.state import init_state
    from pipeline.tests import fakes
    from pipeline.tests.conftest import patch_client
    cfg, _ = artifacts
    rd = tools.run_dir("tr_rec")
    rd.mkdir(parents=True, exist_ok=True)
    patch_client(monkeypatch, fakes.FakeAnthropic(
        {"norm_stats": [fakes.done(paths=("norm_stats.json",))]}))
    seeds, _, _ = reconcile.seed(cfg)
    g = build_single_node_graph("norm_stats")
    state = {**init_state("tr_rec", cfg),
             **{k: v for k, v in seeds.items() if k != "norm_stats"}}
    out = g.invoke(state)
    assert out["norm_stats"]["status"] == "succeeded"
    # other stages were never executed (still as seeded/pending), proving single-node scope
    assert out["train"]["artifact_paths"] and "reconciled" in out["train"]["artifact_paths"][0]
    assert out["upload_model"]["status"] == "pending"
    with pytest.raises(ValueError):
        build_single_node_graph("nonsense")
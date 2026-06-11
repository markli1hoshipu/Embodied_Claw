import helpers as H
import pytest
import yaml

import pipeline.graph as G
from pipeline import cli, nodes


def test_load_config_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_id: x\n")
    with pytest.raises(SystemExit):
        cli.load_config(str(p))


def test_load_config_roundtrip(tmp_path, cfg):
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert cli.load_config(str(p))["dataset_name"] == cfg["dataset_name"]


def test_dry_run_table_attach_mode(cfg, sandbox, monkeypatch, capsys):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    H.make_norm(nodes, cfg)
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [4242])
    monkeypatch.setattr(nodes, "last_progress", lambda c: "Progress on: 2.45kit/60.0kit rate:3.4it/s")
    cli.dry_run(cfg)
    out = capsys.readouterr().out
    assert "SKIP (artifacts present: 2/2 sources ready)" in out
    assert "SKIP (info.json present, total_episodes=13 == expected 13)" in out
    assert "SKIP (norm_stats.json present, 2048 bytes)" in out
    assert "ATTACH (already running, pids=[4242], last progress: Progress on: 2.45kit" in out
    assert "PENDING (blocked on train)" in out


def test_dry_run_would_run_stages(cfg, sandbox, monkeypatch, capsys):
    monkeypatch.setattr(nodes, "find_train_pids", lambda n: [])
    cli.dry_run(cfg)
    out = capsys.readouterr().out
    assert "RUN (missing sources: org/official, org/perturb)" in out
    assert "RUN (meta/info.json missing or stale" in out
    assert "RUN (norm_stats.json missing)" in out
    assert "RUN (no live process" in out


def test_dry_run_stays_offline_when_train_complete(cfg, sandbox, monkeypatch, capsys):
    """dry-run promises 'side-effect free AND offline' — it must not issue HF API GETs even
    when train is complete and the upload skip-check would normally query the Hub."""
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    H.make_norm(nodes, cfg)
    H.make_final_ckpt(nodes, cfg)
    monkeypatch.setattr(nodes, "needs_upload",
                        lambda c: (_ for _ in ()).throw(AssertionError("network call in dry-run")))
    cli.dry_run(cfg)
    out = capsys.readouterr().out
    assert "would check HF repo contents" in out


def test_run_then_status(cfg, sandbox, monkeypatch, tmp_path, capsys):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    H.make_built(nodes, cfg, episodes=13)
    H.make_norm(nodes, cfg)
    H.make_final_ckpt(nodes, cfg)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=True))
    runs_dir = tmp_path / "pipeline_runs"
    monkeypatch.setattr(G, "PIPELINE_RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "PIPELINE_RUNS_DIR", runs_dir)
    cli.run(cfg)
    cli.show_status(cfg["run_id"])
    out = capsys.readouterr().out
    assert '"status": "skipped"' in out
    assert cfg["dataset_name"] in out


def test_status_never_executed_run_is_clean_exit_zero(tmp_path, monkeypatch, capsys):
    """'Never executed' is an expected, legitimate condition — exit 0, clear message."""
    monkeypatch.setattr(cli, "PIPELINE_RUNS_DIR", tmp_path / "nope")
    cli.show_status("ghost")  # must not SystemExit
    assert "never been executed" in capsys.readouterr().out


def test_run_refuses_concurrent_same_run_id(cfg, tmp_path, monkeypatch):
    """Single-instance flock: a second CLI on the same run_id must refuse to start (it could
    double-launch the --overwrite train script)."""
    import fcntl
    runs_dir = tmp_path / "pipeline_runs"
    runs_dir.mkdir()
    monkeypatch.setattr(cli, "PIPELINE_RUNS_DIR", runs_dir)
    holder = open(runs_dir / f"{cfg['run_id']}.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(SystemExit, match="refusing to run"):
            cli.run(cfg)
    finally:
        holder.close()

import helpers as H

from pipeline import nodes


def test_needs_ingest(cfg, sandbox):
    assert nodes.needs_ingest(cfg)
    H.make_sources(cfg)
    assert not nodes.needs_ingest(cfg)


def test_expected_episodes_math(cfg, sandbox):
    H.make_sources(cfg)            # 4 official, 3 perturb
    H.make_drop_lists(nodes, cfg)  # 1 drop each
    assert nodes.expected_episodes(cfg) == (4 - 1) + (3 - 1) * 5


def test_expected_episodes_none_when_inputs_missing(cfg, sandbox):
    assert nodes.expected_episodes(cfg) is None


def test_expected_episodes_scoped_to_allow_patterns(cfg, sandbox):
    """Parquets outside this run's allow_patterns (e.g. another task ingested into the SHARED
    official dir) must not drift the expected count — drift arms a destructive rebuild."""
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    other = nodes.Path(cfg["sources"][0]["local_dir"]) / "data" / "task-0001"
    other.mkdir(parents=True)
    (other / "ep0.parquet").write_bytes(b"x")  # cross-task contamination
    cfg["sources"][0]["allow_patterns"] = ["data/task-0000/*"]
    assert nodes.expected_episodes(cfg) == (4 - 1) + (3 - 1) * 5  # unchanged


def test_needs_build(cfg, sandbox):
    H.make_sources(cfg)
    H.make_drop_lists(nodes, cfg)
    assert nodes.needs_build(cfg)            # no info.json
    H.make_built(nodes, cfg, episodes=13)
    assert not nodes.needs_build(cfg)        # matches expected
    H.make_built(nodes, cfg, episodes=12)
    assert nodes.needs_build(cfg)            # stale count


def test_needs_build_warns_when_expected_unverifiable(cfg, sandbox, capsys):
    H.make_built(nodes, cfg, episodes=13)    # no sources/drop lists -> expected is None
    assert not nodes.needs_build(cfg)        # conservative: keep the build...
    assert "cannot verify staleness" in capsys.readouterr().out  # ...but loudly


def test_needs_norm(cfg, sandbox):
    assert nodes.needs_norm(cfg)
    H.make_norm(nodes, cfg)
    assert not nodes.needs_norm(cfg)


def test_train_done_requires_metadata_and_params(cfg, sandbox):
    assert nodes.needs_train(cfg)
    (nodes.final_ckpt(cfg) / "params").mkdir(parents=True)
    assert not nodes.train_done(cfg)         # _CHECKPOINT_METADATA still missing
    H.make_final_ckpt(nodes, cfg)
    assert nodes.train_done(cfg) and not nodes.needs_train(cfg)


def test_needs_upload(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=True))
    assert not nodes.needs_upload(cfg)
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(exists=False))
    assert nodes.needs_upload(cfg)


def test_needs_upload_is_content_aware_not_existence_aware(cfg, sandbox, monkeypatch):
    """create_repo runs before upload, so a repo can EXIST while empty/incomplete; the skip
    check must look at content or an interrupted upload is 'skipped' forever."""
    monkeypatch.setattr(nodes, "_hf_api",
                        lambda token=None: H.FakeApi(exists=True, files={"Org/model": []}))
    assert nodes.needs_upload(cfg)           # model repo exists but holds no ckpt-*/params
    monkeypatch.setattr(nodes, "_hf_api", lambda token=None: H.FakeApi(
        exists=True, files={"Org/model": ["ckpt-60000/params/params"], "Org/dataset": ["README.md"]}))
    assert nodes.needs_upload(cfg)           # dataset repo exists but lacks meta/info.json

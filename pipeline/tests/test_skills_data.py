"""Smoke tests for every section-5 data skill — subprocess + pyarrow + HfApi all mocked."""
import json
import types

import pytest

from pipeline import tools
from pipeline.skills import data_skills as d
from pipeline.tests.fakes import FakeHfApi


# ---- minimal pyarrow stand-ins -------------------------------------------------------------
class FakeArr:
    def __init__(self, vals, type=None):
        self.vals, self.type = list(vals), type

    def to_numpy(self):
        return self.vals


class FakeTable:
    def __init__(self, cols):
        self.cols = {k: (v if isinstance(v, FakeArr) else FakeArr(v)) for k, v in cols.items()}

    column_names = property(lambda self: list(self.cols))
    num_rows = property(lambda self: len(next(iter(self.cols.values())).vals))

    def slice(self, start, length):
        return FakeTable({k: a.vals[start:start + length] for k, a in self.cols.items()})

    def set_column(self, i, name, arr):
        out = dict(self.cols)
        out[name] = arr
        return FakeTable(out)

    def column(self, name):
        return self.cols[name]


def fake_pq(tables: dict, rows: dict, written: dict):
    def _meta(p):
        if str(p) in rows:
            n = rows[str(p)]
        elif str(p) in tables:
            n = tables[str(p)].num_rows
        else:
            n = written[str(p)].num_rows
        return types.SimpleNamespace(num_rows=n)
    return types.SimpleNamespace(read_table=lambda p: tables[str(p)], read_metadata=_meta,
                                 write_table=lambda t, p: written.__setitem__(str(p), t))


def fake_pa():
    return types.SimpleNamespace(array=lambda vals, type=None: FakeArr(vals, type))


# ---- source tree helper --------------------------------------------------------------------
def make_b1k_tree(root, eps=(0,), with_videos=True, vkeys=("observation.images.rgb",)):
    for e in eps:
        p = root / "data/task-0000" / f"episode_{e:08d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PAR1")
        a = root / "annotations/task-0000" / f"episode_{e:08d}.json"
        a.parent.mkdir(parents=True, exist_ok=True)
        a.write_text(json.dumps({"skill_annotation": [
            {"skill_id": [1], "frame_duration": [2, 5]},
            {"skill_id": [2], "frame_duration": [5, 7]},
            {"skill_id": [67], "frame_duration": [7, 9]},
            {"skill_id": [3], "frame_duration": [9, 10]}]}))
        if with_videos:
            for vk in vkeys:
                v = root / f"videos/chunk-000/{vk}" / f"episode_{e:06d}.mp4"
                v.parent.mkdir(parents=True, exist_ok=True)
                v.write_bytes(b"\x00")
    return root


INFO = {"chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.rgb": {"dtype": "video", "shape": [3, 224, 224]},
                     "action": {"dtype": "float32", "shape": [2]}}}


def test_download_hf_snapshot_verifies_counts(env, tmp_path, monkeypatch):
    local = tmp_path / "src"
    repo_files = ["data/task-0000/episode_00000000.parquet",
                  "annotations/task-0000/episode_00000000.json",
                  "videos/chunk-000/observation.images.rgb/episode_000000.mp4"]
    monkeypatch.setattr(tools, "hf_api", lambda token=None: FakeHfApi({"r/x": repo_files}))

    def sh(cmd, env=None, timeout=None):
        make_b1k_tree(local)
        assert env["HF_HUB_DISABLE_XET"] == "1"
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tools, "sh", sh)
    out = d.download_hf_snapshot("r/x", "dataset", None, str(local))
    assert out["files_downloaded"] == 3 and out["total_bytes"] > 0
    assert json.loads((local / ".pipeline_expected.json").read_text())["parquets"] == 1
    # second call: manifest satisfied -> skip
    assert d.download_hf_snapshot("r/x", "dataset", None, str(local))["skipped"] == 3


def test_download_hf_snapshot_short_download_fails(env, tmp_path, monkeypatch):
    local = tmp_path / "src"
    monkeypatch.setattr(tools, "hf_api", lambda token=None: FakeHfApi(
        {"r/x": ["data/a.parquet", "data/b.parquet"]}))
    monkeypatch.setattr(tools, "sh", lambda cmd, env=None, timeout=None: (
        make_b1k_tree(local), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[1])
    out = d.download_hf_snapshot("r/x", "dataset", None, str(local))
    assert "incomplete" in out["error"] and "parquets 1/2" in out["error"]


def test_download_hf_zip(env, tmp_path, monkeypatch):
    local = tmp_path / "pr3"

    def sh(cmd, env=None, timeout=None):
        assert "unzip -q -o" in cmd
        (local / "output_0601.zip").parent.mkdir(parents=True, exist_ok=True)
        (local / "output_0601.zip").write_bytes(b"Z" * 10)
        make_b1k_tree(local / "output_0601", with_videos=False)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(tools, "sh", sh)
    out = d.download_hf_zip("n8wishh/failure_recovery", "output_0601.zip", str(local))
    assert out["top_dir"].endswith("output_0601") and out["zip_size"] == 10
    assert out["extracted_files"] >= 2


def test_verify_dataset_integrity_layouts(env, tmp_path, monkeypatch):
    raw = make_b1k_tree(tmp_path / "raw")
    out = d.verify_dataset_integrity(str(raw), "b1k_raw")
    assert out["ok"] and out["parquet_count"] == 1 and out["annotation_count"] == 1
    assert not d.verify_dataset_integrity(str(tmp_path / "nope"), "b1k_raw")["ok"]
    lr = tmp_path / "lr"
    (lr / "meta").mkdir(parents=True)
    (lr / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))
    (lr / "meta" / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    monkeypatch.setattr(tools, "sh",
                        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert d.verify_dataset_integrity(str(lr), "lerobot")["ok"]
    (lr / "meta" / "info.json").write_text(json.dumps({"total_episodes": 5}))
    bad = d.verify_dataset_integrity(str(lr), "lerobot")
    assert not bad["ok"] and "total_episodes" in bad["issues"][0]


def test_inventory_dataset(env, tmp_path, monkeypatch):
    root = make_b1k_tree(tmp_path / "inv", eps=(0, 3))
    monkeypatch.setattr(d, "_pq", lambda: fake_pq({}, {str(p): 10 for p in root.rglob("*.parquet")}, {}))
    out = d.inventory_dataset(str(root))
    assert out["episodes"] == 2 and out["episode_id_range"] == [0, 3]
    assert out["skills_seen"] == [1, 2, 3, 67]
    assert set(out["sample_lengths"].values()) == {10}


def test_run_pca_parses_cuts_and_droplist(env, tmp_path, monkeypatch):
    outd = tmp_path / "pca"
    outd.mkdir()

    def sh(cmd, env=None, timeout=None):
        assert "--drop_thresh 4.83" in cmd and "--force_drop 280" in cmd
        (outd / "drop_list.json").write_text(json.dumps(
            {"drop": [102, 103, 137], "n_drop": 3, "drop_thresh": 4.83}))
        return types.SimpleNamespace(returncode=0, stderr="", stdout=(
            "  p95       d_comb> 4.42 -> drop   7\n  p98       d_comb> 4.83 -> drop   3\n"
            "  mean+2sd  d_comb> 4.49 -> drop   5\n"))
    monkeypatch.setattr(tools, "sh", sh)
    out = d.run_pca("/data", str(outd), drop_thresh=4.83, force_drop=[280])
    assert out["drop_list"] == [102, 103, 137] and out["n_drop"] == 3
    assert out["cuts"]["p98"] == 4.83 and out["cuts"]["mean+2sd"] == 4.49


def test_apply_drop_list():
    assert d.apply_drop_list([5, 1, 3, 2], [3, 9]) == [1, 2, 5]


def test_trim_by_skills_and_empty_guard(env, tmp_path, monkeypatch):
    root = make_b1k_tree(tmp_path / "t")
    pqp = str(root / "data/task-0000/episode_00000000.parquet")
    ann = str(root / "annotations/task-0000/episode_00000000.json")
    monkeypatch.setattr(d, "_pq", lambda: fake_pq({}, {pqp: 10}, {}))
    assert d.trim_by_skills(pqp, ann, [1, 2, 67]) == (2, 9)  # place-on tail dropped
    with pytest.raises(ValueError, match="no segments"):
        d.trim_by_skills(pqp, ann, [99])


def test_extract_keyframes_windows(env, tmp_path, monkeypatch):
    kf = tmp_path / "keyframes_task0.json"
    kf.write_text(json.dumps({"episodes": [{"episode": 0, "t_grasp": 130, "t_press": 100}]}))
    monkeypatch.setattr(d, "_pq", lambda: fake_pq({}, {"episode_00000000.parquet": 200}, {}))
    wins = d.extract_keyframes("episode_00000000.parquet", str(kf), "grasp_press")
    assert wins == [(10, 160), (0, 100)]  # grasp [t-120,t+30), press [t-150,t)


def test_cap_at_video_length(env, monkeypatch):
    monkeypatch.setattr(d.subprocess, "check_output", lambda *a, **kw: b"80\n")
    assert d.cap_at_video_length(100, "v.mp4") == {"effective_rows": 80, "video_frames": 80}
    monkeypatch.setattr(d.subprocess, "check_output", lambda *a, **kw: b"N/A\n")
    assert d.cap_at_video_length(100, "v.mp4")["effective_rows"] == 100  # unparseable -> uncapped


def test_slice_and_renumber(env, tmp_path, monkeypatch):
    src = FakeTable({"episode_index": [9] * 10, "index": list(range(100, 110)),
                     "task_index": [4] * 10, "action": list(range(10))})
    written = {}
    monkeypatch.setattr(d, "_pq", lambda: fake_pq({"src.parquet": src}, {}, written))
    monkeypatch.setattr(d, "_pa", fake_pa)
    out = d.slice_and_renumber("src.parquet", 2, 9, new_idx=5, cum_index=50,
                               dst_path=str(tmp_path / "dst.parquet"))
    assert out.num_rows == 7
    assert out.column("episode_index").vals == [5] * 7
    assert out.column("index").vals == list(range(50, 57))
    assert out.column("task_index").vals == [0] * 7
    assert out.column("action").vals == [2, 3, 4, 5, 6, 7, 8]
    assert str(tmp_path / "dst.parquet") in written


def test_slice_and_renumber_tool_wrapper_json_roundtrips(env, tmp_path, monkeypatch):
    """The agent-facing wrapper writes dst_path and returns JSON (a Table can't round-trip a
    tool_result); it must be registered for the filter_build node per spec section 4/5."""
    src = FakeTable({"episode_index": [9] * 10, "index": list(range(10)), "action": list(range(10))})
    written = {}
    monkeypatch.setattr(d, "_pq", lambda: fake_pq({"src.parquet": src}, {}, written))
    monkeypatch.setattr(d, "_pa", fake_pa)
    out = d.slice_and_renumber_to_file("src.parquet", 2, 9, 5, 50,
                                       dst_path=str(tmp_path / "dst.parquet"))
    assert out == {"rows": 7, "dst_path": str(tmp_path / "dst.parquet")}
    assert json.dumps(out)  # JSON-able tool_result
    from pipeline.agents.data_agent import REGISTRY
    from pipeline.nodes.filter_build import SKILLS
    assert REGISTRY["slice_and_renumber"] is d.slice_and_renumber_to_file
    assert "slice_and_renumber" in SKILLS


def test_symlink_videos_realpath(env, tmp_path):
    import os
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    (tmp_path / "rgb").mkdir()
    (tmp_path / "rgb" / "chain.mp4").symlink_to(real)  # chains must resolve to real bytes
    made = d.symlink_videos(str(tmp_path / "{video_key}" / "chain.mp4"),
                            str(tmp_path / "out" / "{video_key}.mp4"), video_keys=["rgb"])
    assert made == [str(tmp_path / "out" / "rgb.mp4")]
    assert os.readlink(made[0]) == str(real)
    d.symlink_videos(str(tmp_path / "{video_key}" / "chain.mp4"),
                     str(tmp_path / "out" / "{video_key}.mp4"), video_keys=["rgb"])  # re-entrant


def test_emit_lerobot_meta(env, tmp_path):
    tpl = tmp_path / "info.json"
    tpl.write_text(json.dumps(INFO))
    eps = [{"episode_index": 0, "tasks": ["t"], "length": 7, "stats": {"action": {"count": [7]}}},
           {"episode_index": 1, "tasks": ["t"], "length": 8}]
    out = d.emit_lerobot_meta(str(tmp_path / "ds"), eps, ["t"], str(tpl))
    assert out == {"meta_dir": str(tmp_path / "ds" / "meta"), "total_episodes": 2,
                   "total_frames": 15}
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_videos"] == 2
    assert info["splits"] == {"train": "0:2"}
    lines = (tmp_path / "ds/meta/episodes.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"episode_index": 0, "tasks": ["t"], "length": 7}
    assert json.loads((tmp_path / "ds/meta/tasks.jsonl").read_text()) == \
        {"task_index": 0, "task": "t"}


def test_build_curated_dataset(env, tmp_path, monkeypatch):
    gate = {}
    monkeypatch.setattr(tools, "live_pids", lambda c: (gate.setdefault("cfg", c), [])[1])
    official = make_b1k_tree(tmp_path / "official", eps=(0, 1))
    perturb = make_b1k_tree(tmp_path / "perturb", eps=(0,))
    tpl = tmp_path / "info.json"
    tpl.write_text(json.dumps(INFO))
    drops = tmp_path / "off_drop.json"
    drops.write_text(json.dumps({"drop": [1]}))
    tables = {str(p): FakeTable({"episode_index": [0] * 10, "index": list(range(10)),
                                 "task_index": [1] * 10, "action": list(range(10))})
              for base in (official, perturb) for p in base.rglob("*.parquet")}
    monkeypatch.setattr(d, "_pq", lambda: fake_pq(tables, {}, {}))
    monkeypatch.setattr(d, "_pa", fake_pa)
    monkeypatch.setattr(d, "_episode_stats", lambda sliced, info: {"action": {"count": [sliced.num_rows]}})
    monkeypatch.setattr(d, "video_nframes", lambda p: 8)  # perturb videos shorter than parquet
    out = d.build_curated_dataset(str(official), str(perturb), str(tmp_path / "ds"), "Turn on…",
                                  str(tpl), official_drop_list=str(drops), dup_factor=3)
    # official ep0 trimmed [2,9)=7 rows (ep1 PCA-dropped); perturb ep0 capped at 8, dup x3
    assert out["total_episodes"] == 4 and out["total_frames"] == 7 + 3 * 8
    assert out["official_kept"] == 1 and out["perturb_kept"] == 1 and out["truncated"] == 1
    assert gate["cfg"] == "pi05_b1k_ds"  # liveness gate armed with the derived config name
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["total_episodes"] == 4
    assert not list((tmp_path / "ds").glob("**/*.tmp"))
    # re-entry guard: existing build is not clobbered without overwrite=True
    again = d.build_curated_dataset(str(official), str(perturb), str(tmp_path / "ds"), "p", str(tpl))
    assert again == {"skipped": True, "out_dir": str(tmp_path / "ds"), "total_episodes": 4}


def test_build_refuses_while_training_live(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "live_pids", lambda c: [200738])
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    (tmp_path / "ds" / "meta" / "info.json").write_text("{}")
    out = d.build_curated_dataset("o", "p", str(tmp_path / "ds"), "x", "tpl",
                                  train_config_name="cfg", overwrite=True)
    assert "refusing destructive rebuild" in out["error"]
    assert (tmp_path / "ds" / "meta" / "info.json").exists()  # nothing was rmtree'd


def test_build_liveness_gate_is_unconditional(env, tmp_path, monkeypatch):
    """Regression for the opt-in-gate blocker: the two bypasses (overwrite without
    train_config_name; partial dir without meta/info.json) must BOTH hit the liveness gate,
    with the config name derived from the out_dir basename (pi05_b1k_<dataset_name>)."""
    seen = {}

    def live(c):
        seen["cfg"] = c
        return [200738]
    monkeypatch.setattr(tools, "live_pids", live)
    # (a) complete dataset + overwrite=True + NO train_config_name -> refuse via derived name
    ds = tmp_path / "perturb_recovery3_task0_curated"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    out = d.build_curated_dataset("o", "p", str(ds), "x", "tpl", overwrite=True)
    assert "refusing destructive rebuild" in out["error"]
    assert seen["cfg"] == "pi05_b1k_perturb_recovery3_task0_curated"
    assert (ds / "meta" / "info.json").exists()
    # (b) partial build dir (no meta/info.json) + NO overwrite + NO train_config_name -> refuse
    part = tmp_path / "partial_ds"
    (part / "data").mkdir(parents=True)
    (part / "data" / "leftover.parquet").write_bytes(b"PAR1")
    out = d.build_curated_dataset("o", "p", str(part), "x", "tpl")
    assert "refusing destructive rebuild" in out["error"]
    assert (part / "data" / "leftover.parquet").exists()  # partial dir not rmtree'd either


def test_parse_user_request_and_write_run_config(run_dir):
    out = d.parse_user_request("tr1")
    assert "PCA p98" in out["request"] and "train_request" in out["expected_schema"]
    bad = d.write_run_config("tr1", {"run_id": "tr1"})
    assert "missing" in bad["error"] and "train_request" in bad["error"]
    from pipeline.tests.fakes import MINIMAL_CONFIG
    ok = d.write_run_config("tr1", {**MINIMAL_CONFIG, "run_id": "tr1"},
                            confidence={"train_request.num_train_steps": "explicit",
                                        "outputs.hf_model_repo": "confirmed"})
    written = json.loads((run_dir / "config.json").read_text())
    assert ok["ok"] and written["run_id"] == "tr1"
    # per-field confidence audit trail persisted (spec section 1 step 3)
    assert written["_confidence"]["train_request.num_train_steps"] == "explicit"

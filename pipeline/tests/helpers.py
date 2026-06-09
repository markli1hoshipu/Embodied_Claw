"""Shared fixtures helpers: build fake artifact trees + fake HfApi. No real subprocess/network."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

# With defaults: expected = (4 - 1) + (3 - 1) * 5 = 13 episodes.


def CP(rc, out="", err=""):
    return subprocess.CompletedProcess(["bash"], rc, out, err)


def make_official(cfg, n=4):
    off = Path(cfg["sources"][0]["local_dir"]) / "data" / "task-0000"
    off.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (off / f"ep{i}.parquet").write_bytes(b"x")


def make_perturb(cfg, n=3):
    per = Path(cfg["sources"][1]["local_dir"]) / "out" / "data"
    per.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (per / f"ep{i}.parquet").write_bytes(b"x")


def make_sources(cfg, n_off=4, n_per=3):
    make_official(cfg, n_off)
    make_perturb(cfg, n_per)


def official_repo_files(n=4):
    """Repo-side listing matching make_official(), for ingest count verification."""
    return [f"data/task-0000/ep{i}.parquet" for i in range(n)]


def make_drop_lists(nodes, cfg, off_drop=1, per_drop=1):
    for sub, n in (("merged", off_drop), (cfg["run_id"], per_drop)):
        d = nodes.PCA_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "drop_list.json").write_text(json.dumps({"drop": list(range(n))}))


def make_built(nodes, cfg, episodes=13):
    meta = nodes.LEROBOT_ROOT / cfg["dataset_name"] / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(
        {"total_episodes": episodes, "total_frames": 999, "total_videos": episodes * 3}))


def make_norm(nodes, cfg, size=2048):
    p = nodes.norm_stats_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"n" * size)


def make_final_ckpt(nodes, cfg):
    f = nodes.final_ckpt(cfg)
    (f / "params").mkdir(parents=True, exist_ok=True)
    (f / "_CHECKPOINT_METADATA").write_text("{}")


class FakeApi:
    def __init__(self, exists=False, fail_uploads=0, files=None, path_shas=None):
        self.calls, self.exists, self.fail_uploads = [], exists, fail_uploads
        self.files = files or {}          # repo_id -> explicit file listing
        self.path_shas = path_shas or {}  # path_in_repo -> lfs sha256 for get_paths_info

    def model_info(self, repo):
        self.calls.append(("model_info", repo))
        if not self.exists:
            raise RuntimeError("404 repo not found")

    def dataset_info(self, repo):
        self.calls.append(("dataset_info", repo))
        if not self.exists:
            raise RuntimeError("404 repo not found")

    def list_repo_files(self, repo, repo_type=None):
        self.calls.append(("list_repo_files", repo, repo_type))
        if repo in self.files:
            return self.files[repo]
        if not self.exists:
            raise RuntimeError("404 repo not found")
        # default: content-complete listings so FakeApi(exists=True) reads as upload-done
        return ["meta/info.json"] if repo_type == "dataset" else ["ckpt-60000/params/params"]

    def create_repo(self, repo, repo_type=None, exist_ok=False):
        self.calls.append(("create_repo", repo, repo_type))

    def upload_large_folder(self, **kw):
        if self.fail_uploads > 0:
            self.fail_uploads -= 1
            raise RuntimeError("simulated final-shard stall")
        self.calls.append(("upload_large_folder", kw))  # full kwargs: ignore_patterns must be visible

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw["path_in_repo"]))

    def get_paths_info(self, repo, paths, repo_type=None):
        self.calls.append(("get_paths_info", repo, tuple(paths), repo_type))
        return [SimpleNamespace(path=p,
                                lfs=SimpleNamespace(sha256=self.path_shas[p]) if p in self.path_shas else None,
                                blob_id=None)
                for p in paths]

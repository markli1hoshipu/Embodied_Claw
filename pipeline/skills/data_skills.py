"""data_agent skills (spec section 5). Hardened logic ported from legacy/pipeline_v1/nodes.py +
build_perturb_recovery3_task0_curated_lerobot.py: 429 backoff, per-file snapshot fallback,
expected-counts manifest, ffprobe capping, realpath symlinks, destructive-rebuild gates."""
import fnmatch
import glob
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import time

from pipeline import tools

GRASP_BEFORE, GRASP_AFTER, PRESS_WINDOW = 120, 30, 150  # frames @30fps (keyframes build)
BACKOFFS = (5, 10, 20, 40)  # exponential backoff on HF 429 (spec section 9 row 1)


def _pq():
    import pyarrow.parquet as pq  # lazy: not in the driver venv; heavy builds need it installed
    return pq


def _pa():
    import pyarrow as pa
    return pa


def _np():
    import numpy as np
    return np


def _count_kinds(root: pathlib.Path) -> dict:
    ann = root / "annotations"
    return {"parquets": sum(1 for _ in root.rglob("*.parquet")),
            "mp4s": sum(1 for _ in root.rglob("*.mp4")),
            "annotations": sum(1 for p in ann.rglob("*") if p.is_file()) if ann.is_dir() else 0}


def _manifest(root: pathlib.Path) -> pathlib.Path:
    return root / ".pipeline_expected.json"


def _ready(root: pathlib.Path) -> bool:
    if not ((root / "data").is_dir() and any((root / "data").rglob("*.parquet"))):
        return False
    try:
        expected = json.loads(_manifest(root).read_text())
    except (OSError, json.JSONDecodeError):
        return True  # pre-pipeline manual downloads carry no manifest -> weak check
    have = _count_kinds(root)
    return all(have.get(k, 0) >= v for k, v in expected.items())


def _hf_sh(cmd: str, timeout: int = 6 * 3600):
    """Run an HF download command under conda xvla-stable with Xet/hf_transfer off, retrying
    429s with exponential backoff (spec section 9). Returns (rc, tail)."""
    env = {**tools.HF_ENV, "HF_TOKEN": tools.require_hf_token("ingest")}
    for i, pause in enumerate((*BACKOFFS, None)):
        r = tools.sh(f"{tools.CONDA} && {cmd}", env=env, timeout=timeout)
        tail = ((r.stderr or "") + (r.stdout or ""))[-2000:]
        if r.returncode == 0 or "429" not in tail or pause is None:
            return r.returncode, tail
        time.sleep(pause)
    return 1, "unreachable"


def download_hf_snapshot(repo_id: str, repo_type: str = "dataset", allow_patterns: list = None,
                         local_dir: str = "", max_workers: int = 8) -> dict:
    """Download an HF snapshot to local_dir, verify local counts cover the repo listing
    (filtered by allow_patterns), and persist an expected-counts manifest. 429s are retried
    with backoff; snapshot failures fall back to per-file downloads."""
    root = pathlib.Path(local_dir)
    pre = _count_kinds(root) if root.exists() else {"parquets": 0, "mp4s": 0, "annotations": 0}
    if _ready(root):
        return {"files_downloaded": 0, "total_bytes": 0, "skipped": sum(pre.values())}
    py = (f"from huggingface_hub import snapshot_download; snapshot_download(repo_id={repo_id!r}, "
          f"repo_type={repo_type!r}, allow_patterns={allow_patterns!r}, local_dir={local_dir!r}, "
          f"max_workers={max_workers})")
    rc, tail = _hf_sh(f"python -c {shlex.quote(py)}")
    if rc != 0:
        if "429" in tail:
            return {"error": f"HF 429 rate limit on {repo_id} after backoff exhausted: {tail[-300:]}"}
        fb = ("import fnmatch\nfrom huggingface_hub import HfApi, hf_hub_download\n"
              f"files = HfApi().list_repo_files({repo_id!r}, repo_type={repo_type!r})\n"
              f"pats = {allow_patterns!r}\n"
              "for f in files:\n"
              "    if pats and not any(fnmatch.fnmatch(f, p) for p in pats): continue\n"
              f"    hf_hub_download(repo_id={repo_id!r}, repo_type={repo_type!r}, filename=f, "
              f"local_dir={local_dir!r})\n")
        rc, tail = _hf_sh(f"python -c {shlex.quote(fb)}")
        if rc != 0:
            return {"error": f"{repo_id}: snapshot + per-file fallback failed: {tail[-800:]}"}
    have = _count_kinds(root)
    try:
        files = tools.hf_api().list_repo_files(repo_id, repo_type=repo_type)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{repo_id}: cannot list repo files to verify: {str(e)[-500:]}"}
    kept = [f for f in files if not allow_patterns
            or any(fnmatch.fnmatch(f, p) for p in allow_patterns)]
    expected = {"parquets": sum(f.endswith(".parquet") for f in kept),
                "mp4s": sum(f.endswith(".mp4") for f in kept),
                "annotations": sum(f.startswith("annotations/") for f in kept)}
    short = {k: (have.get(k, 0), v) for k, v in expected.items() if have.get(k, 0) < v}
    if short:
        return {"error": f"{repo_id}: download incomplete under {root} — "
                         + ", ".join(f"{k} {h}/{v}" for k, (h, v) in short.items())}
    _manifest(root).write_text(json.dumps(expected))
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return {"files_downloaded": sum(have.values()) - sum(pre.values()),
            "total_bytes": total, "skipped": sum(pre.values())}


def download_hf_zip(repo_id: str, filename: str, local_dir: str, unzip: bool = True) -> dict:
    """hf_hub_download one zip + `unzip -q -o`. The zip stem becomes top_dir."""
    root = pathlib.Path(local_dir)
    top = root / pathlib.Path(filename).stem
    if _ready(top):
        return {"zip_size": 0, "extracted_files": sum(_count_kinds(top).values()),
                "top_dir": str(top), "skipped": True}
    py = (f"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id={repo_id!r}, "
          f"repo_type='dataset', filename={filename!r}, local_dir={local_dir!r})")
    cmd = f"python -c {shlex.quote(py)}"
    if unzip:
        cmd += f" && cd {shlex.quote(local_dir)} && unzip -q -o {shlex.quote(filename)}"
    rc, tail = _hf_sh(cmd)
    if rc != 0:
        return {"error": f"{repo_id}/{filename}: {tail[-800:]}"}
    have = _count_kinds(top)
    if have["parquets"] == 0:
        return {"error": f"{repo_id}: unzip finished but no parquets under {top}"}
    _manifest(top).write_text(json.dumps(have))
    zp = root / filename
    return {"zip_size": zp.stat().st_size if zp.exists() else 0,
            "extracted_files": sum(have.values()), "top_dir": str(top)}


def verify_dataset_integrity(root: str, expected_layout: str = "b1k_raw") -> dict:
    """Check a tree against its expected layout: b1k_raw | perturb_zip | lerobot."""
    r = pathlib.Path(root)
    counts, issues = _count_kinds(r), []
    if not r.is_dir():
        issues.append(f"{root} does not exist")
    elif expected_layout == "lerobot":
        try:
            info = json.loads((r / "meta" / "info.json").read_text())
            n_eps = sum(1 for _ in open(r / "meta" / "episodes.jsonl"))
            if info["total_episodes"] != n_eps:
                issues.append(f"info total_episodes={info['total_episodes']} != "
                              f"episodes.jsonl lines={n_eps}")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            issues.append(f"meta unreadable: {e}")
        broken = tools.sh(f"find -L {shlex.quote(root)} -name '*.mp4' -type l").stdout.strip()
        if broken:
            issues.append(f"broken video symlinks:\n{broken[:1000]}")
    else:
        if counts["parquets"] == 0:
            issues.append("no parquets under data/")
        if expected_layout == "b1k_raw":
            anns = sorted((r / "annotations").rglob("*.json")) if (r / "annotations").is_dir() else []
            if not anns:
                issues.append("no annotation JSONs")
            else:
                try:
                    json.loads(anns[0].read_text())["skill_annotation"]
                except (json.JSONDecodeError, KeyError) as e:
                    issues.append(f"annotation schema unexpected ({anns[0].name}): {e}")
    return {"ok": not issues, "parquet_count": counts["parquets"], "video_count": counts["mp4s"],
            "annotation_count": counts["annotations"], "issues": issues}


def inventory_dataset(root: str) -> dict:
    """Summary table of one dataset tree: episode ids, file counts, skills seen, sample lengths."""
    r = pathlib.Path(root)
    counts = _count_kinds(r)
    eps = sorted(int(m.group(1)) for p in r.rglob("episode_*.parquet")
                 if (m := re.search(r"episode_(\d+)\.parquet", p.name)))
    skills, lengths = set(), {}
    for ap in sorted(r.rglob("annotations/**/episode_*.json"))[:20]:
        try:
            for seg in json.loads(ap.read_text()).get("skill_annotation", []):
                skills.add(int(seg["skill_id"][0]))
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            pass
    try:
        pq = _pq()
        for p in list(r.rglob("episode_*.parquet"))[:5]:
            lengths[p.name] = pq.read_metadata(str(p)).num_rows
    except ImportError:
        lengths = {"unavailable": "pyarrow not installed in this venv"}
    return {"root": str(r), "episodes": len(eps),
            "episode_id_range": [eps[0], eps[-1]] if eps else None, **counts,
            "skills_seen": sorted(skills), "sample_lengths": lengths}


def run_pca(data_dir: str, output_dir: str, drop_thresh: float = None,
            force_drop: list = None, script: str = None) -> dict:
    """Run the PCA outlier filter (action+proprio d_combined). Two-pass workflow: call once
    without drop_thresh to get the percentile cuts, then again with the chosen threshold to
    write drop_list.json. Returns the drop list + cuts table."""
    script = script or str(tools.PCA_DIR / "run_pca.py")
    args = f"--data_root {shlex.quote(data_dir)} --out_dir {shlex.quote(output_dir)}"
    if drop_thresh is not None:
        args += f" --drop_thresh {drop_thresh}"
    if force_drop:
        args += " --force_drop " + " ".join(str(int(x)) for x in force_drop)
    r = tools.sh(f"{tools.CONDA} && python {shlex.quote(script)} {args}", timeout=3600)
    if r.returncode != 0:
        return {"error": ((r.stderr or "") + (r.stdout or ""))[-1500:]}
    cuts = {m.group(1): float(m.group(2)) for m in
            re.finditer(r"(p\d+|mean\+\d+sd)\s*[=>]*\s*d?_?comb?[>=\s]*([\d.]+)", r.stdout)}
    out = {"cuts": cuts, "drop_list": [], "n_drop": 0, "stdout_tail": r.stdout[-1200:]}
    dl = pathlib.Path(output_dir) / "drop_list.json"
    if dl.exists():
        d = json.loads(dl.read_text())
        out.update({"drop_list": d.get("drop", []), "n_drop": d.get("n_drop", len(d.get("drop", []))),
                    "drop_list_path": str(dl)})
    return out


def apply_drop_list(source_eps: list, drop_list: list) -> list:
    """Filtered episode-id list: source minus drops."""
    return [e for e in sorted(source_eps) if e not in set(drop_list)]


def trim_by_skills(parquet_path: str, annotation_path: str, skills_keep: list = (1, 2, 67),
                   skills_drop: list = ()) -> tuple:
    """(start, end) row range covering the kept skill segments (b1k annotation schema:
    segments sorted by frame_duration[0]; skill ids 1=move-to 2=pick-up 67=press, 3=place-on)."""
    n_rows = _pq().read_metadata(parquet_path).num_rows
    ann = json.loads(pathlib.Path(annotation_path).read_text())
    segs = sorted(ann.get("skill_annotation", []), key=lambda s: s["frame_duration"][0])
    keep, drop = set(skills_keep), set(skills_drop)
    kept = [s for s in segs if s["skill_id"][0] in keep and s["skill_id"][0] not in drop]
    if not kept:
        raise ValueError(f"no segments with skill ids {sorted(keep)} in {annotation_path} "
                         f"(present: {sorted({s['skill_id'][0] for s in segs})})")
    return max(0, int(kept[0]["frame_duration"][0])), min(int(kept[-1]["frame_duration"][1]), n_rows)


def extract_keyframes(parquet_path: str, annotation_path: str, strategy: str = "grasp_press") -> list:
    """Keyframe windows from a keyframes_task0.json-style entry ({"t_grasp": t, "t_press": t}):
    grasp=[t-120,t+30), press=[t-150,t). Windows shorter than 2 rows are skipped."""
    n_rows = _pq().read_metadata(parquet_path).num_rows
    meta = json.loads(pathlib.Path(annotation_path).read_text())
    if "episodes" in meta:  # whole keyframes file: pick the entry for this episode id
        ep = int(re.search(r"episode_(\d+)", pathlib.Path(parquet_path).name).group(1))
        meta = next(e for e in meta["episodes"] if e["episode"] == ep)
    wins = []
    if "grasp" in strategy and "t_grasp" in meta:
        wins.append((max(0, meta["t_grasp"] - GRASP_BEFORE), min(n_rows, meta["t_grasp"] + GRASP_AFTER)))
    if "press" in strategy and "t_press" in meta:
        wins.append((max(0, meta["t_press"] - PRESS_WINDOW), min(n_rows, meta["t_press"])))
    return [(s, e) for s, e in wins if e - s >= 2]


def video_nframes(path: str) -> int:
    out = subprocess.check_output(
        [tools.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]).decode().strip()
    return int(out) if out.isdigit() else 0


def cap_at_video_length(parquet_R: int, video_path: str) -> dict:
    """min(parquet rows, ffprobe nb_frames) — spec section 9 row 3 (source video < parquet rows).
    video_path may contain '{video_key}'-expanded siblings; pass each and take the min upstream,
    or pass one path here. nb_frames==0 (unparseable) leaves R uncapped."""
    v = video_nframes(video_path)
    return {"effective_rows": min(int(parquet_R), v) if v > 0 else int(parquet_R),
            "video_frames": v}


def slice_and_renumber(parquet: str, start: int, end: int, new_idx: int, cum_index: int,
                       dst_path: str = None):
    """Slice [start,end) rows, renumber episode_index/new contiguous index/task_index=0,
    preserving original arrow types. Returns the sliced pyarrow Table (writes it if dst_path)."""
    pq, pa = _pq(), _pa()
    length = end - start
    sliced = pq.read_table(parquet).slice(start, length)
    cols = sliced.column_names
    sliced = sliced.set_column(cols.index("episode_index"), "episode_index",
                               pa.array([new_idx] * length, type=sliced.column("episode_index").type))
    if "index" in cols:
        sliced = sliced.set_column(cols.index("index"), "index",
                                   pa.array(list(range(cum_index, cum_index + length)),
                                            type=sliced.column("index").type))
    if "task_index" in cols:
        sliced = sliced.set_column(cols.index("task_index"), "task_index",
                                   pa.array([0] * length, type=sliced.column("task_index").type))
    if dst_path:
        p = pathlib.Path(dst_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sliced, p)
        assert pq.read_metadata(p).num_rows == length
    return sliced


def slice_and_renumber_to_file(parquet: str, start: int, end: int, new_idx: int,
                               cum_index: int, dst_path: str) -> dict:
    """Tool-facing slice_and_renumber: same slicing/renumbering, but writes to dst_path and
    returns a JSON-able summary (the Table-returning variant cannot round-trip a tool_result)."""
    sliced = slice_and_renumber(parquet, start, end, new_idx, cum_index, dst_path=dst_path)
    return {"rows": sliced.num_rows, "dst_path": dst_path}


def symlink_videos(src_video: str, dst_video: str, video_keys: list = None) -> list:
    """Symlink dst -> realpath(src) per video key ('{video_key}' placeholder in both paths).
    realpath resolution is load-bearing: uploads must resolve real bytes, never chains."""
    made = []
    for vk in (video_keys or [None]):
        s = src_video.format(video_key=vk) if vk else src_video
        d = pathlib.Path(dst_video.format(video_key=vk) if vk else dst_video)
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.is_symlink() or d.exists():
            d.unlink()
        d.symlink_to(pathlib.Path(os.path.realpath(s)))
        made.append(str(d))
    return made


def emit_lerobot_meta(out_dir: str, episodes: list, tasks: list, info_template: str) -> dict:
    """Write meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}. episodes entries:
    {"episode_index","tasks","length"[,"stats"]}; info_template = a reference build's info.json."""
    out = pathlib.Path(out_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    info = json.loads(pathlib.Path(info_template).read_text())
    video_keys = [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")]
    n, frames = len(episodes), sum(e["length"] for e in episodes)
    info.update({"total_episodes": n, "total_frames": frames, "total_tasks": len(tasks),
                 "total_videos": len(video_keys) * n, "splits": {"train": f"0:{n}"}})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({"episode_index": e["episode_index"], "tasks": e["tasks"],
                                "length": e["length"]}) + "\n")
    with open(out / "meta" / "episodes_stats.jsonl", "w") as f:
        for e in episodes:
            if "stats" in e:
                f.write(json.dumps({"episode_index": e["episode_index"], "stats": e["stats"]}) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    return {"meta_dir": str(out / "meta"), "total_episodes": n, "total_frames": frames}


def _img_stats(n: int, shape: list) -> dict:
    c = shape[-1] if shape[-1] in (1, 3) else 3
    return {"min": [[[0.0]]] * c, "max": [[[255.0]]] * c, "mean": [[[128.0]]] * c,
            "std": [[[64.0]]] * c, "count": [n]}


def _scalar_stats(arr) -> dict:
    np = _np()
    if arr.ndim == 1:
        return {"min": [float(arr.min())], "max": [float(arr.max())], "mean": [float(arr.mean())],
                "std": [float(arr.std())], "count": [int(arr.shape[0])]}
    return {"min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).astype(np.float32).tolist(),
            "std": arr.std(axis=0).astype(np.float32).tolist(), "count": [int(arr.shape[0])]}


def _episode_stats(sliced, info: dict) -> dict:
    np, stats, scols = _np(), {}, set(sliced.column_names)
    for key, ft in info["features"].items():
        if ft["dtype"] in ("video", "image"):
            stats[key] = _img_stats(sliced.num_rows, ft["shape"])
        elif key in scols:
            arr = np.asarray(sliced.column(key).to_numpy())
            if arr.dtype == object:
                if not (len(arr) and isinstance(arr[0], (list, np.ndarray))):
                    continue
                arr = np.stack(list(arr), axis=0).astype(np.float32)
            else:
                arr = arr.astype(np.float32)
            stats[key] = _scalar_stats(arr)
    return stats


def build_curated_dataset(official_dir: str, perturb_dir: str, out_dir: str, prompt: str,
                          info_template: str, official_drop_list: str = None,
                          perturb_drop_list: str = None, dup_factor: int = 5,
                          skills_keep: list = (1, 2, 67), train_config_name: str = None,
                          overwrite: bool = False) -> dict:
    """Composite build (ports build_perturb_recovery3_task0_curated_lerobot.py): official
    episodes skill-trimmed + perturb clips video-capped and duplicated dup_factor x, contiguous
    reindex, task_index=0, single prompt. Safety gates from v1: UNCONDITIONALLY refuses to run
    while a matching train process is live (train_config_name defaults to the
    pi05_b1k_<dataset_name> convention derived from out_dir), and refuses to clobber an
    existing build unless overwrite=True."""
    out = pathlib.Path(out_dir)
    if (out / "meta" / "info.json").exists() and not overwrite:
        info = json.loads((out / "meta" / "info.json").read_text())
        return {"skipped": True, "out_dir": out_dir, "total_episodes": info.get("total_episodes")}
    # Liveness gate runs before ANY rmtree (overwrite path AND partial-dir path) — v1 parity.
    cfg = train_config_name or f"pi05_b1k_{out.name}"
    pids = tools.live_pids(cfg)
    if pids:
        return {"error": f"refusing destructive rebuild while training is live "
                         f"(config={cfg}, pids={pids}): the build rmtree's {out}, "
                         f"which the run may be streaming from"}
    pq = _pq()
    info = json.loads(pathlib.Path(info_template).read_text())
    video_keys = [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")]
    src_tmpl = "data/task-{chunk:04d}/episode_{idx:08d}.parquet"

    def _ids(base):
        return sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                      for p in glob.glob(f"{base}/data/task-0000/episode_*.parquet"))

    def _drops(path):
        return set(json.loads(pathlib.Path(path).read_text())["drop"]) if path else set()

    off_kept = apply_drop_list(_ids(official_dir), sorted(_drops(official_drop_list)))
    per_kept = apply_drop_list(_ids(perturb_dir), sorted(_drops(perturb_drop_list)))
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    episodes, cum, new_idx, truncated, skipped = [], 0, 0, [], []

    def _emit(base, src_idx, start, end):
        nonlocal cum, new_idx
        src_pq = f"{base}/{src_tmpl.format(chunk=0, idx=src_idx)}"
        dst = out / info["data_path"].format(episode_chunk=new_idx // info["chunks_size"],
                                             episode_index=new_idx)
        sliced = slice_and_renumber(src_pq, start, end, new_idx, cum, dst_path=str(dst))
        for vk in video_keys:
            src_v = f"{base}/" + info["video_path"].format(episode_chunk=0, episode_index=src_idx,
                                                           video_key=vk)
            dst_v = str(out / info["video_path"].format(episode_chunk=new_idx // info["chunks_size"],
                                                        episode_index=new_idx, video_key=vk))
            symlink_videos(src_v, dst_v)
        episodes.append({"episode_index": new_idx, "tasks": [prompt], "length": end - start,
                         "stats": _episode_stats(sliced, info)})
        cum += end - start
        new_idx += 1

    for did in off_kept:
        ann = f"{official_dir}/annotations/task-0000/episode_{did:08d}.json"
        s, e = trim_by_skills(f"{official_dir}/{src_tmpl.format(chunk=0, idx=did)}", ann,
                              skills_keep=list(skills_keep))
        _emit(official_dir, did, s, e)
    for pid in per_kept:
        src_pq = f"{perturb_dir}/{src_tmpl.format(chunk=0, idx=pid)}"
        R = pq.read_metadata(src_pq).num_rows
        vmin = R
        for vk in video_keys:  # SHORTEST of the RGB streams caps the episode
            v = video_nframes(f"{perturb_dir}/" + info["video_path"].format(
                episode_chunk=0, episode_index=pid, video_key=vk))
            if v > 0:
                vmin = min(vmin, v)
        if vmin < R:
            truncated.append([pid, R, vmin])
        if vmin < 2:
            skipped.append(pid)
            continue
        for _ in range(dup_factor):
            _emit(perturb_dir, pid, 0, vmin)
    emit_lerobot_meta(str(out), episodes, [prompt], info_template)
    broken = tools.sh(f"find -L {shlex.quote(str(out))} -name '*.mp4' -type l").stdout.strip()
    if broken:
        return {"error": f"broken video symlinks after build:\n{broken[:1200]}"}
    return {"out_dir": str(out), "total_episodes": new_idx, "total_frames": cum,
            "official_kept": len(off_kept), "perturb_kept": len(per_kept),
            "dup_factor": dup_factor, "truncated": len(truncated), "skipped_eps": skipped}


def parse_user_request(run_id: str) -> dict:
    """Return the raw natural-language request plus the RunConfig schema the intake must fill."""
    txt = (tools.run_dir(run_id) / "request.txt").read_text()
    return {"request": txt, "expected_schema": {
        "run_id": "str", "description": "the raw request, verbatim",
        "data_request": {"task_description": "str", "sources": "[SourceSpec: description, "
                         "hf_repo, repo_type, kind(snapshot|single_file_zip), allow_patterns, "
                         "filename, local_dir]", "filter_description": "free-form str"},
        "train_request": {"base_model": "str", "num_train_steps": "int", "batch_size": "int",
                          "peak_lr": "float", "fsdp_devices": "int", "save_interval": "int",
                          "wandb_enabled": "bool"},
        "outputs": {"hf_dataset_repo": "str|null", "hf_model_repo": "str|null"}}}


def write_run_config(run_id: str, config: dict, confidence: dict = None) -> dict:
    """Validate the drafted RunConfig and write runs/<run_id>/config.json. confidence is an
    optional {field: 'explicit'|'inferred'|'confirmed'} audit map persisted as '_confidence'
    (spec section 1 step 3); downstream nodes ignore it."""
    missing = [k for k in ("run_id", "description", "data_request", "train_request", "outputs")
               if k not in config]
    missing += [f"train_request.{k}" for k in ("num_train_steps", "batch_size", "peak_lr",
                                               "fsdp_devices", "save_interval", "wandb_enabled")
                if k not in config.get("train_request", {})]
    if missing:
        return {"error": f"config is missing: {', '.join(missing)}"}
    if confidence:
        config = {**config, "_confidence": confidence}
    p = tools.run_dir(run_id) / "config.json"
    p.write_text(json.dumps(config, indent=2))
    return {"ok": True, "path": str(p)}

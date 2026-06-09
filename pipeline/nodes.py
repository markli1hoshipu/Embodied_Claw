"""Pipeline nodes. Each node is `(state) -> partial state update` per the spec node contracts.

Every node checks artifact existence at entry and returns status "skipped" if its outputs are
already present, which is what makes whole-graph reruns cheap and safe.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from pipeline.state import PipelineState, RunConfig, SourceSpec, StageStatus

# Concrete paths from the spec — do not parameterize away.
OPENPI = Path("/work/markhsp/openpi")
LEROBOT_ROOT = Path("/work/markhsp/datasets/lerobot")
CKPT_ROOT = OPENPI / "checkpoints"
LOGS_DIR = OPENPI / "logs"
PCA_DIR = Path("/work/markhsp/behavior1k-xvla/behavior1k_training/pca_filter")
STAGING_ROOT = Path("/work/markhsp/hf_staging")
TMP_LOG_DIR = Path("/tmp")  # detached-launch logs: /tmp/train_<run_id>.log (redirectable in tests)
FFMPEG_LIB = "/work/markhsp/miniforge3/envs/ffmpeg7/lib"
EXP_NAME = "curated_lr2.5e5"  # ckpt root is checkpoints/<train_config_name>/curated_lr2.5e5/<step>
CONDA = "source /work/markhsp/miniforge3/etc/profile.d/conda.sh && conda activate xvla-stable"

HF_ENV = {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
TRAIN_ENV = {**HF_ENV, "HF_LEROBOT_HOME": "/home/mark-li/.cache/huggingface/lerobot",
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95", "JAX_PLATFORMS": "cuda",
             "NCCL_NVLS_ENABLE": "0", "NCCL_P2P_DISABLE": "0", "NCCL_IB_DISABLE": "0"}

UPLOAD_CACHE = Path.home() / ".cache/huggingface/upload"  # cleared on upload stalls
POLL_SECONDS = 60
TRAIN_TIMEOUT_SECONDS = 24 * 3600  # patient: multi-hour runs (>= 8h required)
MAX_RELAUNCHES = 2

PREFLIGHT_PY = """import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
mesh = Mesh(jax.devices(), ('x',))
y = jax.jit(lambda a: shard_map(lambda a: jax.lax.psum(a,'x'), mesh=mesh, in_specs=P('x'), out_specs=P())(a))(
    jax.device_put(jnp.ones((jax.device_count(),)), NamedSharding(mesh, P('x'))))
assert float(y) == float(jax.device_count())
print('NCCL OK')"""

class TransientHFError(RuntimeError):
    """HF 429 rate limit — retried by the graph-level RetryPolicy on the ingest node."""

def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def _sh(cmd: str, env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=timeout)

def _stage(status: str, started: str | None, error: str | None = None, artifacts=()) -> StageStatus:
    return {"status": status, "started_at": started, "finished_at": _now(),
            "error": error, "artifact_paths": [str(a) for a in artifacts]}

def _require_hf_token(stage: str) -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(f"HF_TOKEN is not set in the environment but the '{stage}' stage needs it. "
                           "Run `export HF_TOKEN=...` and rerun; the token is never hardcoded.")
    return tok

# ---------------------------------------------------------------- predicates (needs_*)

def _source_root(src: SourceSpec) -> Path:
    root = Path(src["local_dir"])
    if src["kind"] == "single_file_zip" and src.get("filename"):
        root = root / Path(src["filename"]).stem
    return root

def _count_kinds(root: Path) -> dict[str, int]:
    """Counts of the spec-mandated file kinds: parquets, mp4s, annotation files."""
    ann = root / "annotations"
    return {"parquets": sum(1 for _ in root.rglob("*.parquet")),
            "mp4s": sum(1 for _ in root.rglob("*.mp4")),
            "annotations": sum(1 for p in ann.rglob("*") if p.is_file()) if ann.is_dir() else 0}

def _manifest_path(root: Path) -> Path:
    return root / ".pipeline_expected.json"

def _read_manifest(root: Path) -> dict | None:
    try:
        return json.loads(_manifest_path(root).read_text())
    except (OSError, json.JSONDecodeError):
        return None

def source_ready(src: SourceSpec) -> bool:
    """Ready when data/ has parquets AND, if an expected-counts manifest was persisted by a prior
    successful ingest, the on-disk parquet/mp4/annotation counts cover those counts ('>=' because
    the official dir is shared across runs and may legitimately grow). A single stray parquet from
    an interrupted download therefore no longer marks the source ready once a manifest exists.
    Data downloaded manually before this pipeline existed has no manifest -> weak check fallback."""
    root = _source_root(src)
    data = root / "data"
    if not (data.is_dir() and any(data.rglob("*.parquet"))):
        return False
    expected = _read_manifest(root)
    if expected is None:
        return True
    have = _count_kinds(root)
    return all(have.get(k, 0) >= v for k, v in expected.items())

def needs_ingest(cfg: RunConfig) -> bool:
    return not all(source_ready(s) for s in cfg["sources"])

def _matched_parquets(local_dir: str, pats: list[str] | None) -> int:
    """Parquets under <local_dir>/data restricted to this run's allow_patterns. The official source
    dir is SHARED across run configs/tasks; counting all of data/ would drift as soon as another
    task is ingested there — and a drifted count arms a destructive rebuild (see needs_build)."""
    root = Path(local_dir)
    rels = (str(p.relative_to(root)) for p in (root / "data").rglob("*.parquet"))
    if not pats:
        return sum(1 for _ in rels)
    return sum(1 for f in rels if any(fnmatch.fnmatch(f, p) for p in pats))

def expected_episodes(cfg: RunConfig) -> int | None:
    """len(official kept) + len(perturb kept) * dup_factor, from source parquet counts minus the
    PCA drop lists. The builder reads TWO drop lists: pca_filter/merged/ (official drops) and
    pca_filter/<run_id>/ (perturb drops). Returns None if anything needed is missing."""
    try:
        snap = next(s for s in cfg["sources"] if s["kind"] == "snapshot")
        zsrc = next(s for s in cfg["sources"] if s["kind"] == "single_file_zip")
        off = _matched_parquets(snap["local_dir"], snap.get("allow_patterns"))
        per = len(list((_source_root(zsrc) / "data").rglob("*.parquet")))
        off_drop = len(json.loads((PCA_DIR / "merged" / "drop_list.json").read_text())["drop"])
        per_drop = len(json.loads((PCA_DIR / cfg["run_id"] / "drop_list.json").read_text())["drop"])
        return (off - off_drop) + (per - per_drop) * cfg["dup_factor"]
    except (StopIteration, OSError, KeyError, json.JSONDecodeError):
        return None

def built_info(cfg: RunConfig) -> dict | None:
    p = LEROBOT_ROOT / cfg["dataset_name"] / "meta" / "info.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None

def needs_build(cfg: RunConfig) -> bool:
    """LOAD-BEARING gate: the builder unconditionally rmtree's its output dir — including
    norm_stats.json — so it must only run when info.json is missing or stale. When the expected
    count cannot be computed we keep the existing build (conservative), but loudly: the spec's
    staleness check is an AND condition and silently skipping it can mask a stale build."""
    info = built_info(cfg)
    if info is None:
        return True
    exp = expected_episodes(cfg)
    if exp is None:
        print(f"[filter_build] WARNING: cannot verify staleness of {cfg['dataset_name']} "
              "(sources/drop lists unreadable) — keeping the existing build unverified", flush=True)
        return False
    return info.get("total_episodes") != exp

def norm_stats_path(cfg: RunConfig) -> Path:
    return LEROBOT_ROOT / cfg["dataset_name"] / "norm_stats.json"

def needs_norm(cfg: RunConfig) -> bool:
    return not norm_stats_path(cfg).is_file()

def ckpt_dir(cfg: RunConfig) -> Path:
    return CKPT_ROOT / cfg["train_config_name"] / EXP_NAME

def final_ckpt(cfg: RunConfig) -> Path:
    return ckpt_dir(cfg) / str(cfg["num_train_steps"] - 1)  # final save lands at N-1 (e.g. 59999)

def train_done(cfg: RunConfig) -> bool:
    f = final_ckpt(cfg)
    return (f / "_CHECKPOINT_METADATA").exists() and (f / "params").is_dir()

def needs_train(cfg: RunConfig) -> bool:
    return not train_done(cfg)

def needs_upload(cfg: RunConfig) -> bool:
    """True unless both HF repos exist AND hold the expected content (model: at least one
    ckpt-*/params file; dataset: meta/info.json). Mere existence is not enough: create_repo runs
    before upload_large_folder, so an interrupted upload leaves a (possibly empty) repo behind —
    a content check keeps the stage re-entrant so only the missing shards re-transfer."""
    try:
        api = _hf_api(os.environ.get("HF_TOKEN"))
        model_files = api.list_repo_files(cfg["hf_model_repo"])
        if not any(f.startswith("ckpt-") and "/params" in f for f in model_files):
            return True
        ds_files = api.list_repo_files(cfg["hf_dataset_repo"], repo_type="dataset")
        return "meta/info.json" not in ds_files
    except Exception:
        return True

def _hf_api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token)

# ---------------------------------------------------------------- train helpers

def find_train_pids(config_name: str) -> list[int]:
    """pgrep for the live training python. Match `python.*scripts/train\\.py.*<config>( |$)` — NOT a
    bare substring: bash watchers/tee carry 'scripts/train.py' in their cmdline (spec failure mode 3),
    and the trailing boundary prevents prefix collisions between config names (recovery vs recovery2/3).
    Raises RuntimeError when pgrep ITSELF fails (rc >= 2: fork failure, bad regex) — rc 1 means
    'no match', rc >= 2 must never be read as 'no process running' (a false [] could double-launch
    the --overwrite train script over a live run). Callers wanting fail-safe use _live_pids()."""
    pat = rf"python.*scripts/train\.py.*{re.escape(config_name)}( |$)"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    if r.returncode >= 2:
        raise RuntimeError(f"pgrep failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()}")
    return [int(x) for x in r.stdout.split()]

def _live_pids(config_name: str) -> list[int]:
    """find_train_pids with the fail-safe direction: a pgrep error reads as 'assume alive'
    (sentinel [-1]) so nothing destructive (launch/rebuild) can fire on an unhandled error path."""
    try:
        return find_train_pids(config_name)
    except RuntimeError as e:
        print(f"[train] WARNING: {e}; assuming the run is alive", flush=True)
        return [-1]

def _tmp_log(cfg: RunConfig) -> Path:
    return TMP_LOG_DIR / f"train_{cfg['run_id']}.log"

def _progress_logs(cfg: RunConfig) -> list[Path]:
    """Newest logs/<config>_*.log (train .sh tees there) + /tmp/train_<run_id>.log (only exists
    when THIS pipeline launched the script; in attach mode only the logs/ file exists)."""
    cands = sorted(LOGS_DIR.glob(cfg["train_config_name"] + "_*.log"))[-1:]
    tmp = _tmp_log(cfg)
    if tmp.exists():
        cands.append(tmp)
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)

def last_progress(cfg: RunConfig) -> str | None:
    """Latest 'Progress on:' / 'Step N:' line. tqdm uses carriage returns, so tr '\\r' '\\n' first."""
    for p in _progress_logs(cfg):
        cmd = f'tail -c 200000 {shlex.quote(str(p))} | tr "\\r" "\\n" | grep -aE "Progress on:|Step [0-9]+:" | tail -1'
        out = _sh(cmd).stdout.strip()
        if out:
            return out
    return None

def _log_marks(cfg: RunConfig) -> dict[str, int]:
    """Log sizes at attach/launch time. Crash classification only reads bytes written AFTER these
    marks, so a stale /tmp/train_<run_id>.log or an old 401 surviving in the tee'd log can never
    trigger an auto-relaunch of a run that died for a different reason."""
    out: dict[str, int] = {}
    for p in _progress_logs(cfg):
        try:
            out[str(p)] = p.stat().st_size
        except OSError:
            pass
    return out

def _log_tail(cfg: RunConfig, marks: dict[str, int] | None = None) -> str:
    """Last 8000 bytes of each candidate log (only bytes past the mark when marks given), \\r->\\n."""
    parts = []
    for p in _progress_logs(cfg):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        start = (marks or {}).get(str(p), 0)
        if marks is not None and size <= start:
            continue  # nothing new since attach/launch — stale content is not evidence
        with open(p, "rb") as f:
            f.seek(max(start, size - 8000))
            parts.append(f.read().decode("utf-8", "replace").replace("\r", "\n"))
    return "\n".join(parts)

def _is_nccl_401(text: str) -> bool:
    t = text.lower()
    return "cuda failure 401" in t or "nvls.cc" in t

def latest_step(cfg: RunConfig) -> int | None:
    """Newest intermediate step dir under the ckpt dir — the spec poll loop's second progress
    signal, independent of log Progress lines."""
    steps = [int(p.name) for p in ckpt_dir(cfg).glob("[0-9]*") if p.is_dir() and p.name.isdigit()]
    return max(steps, default=None)

_ALWAYS_EXPORTS = (f'export PATH="$HOME/.local/bin:$PATH" && '
                   f"export LD_LIBRARY_PATH={FFMPEG_LIB}:${{LD_LIBRARY_PATH:-}}")

def _preflight() -> str | None:
    """8-GPU psum check before burning hours. Returns an error string on failure, None if OK."""
    r = _sh(f"cd {OPENPI} && {_ALWAYS_EXPORTS} && uv run python -c {shlex.quote(PREFLIGHT_PY)}",
            env=TRAIN_ENV, timeout=600)
    if r.returncode != 0 or "NCCL OK" not in r.stdout:
        return ("multi-GPU psum pre-flight failed — check the NCCL_NVLS_ENABLE=0 workaround "
                "(project_new_server_nccl_nvls.md): " + (r.stderr or r.stdout)[-800:])
    return None

def _launch(cfg: RunConfig) -> None:
    """Detached launch — mandatory pattern; the run outlives this process. Re-checks liveness
    IMMEDIATELY before launching (TOCTOU guard: the pre-flight can take minutes, and the train .sh
    passes --overwrite, so a duplicate launch would wipe an in-flight run). After launching,
    confirms the log file appears — `setsid ... & disown` exits 0 even for a bad script path."""
    pids = _live_pids(cfg["train_config_name"])
    if pids:
        raise RuntimeError(f"refusing to launch {cfg['train_script']}: live train process {pids} "
                           f"matches {cfg['train_config_name']} (--overwrite would wipe it)")
    log = _tmp_log(cfg)
    cmd = (f"cd {OPENPI} && {_ALWAYS_EXPORTS} && setsid bash {shlex.quote(cfg['train_script'])} "
           f"</dev/null >{shlex.quote(str(log))} 2>&1 & disown")
    r = _sh(cmd, env=TRAIN_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"failed to launch {cfg['train_script']}: {(r.stderr or r.stdout)[-500:]}")
    for _ in range(10):
        if log.exists():
            return
        time.sleep(0.5)
    raise RuntimeError(f"train script did not start: no log at {log} after launch")

# ---------------------------------------------------------------- nodes

def _raise_if_429(tail: str, src: SourceSpec) -> None:
    if "429" in tail:
        raise TransientHFError(f"HF 429 rate limit on {src['hf_repo']}: {tail[-300:]}")

def ingest_source(state: PipelineState) -> dict:
    """Download each SourceSpec: 'snapshot' via snapshot_download(allow_patterns, local_dir),
    'single_file_zip' via hf_hub_download + unzip -q -o. Runs inside conda env xvla-stable.
    HF_TOKEN from env; HF_HUB_DISABLE_XET=1. Raises TransientHFError on 429 (graph retries, 5x
    exponential backoff) — on the first attempt AND on the per-file fallback used when
    snapshot_download chokes. Skips if local files exist. End-of-node verification: local
    parquet/mp4/annotation counts must cover the repo file listing (snapshot) and the counts are
    persisted so later runs detect partially-downloaded trees instead of skipping them."""
    cfg, started = state["config"], _now()
    if not needs_ingest(cfg):
        return {"ingest": _stage("skipped", started, artifacts=[s["local_dir"] for s in cfg["sources"]])}
    try:
        token = _require_hf_token("ingest")
    except RuntimeError as e:
        return {"ingest": _stage("failed", started, error=str(e))}
    env = {**HF_ENV, "HF_TOKEN": token}
    arts = []
    for src in cfg["sources"]:
        if source_ready(src):
            arts.append(src["local_dir"])
            continue
        r = _sh(_ingest_cmd(src), env=env, timeout=6 * 3600)
        if r.returncode != 0:
            tail = ((r.stderr or "") + (r.stdout or ""))[-2000:]
            _raise_if_429(tail, src)
            if src["kind"] != "snapshot":
                return {"ingest": _stage("failed", started, error=tail)}
            r = _sh(_fallback_cmd(src), env=env, timeout=6 * 3600)  # per-file fallback
            if r.returncode != 0:
                tail = ((r.stderr or "") + (r.stdout or ""))[-2000:]  # report the FALLBACK's error
                _raise_if_429(tail, src)  # fallback 429s must reach the graph RetryPolicy too
                return {"ingest": _stage("failed", started, error=tail)}
        err = _verify_and_record(src, token)
        if err:
            return {"ingest": _stage("failed", started, error=err)}
        arts.append(src["local_dir"])
    return {"ingest": _stage("succeeded", started, artifacts=arts)}

def _verify_and_record(src: SourceSpec, token: str) -> str | None:
    """Post-download verification (spec: 'counts match expected'). snapshot: expected counts come
    from the repo-side file listing filtered by allow_patterns; fail if the local tree is short.
    single_file_zip: `unzip -q -o` exited 0 so the archive extracted completely — pin the actual
    counts. Either way the expected counts are persisted next to the data (.pipeline_expected.json)
    so source_ready() compares against them on subsequent runs."""
    root = _source_root(src)
    have = _count_kinds(root)
    if src["kind"] == "snapshot":
        try:
            files = _hf_api(token).list_repo_files(src["hf_repo"], repo_type=src["repo_type"])
        except Exception as e:  # noqa: BLE001 — classify rate limits, surface the rest
            if "429" in str(e):
                raise TransientHFError(f"HF 429 rate limit on {src['hf_repo']}: {str(e)[-300:]}") from e
            return f"{src['hf_repo']}: cannot list repo files to verify the download: {str(e)[-500:]}"
        pats = src.get("allow_patterns")
        kept = [f for f in files if not pats or any(fnmatch.fnmatch(f, p) for p in pats)]
        expected = {"parquets": sum(f.endswith(".parquet") for f in kept),
                    "mp4s": sum(f.endswith(".mp4") for f in kept),
                    "annotations": sum(f.startswith("annotations/") for f in kept)}
        short = {k: (have.get(k, 0), v) for k, v in expected.items() if have.get(k, 0) < v}
        if short:
            return (f"{src['hf_repo']}: download incomplete under {root} — "
                    + ", ".join(f"{k} {h}/{v}" for k, (h, v) in short.items()))
    else:
        if have["parquets"] == 0:
            return f"{src['hf_repo']}: unzip finished but no parquets under {root}"
        expected = have
    try:
        _manifest_path(root).write_text(json.dumps(expected))
    except OSError as e:
        return f"cannot persist expected-counts manifest under {root}: {e}"
    return None

def _ingest_cmd(src: SourceSpec) -> str:
    if src["kind"] == "snapshot":
        py = (f"from huggingface_hub import snapshot_download; snapshot_download(repo_id={src['hf_repo']!r}, "
              f"repo_type={src['repo_type']!r}, allow_patterns={src.get('allow_patterns')!r}, local_dir={src['local_dir']!r})")
        return f"{CONDA} && python -c {shlex.quote(py)}"
    py = (f"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id={src['hf_repo']!r}, "
          f"repo_type={src['repo_type']!r}, filename={src['filename']!r}, local_dir={src['local_dir']!r})")
    return (f"{CONDA} && python -c {shlex.quote(py)} && "
            f"cd {shlex.quote(src['local_dir'])} && unzip -q -o {shlex.quote(src['filename'])}")

def _fallback_cmd(src: SourceSpec) -> str:
    py = ("import fnmatch\nfrom huggingface_hub import HfApi, hf_hub_download\n"
          f"files = HfApi().list_repo_files({src['hf_repo']!r}, repo_type={src['repo_type']!r})\n"
          f"pats = {src.get('allow_patterns')!r}\n"
          "for f in files:\n"
          "    if pats and not any(fnmatch.fnmatch(f, p) for p in pats): continue\n"
          f"    hf_hub_download(repo_id={src['hf_repo']!r}, repo_type={src['repo_type']!r}, filename=f, local_dir={src['local_dir']!r})\n")
    return f"{CONDA} && python -c {shlex.quote(py)}"

def filter_and_build(state: PipelineState) -> dict:
    """Run builder_script (python3, conda xvla-stable): applies PCA drop lists, slices parquets,
    symlinks videos, writes meta/*. Verifies info.json total_episodes == expected and that no
    video symlink is broken. Skips if info.json exists with matching episode count — the builder
    is destructive (rmtree of the output dir incl. norm_stats.json), so the gate is load-bearing."""
    cfg, started = state["config"], _now()
    out = LEROBOT_ROOT / cfg["dataset_name"]
    if not needs_build(cfg):
        info = built_info(cfg)
        return {"filter_build": _stage("skipped", started, artifacts=[
            out, f"episodes={info['total_episodes']} frames={info.get('total_frames')} videos={info.get('total_videos')}"])}
    pids = _live_pids(cfg["train_config_name"])
    if pids:
        return {"filter_build": _stage("failed", started, error=(
            f"refusing destructive rebuild while training is live (pids={pids}): the builder "
            f"rmtree's {out}, which the running job may be streaming from right now"))}
    r = _sh(f"{CONDA} && cd {OPENPI} && python3 {shlex.quote(cfg['builder_script'])}", timeout=4 * 3600)
    if r.returncode != 0:
        return {"filter_build": _stage("failed", started, error=((r.stderr or "") + (r.stdout or ""))[-2000:])}
    info = built_info(cfg)
    if info is None:
        return {"filter_build": _stage("failed", started, error=f"builder exited 0 but {out}/meta/info.json is missing")}
    exp = expected_episodes(cfg)
    if exp is not None and info["total_episodes"] != exp:
        return {"filter_build": _stage("failed", started,
                                       error=f"total_episodes={info['total_episodes']} != expected {exp}")}
    broken = _sh(f"find -L {shlex.quote(str(out))} -name '*.mp4' -type l").stdout.strip()
    if broken:
        return {"filter_build": _stage("failed", started, error=f"broken video symlinks:\n{broken[:1500]}")}
    return {"filter_build": _stage("succeeded", started, artifacts=[
        out, f"episodes={info['total_episodes']} frames={info.get('total_frames')} videos={info.get('total_videos')}"])}

def compute_norm_stats(state: PipelineState) -> dict:
    """`uv run python scripts/compute_norm_stats.py --config-name <cfg> --max-frames 50000` from
    /work/markhsp/openpi (fast sampled path; full pass takes 60+ min). Needs LD_LIBRARY_PATH for
    torchcodec/ffmpeg7 and HF_LEROBOT_HOME. Skips if norm_stats.json exists. Verifies size > 1 KB."""
    cfg, started = state["config"], _now()
    p = norm_stats_path(cfg)
    if not needs_norm(cfg):
        return {"norm_stats": _stage("skipped", started, artifacts=[p])}
    env = {"HF_LEROBOT_HOME": "/home/mark-li/.cache/huggingface/lerobot",
           "LD_LIBRARY_PATH": "/work/markhsp/miniforge3/envs/ffmpeg7/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    cmd = (f'cd {OPENPI} && export PATH="$HOME/.local/bin:$PATH" && '
           f"uv run python scripts/compute_norm_stats.py --config-name {cfg['train_config_name']} --max-frames 50000")
    r = _sh(cmd, env=env, timeout=2 * 3600)
    if r.returncode != 0:
        return {"norm_stats": _stage("failed", started, error=((r.stderr or "") + (r.stdout or ""))[-2000:])}
    if not p.is_file() or p.stat().st_size <= 1024:
        return {"norm_stats": _stage("failed", started, error=f"{p} missing or <= 1 KB after compute")}
    return {"norm_stats": _stage("succeeded", started, artifacts=[p])}

def train(state: PipelineState) -> dict:
    """Detached `setsid bash <train_script>` + 60s poll loop on logs/ckpt dir. Skips if the final
    step dir (num_train_steps-1) exists. ATTACH mode: if a matching `python.*scripts/train.py`
    process is already alive, never launch (the script passes --overwrite, which would wipe the
    in-flight run) — go straight to polling. Pre-flight psum check only before a fresh launch.
    Crash triage: NCCL CUDA-401/nvls in bytes written AFTER attach/launch -> relaunch
    (<= MAX_RELAUNCHES, with a liveness re-check inside _launch); anything else -> fail."""
    cfg, started = state["config"], _now()
    if not needs_train(cfg):
        return {"train": _stage("skipped", started, artifacts=[final_ckpt(cfg)])}
    pids = _live_pids(cfg["train_config_name"])
    if pids:
        print(f"[train] ATTACH: live process {pids} for {cfg['train_config_name']}; skipping launch "
              "(duplicate launch would --overwrite the in-flight run)", flush=True)
    else:
        err = _preflight()
        if err:
            return {"train": _stage("failed", started, error=err)}
        try:
            _launch(cfg)  # re-checks liveness internally right before launching (TOCTOU guard)
        except RuntimeError as e:
            if "refusing to launch" not in str(e):
                return {"train": _stage("failed", started, error=str(e))}
            print(f"[train] {e}; attaching to it instead", flush=True)
    # Classify crashes only from log bytes written after this point — stale /tmp logs or old
    # 401s already in the tee'd log must not look like a fresh NCCL crash.
    marks = _log_marks(cfg)
    relaunches = 0
    deadline = time.time() + TRAIN_TIMEOUT_SECONDS
    while time.time() < deadline:
        if train_done(cfg):
            return {"train": _stage("succeeded", started, artifacts=[final_ckpt(cfg)])}
        print(f"[train] {_now()} ckpt_step={latest_step(cfg)} "  # heartbeat: ckpt dir + log progress
              f"{last_progress(cfg) or 'no Progress lines yet'}", flush=True)
        if not _live_pids(cfg["train_config_name"]):
            time.sleep(5)  # grace for the final checkpoint flush
            if train_done(cfg):
                return {"train": _stage("succeeded", started, artifacts=[final_ckpt(cfg)])}
            tail = _log_tail(cfg, marks)
            if _is_nccl_401(tail) and relaunches < MAX_RELAUNCHES:
                relaunches += 1
                print(f"[train] NCCL CUDA-401/NVLS crash; relaunch {relaunches}/{MAX_RELAUNCHES} "
                      "(NCCL_NVLS_ENABLE=0 already exported — known to be flaky anyway)", flush=True)
                try:
                    _launch(cfg)  # internal liveness re-check: never duplicate-launch
                except RuntimeError as e:
                    if "refusing to launch" not in str(e):
                        return {"train": _stage("failed", started, error=str(e))}
                    print(f"[train] {e}; resuming polling", flush=True)
                marks = _log_marks(cfg)
            else:
                return {"train": _stage("failed", started,
                                        error="train process died before final checkpoint:\n"
                                              + (tail or _log_tail(cfg))[-1500:])}
        time.sleep(POLL_SECONDS)
    return {"train": _stage("failed", started, error=f"poll loop timed out after {TRAIN_TIMEOUT_SECONDS}s")}

def upload_to_hf(state: PipelineState) -> dict:
    """Hardlink-stage ckpt steps {30000,40000,50000,final} (59999 staged as ckpt-60000) under
    /work/markhsp/hf_staging/<run_id>/ (per-run subdir: deliberate deviation from the spec's flat
    staging root for multi-run isolation; the resulting repo layout is identical), upload_large_folder
    (train_state/assets excluded), inject norm_stats.json into each ckpt-*/assets/, then upload the
    dataset folder (symlinked videos resolve to real bytes). hf_transfer disabled; an upload that
    FAILS with an exception clears the resumable-upload cache and retries (a silent hang needs the
    manual kill+clear+rerun procedure — see README); HF dedups by SHA so reruns are safe. Skips only
    if both repos already hold the expected content. End-of-node verification: repos respond and the
    first uploaded file's hub SHA matches the local bytes."""
    cfg, started = state["config"], _now()
    urls = [f"https://huggingface.co/{cfg['hf_model_repo']}",
            f"https://huggingface.co/datasets/{cfg['hf_dataset_repo']}"]
    if not needs_upload(cfg):
        return {"upload": _stage("skipped", started, artifacts=urls)}
    try:
        token = _require_hf_token("upload")
    except RuntimeError as e:
        return {"upload": _stage("failed", started, error=str(e))}
    saved = {k: os.environ.get(k) for k in HF_ENV}
    os.environ.update(HF_ENV)  # hf_transfer off (stalls), Xet off (429s) — restored in finally
    try:
        staging = STAGING_ROOT / cfg["run_id"]
        staging.mkdir(parents=True, exist_ok=True)
        staged = []
        for want in (30000, 40000, 50000, cfg["num_train_steps"]):
            for disk in (want, want - 1):  # final save is N-1 -> rounded staging name ckpt-N
                src = ckpt_dir(cfg) / str(disk)
                if src.is_dir():
                    dst = staging / f"ckpt-{want}"
                    if dst.exists():
                        shutil.rmtree(dst)  # stale staging from a prior attempt — re-hardlink fresh
                    r = _sh(f"cp -al {shlex.quote(str(src))} {shlex.quote(str(dst))}")
                    if r.returncode != 0:
                        return {"upload": _stage("failed", started, error=f"cp -al failed: {r.stderr[-800:]}")}
                    staged.append(dst)
                    break
        if not staged:
            return {"upload": _stage("failed", started, error=f"no step dirs found under {ckpt_dir(cfg)}")}
        try:
            api = _hf_api(token)
            api.create_repo(cfg["hf_model_repo"], repo_type="model", exist_ok=True)
            _upload_with_stall_retry(api, folder_path=str(staging), repo_id=cfg["hf_model_repo"], repo_type="model",
                                     ignore_patterns=["ckpt-*/train_state/*", "ckpt-*/train_state/**", "ckpt-*/assets/*"])
            for d in staged:
                api.upload_file(path_or_fileobj=str(norm_stats_path(cfg)),
                                path_in_repo=f"{d.name}/assets/norm_stats.json", repo_id=cfg["hf_model_repo"])
            api.create_repo(cfg["hf_dataset_repo"], repo_type="dataset", exist_ok=True)
            _upload_with_stall_retry(api, folder_path=str(LEROBOT_ROOT / cfg["dataset_name"]),
                                     repo_id=cfg["hf_dataset_repo"], repo_type="dataset")
            err = _verify_uploads(api, cfg, staged)
            if err:
                return {"upload": _stage("failed", started, error=err)}
        except Exception as e:  # noqa: BLE001 — surface any hub error as a failed stage
            return {"upload": _stage("failed", started, error=str(e)[-2000:])}
        return {"upload": _stage("succeeded", started, artifacts=urls)}
    finally:
        for k, v in saved.items():  # don't leak HF_ENV into the host process / other stages
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _upload_with_stall_retry(api, **kw) -> None:
    """Retry on RAISED upload failures, clearing the resumable-upload cache between attempts
    (server-side SHAs dedup what already landed). NOTE: this cannot catch spec failure mode 8's
    silent final-shard HANG — that needs the manual procedure documented in the README."""
    for attempt in range(3):
        try:
            api.upload_large_folder(**kw)
            return
        except Exception:
            if attempt == 2:
                raise
            shutil.rmtree(UPLOAD_CACHE, ignore_errors=True)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

def _hub_sha_matches(api, repo_id: str, repo_type: str, rel: str, local: Path) -> str | None:
    """Compare the hub-side hash of one uploaded file against the local bytes (LFS sha256 for
    large files, git blob sha1 for regular blobs). Returns an error string on mismatch/absence."""
    try:
        infos = api.get_paths_info(repo_id, [rel], repo_type=repo_type)
    except Exception as e:  # noqa: BLE001
        return f"{repo_id}: get_paths_info({rel}) failed: {str(e)[-300:]}"
    if not infos:
        return f"{repo_id}: {rel} missing on the Hub after upload"
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    hub_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    if hub_sha:
        return None if hub_sha == _sha256(local) else f"{repo_id}: {rel} sha256 mismatch (hub != local)"
    blob = getattr(info, "blob_id", None)
    if blob:
        return None if blob == _git_blob_sha1(local) else f"{repo_id}: {rel} git-blob sha mismatch (hub != local)"
    return None  # hub exposed no hash for this path; existence was still verified

def _verify_uploads(api, cfg: RunConfig, staged: list[Path]) -> str | None:
    """End-of-node verification (spec): repos respond AND the first uploaded file's SHA matches the
    local file — this is the guard that catches an interrupted upload being marked done."""
    api.model_info(cfg["hf_model_repo"])
    first = next((p for p in sorted(staged[0].rglob("*"))
                  if p.is_file() and "train_state" not in p.parts and "assets" not in p.parts), None)
    if first is not None:
        rel = f"{staged[0].name}/{first.relative_to(staged[0])}"
        err = _hub_sha_matches(api, cfg["hf_model_repo"], "model", rel, first)
        if err:
            return err
    info_json = LEROBOT_ROOT / cfg["dataset_name"] / "meta" / "info.json"
    if info_json.is_file():
        return _hub_sha_matches(api, cfg["hf_dataset_repo"], "dataset", "meta/info.json", info_json)
    return None

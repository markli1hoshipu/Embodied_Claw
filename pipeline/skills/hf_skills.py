"""hf_agent skills (spec section 5): naming, 3-state repo probe, hardlink staging (want-1
rounding), resilient upload (v1 clear-cache retry + new stall watchdog), SHA confirmation."""
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from pipeline import tools


def lookup_past_naming_conventions(kind: str = "any") -> dict:
    """Past Hoshipu/ repo names from hf_agent long-term memory (naming conventions)."""
    mem = tools.agents_root() / "hf_agent" / "memory.md"
    examples = re.findall(r"Hoshipu/[\w.-]+", mem.read_text()) if mem.exists() else []
    return {"examples": sorted(set(examples)), "conventions": {
        "dataset": "Hoshipu/b1k_<task>_<variant>",
        "model": "Hoshipu/pi05-b1kt0-<variant>-lr<peak_lr>"}}


def propose_repo_name(kind: str, description: str, past_naming_examples: list = ()) -> dict:
    """Heuristic candidates following past conventions; final say is the agent's + the user's."""
    slug = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")[:40]
    if kind == "dataset":
        props = [f"Hoshipu/b1k_{slug}", f"Hoshipu/b1k_{slug}_curated"]
    else:
        props = [f"Hoshipu/pi05-b1kt0-{slug.replace('_', '')[:24]}-lr2.5e5"]
    return {"proposals": props, "reasoning": "slugged from the run description following "
            "conventions; check against past examples", "past_examples": list(past_naming_examples)}


def verify_repo_doesnt_exist_or_confirm_overwrite(repo_id: str, repo_type: str) -> dict:
    """3-state probe: 'new' (no repo) | 'exists_safe' (repo present but no conflicting payload —
    an interrupted upload's empty shell; safe to fill, HF dedups by SHA) | 'exists_conflict'
    (already holds checkpoint params / meta/info.json — needs user confirmation to overwrite)."""
    api = tools.hf_api()
    try:
        files = api.list_repo_files(repo_id, repo_type=repo_type)
    except Exception as e:  # noqa: BLE001 — hub 404s surface as RepositoryNotFoundError etc.
        if "404" in str(e) or "Not Found" in str(e) or type(e).__name__ == "RepositoryNotFoundError":
            return {"status": "new", "n_files": 0}
        return {"error": f"{repo_id}: cannot probe: {str(e)[-400:]}"}
    if repo_type == "model":
        conflict = any(f.startswith("ckpt-") and "/params" in f for f in files)
    else:
        conflict = "meta/info.json" in files
    return {"status": "exists_conflict" if conflict else "exists_safe", "n_files": len(files)}


def hardlink_stage_checkpoints(ckpt_dir: str, staging_dir: str, steps_to_upload: list) -> dict:
    """`cp -al` selected step dirs to <staging_dir>/ckpt-<step>. The final save lands at N-1 on
    disk, so each step tries (want, want-1) and stages under the rounded name (59999->ckpt-60000).
    Stale staging from a prior attempt is re-hardlinked fresh."""
    cd, staging = pathlib.Path(ckpt_dir), pathlib.Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    staged, missing = [], []
    for want in steps_to_upload:
        for disk in (int(want), int(want) - 1):
            src = cd / str(disk)
            if src.is_dir():
                dst = staging / f"ckpt-{int(want)}"
                if dst.exists():
                    shutil.rmtree(dst)
                r = tools.sh(f"cp -al {shlex.quote(str(src))} {shlex.quote(str(dst))}")
                if r.returncode != 0:
                    return {"error": f"cp -al failed for {src}: {r.stderr[-800:]}"}
                staged.append(str(dst))
                break
        else:
            missing.append(int(want))
    if not staged:
        return {"error": f"no step dirs found under {cd} for steps {list(steps_to_upload)}"}
    return {"staged": staged, "missing_steps": missing}


def upload_large_folder_resilient(folder_path: str, repo_id: str, repo_type: str,
                                  ignore_patterns: list = None, attempts: int = 3,
                                  stall_minutes: float = 15, poll_s: float = 10) -> dict:
    """upload_large_folder in a child process with a no-progress watchdog (spec section 9 row 6:
    the silent final-shard hang). Progress signal = child stdout bytes + resumable-cache mtime;
    on stall or raised failure: kill the process group, clear ~/.cache/huggingface/upload, retry
    (server-side SHAs dedup what already landed). hf_transfer/Xet stay disabled."""
    tools.require_hf_token("upload")
    api = tools.hf_api()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    py = (f"from huggingface_hub import HfApi; HfApi().upload_large_folder(folder_path="
          f"{folder_path!r}, repo_id={repo_id!r}, repo_type={repo_type!r}, "
          f"ignore_patterns={ignore_patterns!r})")
    env = {**os.environ, **tools.HF_ENV}
    for attempt in range(1, attempts + 1):
        log = pathlib.Path(folder_path) / f".upload_attempt_{attempt}.log"
        with open(log, "wb") as lf:
            proc = subprocess.Popen([sys.executable, "-c", py], stdout=lf, stderr=lf,
                                    env=env, start_new_session=True)
        last_sig, last_change = None, time.time()
        stalled = False
        while proc.poll() is None:
            time.sleep(poll_s)
            cache_mtime = max((p.stat().st_mtime for p in tools.UPLOAD_CACHE.rglob("*")
                               if p.is_file()), default=0) if tools.UPLOAD_CACHE.is_dir() else 0
            sig = (log.stat().st_size if log.exists() else 0, cache_mtime)
            if sig != last_sig:
                last_sig, last_change = sig, time.time()
            elif time.time() - last_change > stall_minutes * 60:
                os.killpg(proc.pid, signal.SIGKILL)
                stalled = True
                break
        if not stalled and proc.returncode == 0:
            return {"ok": True, "attempts": attempt}
        shutil.rmtree(tools.UPLOAD_CACHE, ignore_errors=True)  # clear resumable-upload cache
    return {"ok": False, "error": f"upload of {folder_path} to {repo_id} failed/stalled "
                                  f"{attempts}x (cache cleared between attempts) — escalate"}


def upload_norm_stats_per_checkpoint(staging_dir: str, norm_stats_path: str, repo_id: str) -> dict:
    """Inject norm_stats.json as <ckpt-N>/assets/norm_stats.json for every staged step."""
    api, out = tools.hf_api(), {}
    for d in sorted(pathlib.Path(staging_dir).glob("ckpt-*")):
        api.upload_file(path_or_fileobj=str(norm_stats_path),
                        path_in_repo=f"{d.name}/assets/norm_stats.json", repo_id=repo_id)
        out[d.name] = "ok"
    if not out:
        return {"error": f"no ckpt-* dirs under {staging_dir}"}
    return {"uploaded": out}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _hub_sha_matches(api, repo_id: str, repo_type: str, rel: str, local: pathlib.Path):
    """Hub-side hash vs local bytes (LFS sha256 for large files, git blob sha1 otherwise)."""
    infos = api.get_paths_info(repo_id, [rel], repo_type=repo_type)
    if not infos:
        return f"{rel} missing on the Hub"
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    hub = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    if hub:
        return None if hub == _sha256(local) else f"{rel}: sha256 mismatch (hub != local)"
    blob = getattr(info, "blob_id", None)
    if blob:
        return None if blob == _git_blob_sha1(local) else f"{rel}: git-blob sha mismatch"
    return None  # hub exposed no hash; existence already verified


def confirm_upload_complete(repo_id: str, repo_type: str, expected_files: list,
                            local_root: str = None) -> dict:
    """repo listing covers expected_files; first 3 present files SHA-match local bytes when
    local_root is given (catches an interrupted upload being marked done)."""
    api = tools.hf_api()
    try:
        files = set(api.list_repo_files(repo_id, repo_type=repo_type))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "missing": list(expected_files), "error": str(e)[-400:]}
    missing = [f for f in expected_files if f not in files]
    sha_errors = []
    if local_root:
        for rel in [f for f in expected_files if f in files][:3]:
            local = pathlib.Path(local_root) / rel
            if local.is_file():
                err = _hub_sha_matches(api, repo_id, repo_type, rel, local)
                if err:
                    sha_errors.append(err)
    return {"ok": not missing and not sha_errors, "missing": missing, "sha_errors": sha_errors}

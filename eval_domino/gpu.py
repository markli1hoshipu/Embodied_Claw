"""GPU lease manager. Lease = atomic O_EXCL file gpu_leases/gpu<N>.lease holding
{run_id, pid, acquired_at}. Stale (owner pid dead) leases are reaped on every snapshot, so a
crashed driver can never wedge a GPU. Free = no live lease AND enough memory headroom (other
users' trainings show up as used memory, not leases)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline.tools import REPO, now, sh

LEASE_ROOT = Path(os.environ.get("EVAL_GPU_LEASES", str(REPO / "gpu_leases")))
DEFAULT_MIN_FREE_MB = 40_000


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def _lease_path(gpu_id: int) -> Path:
    return LEASE_ROOT / f"gpu{gpu_id}.lease"


def reap_stale() -> list[int]:
    reaped = []
    for p in LEASE_ROOT.glob("gpu*.lease") if LEASE_ROOT.is_dir() else ():
        try:
            if not _pid_alive(json.loads(p.read_text()).get("pid")):
                p.unlink()
                reaped.append(int(p.stem[3:]))
        except (json.JSONDecodeError, OSError, ValueError):
            p.unlink(missing_ok=True)
    return reaped


def snapshot() -> list[dict]:
    """[{id, mem_free_mb, leased_by|None}] for every GPU, stale leases reaped first."""
    reap_stale()
    r = sh("nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits")
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {r.stderr.strip()}")
    out = []
    for line in r.stdout.strip().splitlines():
        idx, total, used = [int(x.strip()) for x in line.split(",")]
        lp = _lease_path(idx)
        leased_by = None
        if lp.exists():
            try:
                leased_by = json.loads(lp.read_text()).get("run_id")
            except (json.JSONDecodeError, OSError):
                leased_by = "?"
        out.append({"id": idx, "mem_free_mb": total - used, "leased_by": leased_by})
    return out


def free_gpus(min_free_mb: int = DEFAULT_MIN_FREE_MB) -> list[int]:
    return [g["id"] for g in snapshot()
            if g["leased_by"] is None and g["mem_free_mb"] >= min_free_mb]


def acquire(run_id: str, count: int, min_free_mb: int = DEFAULT_MIN_FREE_MB) -> list[int]:
    """Lease up to `count` free GPUs atomically (O_EXCL per file — concurrent runs cannot
    double-lease). Returns leased ids; caller decides what to do if fewer than requested."""
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    got: list[int] = []
    payload = {"run_id": run_id, "pid": os.getpid(), "acquired_at": now()}
    for gid in free_gpus(min_free_mb):
        if len(got) >= count:
            break
        try:
            fd = os.open(_lease_path(gid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        got.append(gid)
    return got


def release(run_id: str) -> list[int]:
    freed = []
    for p in LEASE_ROOT.glob("gpu*.lease") if LEASE_ROOT.is_dir() else ():
        try:
            if json.loads(p.read_text()).get("run_id") == run_id:
                p.unlink()
                freed.append(int(p.stem[3:]))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return freed
